import os
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

# ----- config -----
# EIA key is read from the environment or Streamlit secrets — never hardcoded.
# Local:  export EIA_API_KEY="your_key"
# Cloud:  set EIA_API_KEY in the deployment's secrets / env vars.
def _get_eia_key():
    k = os.environ.get("EIA_API_KEY")
    if k:
        return k
    try:
        return st.secrets["EIA_API_KEY"]
    except Exception:
        return ""

EIA_API_KEY = _get_eia_key()
LAT, LON = 32.7767, -96.7970               # Dallas
TZ = "America/Chicago"
BASE_C = 18.0
FEATURE_COLS = [
    "temp_c", "humidity", "wind_speed", "cooling_degrees", "heating_degrees",
    "demand_lag_24", "demand_lag_48", "demand_lag_168",
    "demand_roll_mean_24", "demand_roll_std_24",
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]

if not EIA_API_KEY:
    st.error("No EIA API key found. Set the EIA_API_KEY environment variable "
             "(get a free key at https://www.eia.gov/opendata/register.php).")
    st.stop()


@st.cache_resource(show_spinner="Building the forecasting pipeline (first load only)…")
def build():
    from sklearn.ensemble import HistGradientBoostingRegressor

    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(days=730)).strftime("%Y-%m-%dT%H")     # ~2 years
    end   = (now - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H")
    wx_end = (now - pd.Timedelta(days=6)).strftime("%Y-%m-%d")          # archive trails ~5d

    # --- demand (EIA, paginated) ---
    rows, offset = [], 0
    while True:
        r = requests.get("https://api.eia.gov/v2/electricity/rto/region-data/data/",
            params={"api_key": EIA_API_KEY, "frequency": "hourly", "data[0]": "value",
                    "facets[respondent][]": "ERCO", "facets[type][]": "D",
                    "start": start, "end": end, "sort[0][column]": "period",
                    "sort[0][direction]": "asc", "offset": offset, "length": 5000},
            timeout=60)
        r.raise_for_status()
        payload = r.json()["response"]; batch = payload["data"]
        rows.extend(batch); offset += 5000
        if offset >= int(payload["total"]) or not batch:
            break
    dem = pd.DataFrame(rows)
    dem["period_utc"] = pd.to_datetime(dem["period"], utc=True)
    dem["demand_mwh"] = pd.to_numeric(dem["value"], errors="coerce")
    dem = dem[["period_utc", "demand_mwh"]]

    # --- weather (Open-Meteo archive, keyless) ---
    wr = requests.get("https://archive-api.open-meteo.com/v1/archive",
        params={"latitude": LAT, "longitude": LON, "start_date": start[:10],
                "end_date": wx_end, "timezone": "UTC",
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"},
        timeout=60)
    wr.raise_for_status(); h = wr.json()["hourly"]
    wx = pd.DataFrame({"period_utc": pd.to_datetime(h["time"], utc=True),
                       "temp_c": h["temperature_2m"], "humidity": h["relative_humidity_2m"],
                       "wind_speed": h["wind_speed_10m"]})

    # --- join + calendar + clean ---
    df = dem.merge(wx, on="period_utc", how="inner").dropna().sort_values("period_utc")
    loc = df["period_utc"].dt.tz_convert(TZ)
    df["period_local"] = loc
    df["hour"] = loc.dt.hour; df["dayofweek"] = loc.dt.dayofweek
    df["month"] = loc.dt.month; df["day_of_year"] = loc.dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    try:
        import holidays
        us = holidays.US(years=range(loc.dt.year.min(), loc.dt.year.max() + 1))
        df["is_holiday"] = loc.dt.date.isin(us).astype(int)
    except Exception:
        df["is_holiday"] = 0
    clean = df.reset_index(drop=True)

    # --- features (leakage-safe) ---
    f = clean.copy()
    for lag in [24, 48, 168]:
        f[f"demand_lag_{lag}"] = f["demand_mwh"].shift(lag)
    past = f["demand_mwh"].shift(1)
    f["demand_roll_mean_24"] = past.rolling(24).mean()
    f["demand_roll_std_24"]  = past.rolling(24).std()
    f["cooling_degrees"] = (f["temp_c"] - BASE_C).clip(lower=0)
    f["heating_degrees"] = (BASE_C - f["temp_c"]).clip(lower=0)
    f["hour_sin"] = np.sin(2*np.pi*f["hour"]/24);  f["hour_cos"] = np.cos(2*np.pi*f["hour"]/24)
    f["doy_sin"]  = np.sin(2*np.pi*f["day_of_year"]/365.25)
    f["doy_cos"]  = np.cos(2*np.pi*f["day_of_year"]/365.25)
    feats = f.dropna().reset_index(drop=True)

    # --- CQR: train -> calibrate Q -> refit final models on all data ---
    def fit_q(q, data):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                max_iter=600, learning_rate=0.05, random_state=42)
        m.fit(data[FEATURE_COLS], data["demand_mwh"]); return m

    cutoff = feats["period_utc"].max() - pd.Timedelta(days=60)
    tr, cal = feats[feats["period_utc"] < cutoff], feats[feats["period_utc"] >= cutoff]
    lo_m, hi_m = fit_q(0.10, tr), fit_q(0.90, tr)
    E = np.maximum(lo_m.predict(cal[FEATURE_COLS]) - cal["demand_mwh"].values,
                   cal["demand_mwh"].values - hi_m.predict(cal[FEATURE_COLS]))
    alpha = 0.20
    Q = float(np.quantile(E, np.ceil((len(E)+1)*(1-alpha))/len(E), method="higher"))

    models = {"lo": fit_q(0.10, feats), "mid": fit_q(0.50, feats), "hi": fit_q(0.90, feats)}
    return models, Q, feats, clean


models, conformal_Q, features, clean = build()

# ---------- UI ----------
st.title("ERCOT Grid Planner — short-horizon demand forecast")
st.caption("Hourly ERCOT demand with a conformalized 80% confidence band and a "
           "temperature stress-test. Pipeline rebuilt live from EIA + Open-Meteo.")

counts = features.groupby(features["period_local"].dt.date).size()
full_days = sorted(counts[counts >= 24].index)
c1, c2 = st.columns(2)
target = c1.date_input("Day to forecast", value=full_days[-1],
                       min_value=full_days[0], max_value=full_days[-1])
delta = c2.slider("Temperature scenario (°C)", -5.0, 10.0, 0.0, 0.5)
if target not in set(full_days):
    st.warning("No full day of data for that date; using the latest."); target = full_days[-1]

day = features[features["period_local"].dt.date == target].copy().sort_values("period_utc")
x, actual = day["period_local"], day["demand_mwh"].values

def predict(frame, d):
    g = frame.copy(); g["temp_c"] = g["temp_c"] + d
    g["cooling_degrees"] = (g["temp_c"] - BASE_C).clip(lower=0)
    g["heating_degrees"] = (BASE_C - g["temp_c"]).clip(lower=0)
    X = g[FEATURE_COLS]
    return (models["lo"].predict(X) - conformal_Q, models["mid"].predict(X),
            models["hi"].predict(X) + conformal_Q)

lo, mid, hi = predict(day, delta)
base_mid = predict(day, 0.0)[1]
pk = int(np.argmax(mid)); peak = float(mid[pk]); worst = float(hi[pk])

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x, y=lo, fill="tonexty", fillcolor="rgba(65,105,225,0.18)",
                         line=dict(width=0), name="80% band"))
fig.add_trace(go.Scatter(x=x, y=mid, line=dict(color="royalblue", width=3), name="Forecast"))
if delta != 0:
    fig.add_trace(go.Scatter(x=x, y=base_mid, line=dict(color="gray", dash="dot"), name="0°C"))
fig.add_trace(go.Scatter(x=x, y=actual, line=dict(color="black", dash="dot"), name="Actual"))
fig.add_trace(go.Scatter(x=[x.iloc[pk]], y=[peak], mode="markers",
                         marker=dict(color="crimson", size=12, symbol="diamond"), name="Peak"))
fig.update_layout(xaxis_title="Local time", yaxis_title="Demand (MW)", height=480,
                  legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig, use_container_width=True)

daily_peaks = clean.groupby(clean["period_local"].dt.date)["demand_mwh"].max()
pct = float((daily_peaks < peak).mean() * 100)
m1, m2, m3 = st.columns(3)
m1.metric("Predicted peak", f"{peak:,.0f} MW",
          delta=(f"{peak-float(base_mid.max()):+,.0f} vs 0°C" if delta else None))
m2.metric("Peak hour", x.iloc[pk].strftime("%H:%M"))
m3.metric("Reserve to worst case", f"{worst:,.0f} MW")

tier = ("EXTREME — top 5% of days. Pre-stage peaking units; secure reserves to the upper band."
        if pct >= 95 else "ELEVATED. Schedule reserves to cover the upper band." if pct >= 80
        else "TYPICAL-TO-BUSY day. Standard reserve margin." if pct >= 50
        else "LIGHT LOAD — good window for maintenance / low-cost dispatch.")
st.subheader("Operator recommendation")
(st.error if pct >= 95 else st.warning if pct >= 80 else st.info if pct >= 50 else st.success)(
    f"**{tier}**\n\nPeak ~{pct:.0f}th percentile of historical days. "
    f"Size reserves to **{worst:,.0f} MW**.")
