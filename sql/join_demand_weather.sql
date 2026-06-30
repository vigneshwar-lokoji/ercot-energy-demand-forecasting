-- Stage B — Join ERCOT demand with Dallas weather and derive calendar features (DuckDB)
--
-- Inputs (registered as DuckDB tables on the Zerve canvas):
--   demand_raw   from A1 (EIA hourly ERCOT demand)   — columns: period (UTC), value (MW)
--   weather_raw  from A2 (Open-Meteo hourly weather)  — columns: period (UTC), temp_c, humidity, wind_speed
--
-- Both sources are keyed on the hourly UTC timestamp `period`. The join is an INNER
-- join so only hours present in both feeds survive. Calendar features are derived from
-- the timestamp localized to America/Chicago (DST-aware), because demand follows local
-- human activity, not UTC.

WITH joined AS (
    SELECT
        d.period::TIMESTAMPTZ AS period_utc,
        d.value               AS demand_mwh,
        w.temp_c,
        w.humidity,
        w.wind_speed
    FROM demand_raw  AS d
    INNER JOIN weather_raw AS w
        ON d.period = w.period
)
SELECT
    period_utc,
    demand_mwh,
    temp_c,
    humidity,
    wind_speed,
    -- localize to Texas time for the calendar features (handles DST automatically)
    timezone('America/Chicago', period_utc)                               AS period_local,
    EXTRACT(hour  FROM timezone('America/Chicago', period_utc))           AS hour,
    EXTRACT(dow   FROM timezone('America/Chicago', period_utc))           AS dayofweek,    -- 0 = Sunday
    EXTRACT(month FROM timezone('America/Chicago', period_utc))           AS month,
    EXTRACT(doy   FROM timezone('America/Chicago', period_utc))           AS day_of_year,
    CASE
        WHEN EXTRACT(dow FROM timezone('America/Chicago', period_utc)) IN (0, 6)
        THEN 1 ELSE 0
    END                                                                   AS is_weekend
FROM joined
ORDER BY period_utc;
