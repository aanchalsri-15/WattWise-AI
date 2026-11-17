# WattWise AI — Five-slide pitch

## Slide 1 — Hook and problem
**Can we estimate tomorrow's household electricity demand from yesterday's usage?**
- Household demand varies over time.
- Goal: forecast next-day daily mean active power (kW).
- Student capstone: Aanchal Srivastava | Roll no. 2025031101
- B.Tech Electrical Engineering — Second Year

## Slide 2 — Data and insights
- Dataset: UCI Individual Household Electric Power Consumption.
- 2,075,259 minute-level records from one household in Sceaux, France (Dec 2006–Nov 2010).
- Preparation: parse timestamps, handle '?' as missing, aggregate to daily mean, interpolate missing daily aggregates.
- Challenge: historical single-household data may not represent current homes or other regions.

## Slide 3 — Solution and model architecture
- Input: historical daily mean active power.
- Features: prior-day/2/7/14/28-day lags, 7/28-day trailing means, weekday, month, weekend.
- Model: Random Forest Regressor.
- Benchmark: persistence forecast (next day equals latest observed day).
- Evaluation: chronological 80/20 train/test split.

## Slide 4 — Impact, demo and success metrics
- Streamlit dashboard shows recent history, next-day estimate, and held-out metrics.
- Success metrics: MAE, RMSE, R² on the future holdout period.
- **Insert measured values from reports/metrics.json after running python train.py.**
- Do not claim reduced bills or energy use; these outcomes have not been evaluated.

## Slide 5 — Future scope
- Validate on recent data from multiple households.
- Add weather and calendar context where appropriate and consented.
- Compare seasonal baselines and other forecasting models.
- Monitor data drift and test dashboard usability with users.
- Limitation: prototype forecast, not a utility-grade demand management system.
