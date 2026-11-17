# WattWise AI: Household Electricity Consumption Forecasting
**Two-page executive summary | Student capstone**

**Student:** Aanchal Srivastava  
**Roll number:** 2025031101  
**Course / Institution:** B.Tech Electrical Engineering — Second Year | Institution: Madan Mohan Malaviya University of Technology  
**Submission date:** 17 November 2026

## 1. Executive overview
WattWise AI is a prototype that forecasts the next day's average household active power consumption, measured in kilowatts (kW), from historical electricity readings. It demonstrates problem definition, data preparation, feature engineering, model selection, evaluation, and a Streamlit interface.

## 2. Problem and measurable objective
Household electricity demand varies over time. This is framed as supervised time-series regression. The measurable outcome is forecast error on a later held-out period. MAE is the primary metric; RMSE and R² are also reported. The project does not claim to reduce bills or energy use; those outcomes require a separate user study.

## 3. Data source and preparation
The project uses UCI's Individual Household Electric Power Consumption dataset: 2,075,259 one-minute measurements from one household in Sceaux, France, from December 2006 to November 2010. The pipeline interprets '?' as missing, parses timestamps, aggregates valid active-power measurements into daily means, and interpolates missing daily aggregates. Features include lags, trailing averages, weekday, month, and weekend indicators. Predictors are shifted to avoid using the target day's data.

## 4. Model and evaluation
A Random Forest Regressor is compared with a persistence baseline. The earliest 80% of supervised rows train the model and the latest 20% form the test set. Insert actual MAE, RMSE and R² from reports/metrics.json after running python train.py. Do not invent scores or claim the model beats the baseline unless measured results support it.

## 5. Product, impact, and future scope
The Streamlit dashboard displays recent consumption, a next-day forecast, and held-out evaluation metrics. It is a prototype for short-term consumption awareness, not a utility-grade system. The single-household historical dataset may not generalize. The model does not predict appliance-level demand, electricity bills, or grid peaks. Future work includes recent multi-household data, weather/calendar context, model comparisons, drift monitoring, and usability testing.

## 6. Reproducibility
Install with pip install -r requirements.txt; run python train.py, python -m pytest, and streamlit run app.py. The first training run needs internet access to retrieve the UCI archive.

**Dataset citation:** Hebrail, G., & Berard, A. (2006). Individual Household Electric Power Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54. CC BY 4.0.
