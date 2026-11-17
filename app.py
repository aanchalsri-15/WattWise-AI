from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.pipeline import load_daily_consumption, make_next_day_features, FEATURES

st.set_page_config(page_title="WattWise AI", page_icon="⚡", layout="wide")
st.title("⚡ WattWise AI")
st.subheader("Household electricity consumption forecasting")
st.caption("A student capstone prototype using the UCI Individual Household Electric Power Consumption dataset.")
model_path = Path("models/wattwise_model.joblib")
metrics_path = Path("reports/metrics.json")

@st.cache_resource
def get_model():
    return joblib.load(model_path)

@st.cache_data
def get_history():
    return load_daily_consumption()

if not model_path.exists() or not metrics_path.exists():
    st.warning("The model has not been trained yet. Run python train.py in the project folder, then refresh this page.")
    st.stop()
try:
    bundle = get_model()
    daily = get_history()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
except Exception as exc:
    st.error(f"Could not load project artifacts: {exc}")
    st.stop()
X_next = make_next_day_features(daily)
forecast = float(bundle["model"].predict(X_next[FEATURES])[0])
last_observed = float(daily.iloc[-1])
delta = forecast - last_observed
forecast_date = X_next.index[0].date()
c1,c2,c3 = st.columns(3)
c1.metric("Latest observed daily average",f"{last_observed:.3f} kW")
c2.metric(f"Forecast for {forecast_date}",f"{forecast:.3f} kW",f"{delta:+.3f} kW vs latest day")
c3.metric("Dataset through",str(daily.index.max().date()))
st.info("Forecasts are estimates from historical patterns, not guaranteed values or billing advice.")
st.markdown("### Recent consumption")
window = st.slider("Days of history to display",7,90,30,1)
fig,ax = plt.subplots(figsize=(11,3.5))
daily.tail(window).plot(ax=ax,linewidth=1.5)
ax.set_ylabel("Daily mean active power (kW)")
ax.set_xlabel("Date")
ax.grid(alpha=0.25)
st.pyplot(fig,clear_figure=True)
st.markdown("### Model evaluation on the held-out future period")
m1,m2 = st.columns(2)
for col,key,label in [(m1,"baseline_persistence","Persistence baseline"),(m2,"random_forest","Random Forest")]:
    with col:
        score=metrics[key]
        st.markdown(f"**{label}**")
        st.metric("MAE",f"{score['mae_kw']:.4f} kW")
        st.metric("RMSE",f"{score['rmse_kw']:.4f} kW")
        st.metric("R²",f"{score['r2']:.4f}")
st.caption(f"Chronological evaluation: train {metrics['train_start']} to {metrics['train_end']}; test {metrics['test_start']} to {metrics['test_end']}. Dataset represents one household in France (2006–2010), so results may not generalize.")
