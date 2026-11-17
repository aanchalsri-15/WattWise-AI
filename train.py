from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.pipeline import load_daily_consumption, make_supervised, chronological_split, FEATURES

def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    daily = load_daily_consumption()
    supervised = make_supervised(daily)
    train, test = chronological_split(supervised)
    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]
    model = RandomForestRegressor(n_estimators=300, max_features=0.9, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = test["lag_1"].to_numpy()
    def scores(actual, predicted):
        return {"mae_kw":float(mean_absolute_error(actual,predicted)),"rmse_kw":float(np.sqrt(mean_squared_error(actual,predicted))),"r2":float(r2_score(actual,predicted))}
    metrics = {"project":"WattWise AI","target":"next-day daily mean Global_active_power (kW)","dataset_rows_daily":int(len(daily)),"supervised_rows":int(len(supervised)),"train_rows":int(len(train)),"test_rows":int(len(test)),"train_start":str(train.index.min().date()),"train_end":str(train.index.max().date()),"test_start":str(test.index.min().date()),"test_end":str(test.index.max().date()),"baseline_persistence":scores(y_test,baseline),"random_forest":scores(y_test,pred),"features":FEATURES,"random_state":42,"note":"Chronological 80/20 split; test period is later than training period."}
    joblib.dump({"model":model,"features":FEATURES},"models/wattwise_model.joblib")
    Path("reports/metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    pd.DataFrame({"date":test.index,"actual_kw":y_test.to_numpy(),"baseline_predicted_kw":baseline,"random_forest_predicted_kw":pred}).to_csv("reports/test_predictions.csv",index=False)
    pd.Series(model.feature_importances_,index=FEATURES,name="importance").sort_values(ascending=False).to_csv("reports/feature_importance.csv",header=True)
    print(json.dumps(metrics,indent=2))
    print("\nSaved model, metrics, predictions, and feature importances.")

if __name__ == "__main__":
    main()
