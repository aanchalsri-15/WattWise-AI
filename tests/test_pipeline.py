import pandas as pd
import pytest
from src.pipeline import make_supervised, chronological_split, make_next_day_features, FEATURES

def test_features_use_prior_days_and_drop_initial_lags():
    idx=pd.date_range("2020-01-01",periods=50,freq="D")
    series=pd.Series(range(50),index=idx,dtype=float)
    data=make_supervised(series)
    first=data.iloc[0]
    assert first.name==idx[28]
    assert first["target"]==28
    assert first["lag_1"]==27
    assert first["lag_7"]==21
    assert first["rolling_mean_7"]==pytest.approx(24)
    assert list(data[FEATURES].columns)==FEATURES

def test_chronological_split_keeps_time_order():
    idx=pd.date_range("2020-01-01",periods=100,freq="D")
    data=pd.DataFrame({"x":range(100)},index=idx)
    train,test=chronological_split(data)
    assert train.index.max()<test.index.min()
    assert len(train)==80
    assert len(test)==20

def test_invalid_split_fraction_raises():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"x":range(10)}),1.0)

def test_next_day_features_use_latest_observations():
    idx=pd.date_range("2020-01-01",periods=40,freq="D")
    series=pd.Series(range(40),index=idx,dtype=float)
    row=make_next_day_features(series).iloc[0]
    assert row.name==idx[-1]+pd.Timedelta(days=1)
    assert row["lag_1"]==39
    assert row["lag_7"]==33
    assert row["rolling_mean_7"]==pytest.approx(36)
