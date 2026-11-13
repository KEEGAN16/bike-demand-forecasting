
import pandas as pd

from src import predict as predict_module


def test_predict_next_hour(monkeypatch):
    timestamps = pd.date_range(
        "2011-01-01",
        periods=192,
        freq="h"
    )

    data = pd.DataFrame({
        "dteday": timestamps.strftime("%Y-%m-%d"),
        "hr": timestamps.hour,
        "cnt": range(192)
    })

    class DummyModel:
        def predict(self, features):
            assert features.shape == (1, 7)
            assert features.iloc[0]["hour"] == 0
            assert features.iloc[0]["workingday"] == 0
            assert features.iloc[0]["lag_1"] == 191
            assert features.iloc[0]["lag_24"] == 168
            assert features.iloc[0]["lag_168"] == 24

            return [25.4]

    monkeypatch.setattr(
        predict_module.pd,
        "read_csv",
        lambda path: data.copy()
    )

    monkeypatch.setattr(
        predict_module.joblib,
        "load",
        lambda path: DummyModel()
    )

    forecast_hour, predicted_rentals = (
        predict_module.predict_next_hour(workingday=0)
    )

    assert forecast_hour == pd.Timestamp("2011-01-09 00:00:00")
    assert predicted_rentals == 25
