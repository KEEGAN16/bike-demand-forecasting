
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features import build_features
from src.train import FEATURE_COLUMNS


DATA_PATH = Path("data/raw/hour.csv")
MODEL_PATH = Path("models/bike_demand_model.joblib")


def predict_next_hour(workingday: int) -> tuple[pd.Timestamp, int]:
    data = pd.read_csv(DATA_PATH)
    history = build_features(data)

    last_hour = history["timestamp"].max()
    target_hour = last_hour + pd.Timedelta(hours=1)

    rentals = history.set_index("timestamp")["cnt"]

    lag_features = {}

    for hours in [1, 24, 168]:
        previous_hour = target_hour - pd.Timedelta(hours=hours)

        if previous_hour not in rentals.index:
            raise ValueError(
                f"Missing historical data for {previous_hour}"
            )

        lag_features[f"lag_{hours}"] = rentals.loc[previous_hour]

    features = pd.DataFrame([{
        "hour": target_hour.hour,
        "day_of_week": target_hour.day_of_week,
        "is_weekend": int(target_hour.day_of_week >= 5),
        "workingday": workingday,
        **lag_features
    }])[FEATURE_COLUMNS]

    model = joblib.load(MODEL_PATH)

    prediction = model.predict(features)[0]

    return target_hour, max(0, round(float(prediction)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict bike rental demand for the next hour"
    )

    parser.add_argument(
        "--workingday",
        type=int,
        choices=[0, 1],
        required=True,
        help="1 for a working day, 0 for a holiday or weekend"
    )

    args = parser.parse_args()

    forecast_hour, predicted_rentals = predict_next_hour(
        args.workingday
    )

    print(f"Forecast hour: {forecast_hour}")
    print(f"Predicted rentals: {predicted_rentals}")
