import joblib
from pathlib import Path
import pandas as pd

from src.features import build_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "workingday",
    "lag_1",
    "lag_24",
    "lag_168",
]


def prepare_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)

    data = build_features(data)

    model_df = data[
        ["timestamp"] + FEATURE_COLUMNS + ["cnt"]
    ].copy()

    model_df = model_df.dropna(subset=FEATURE_COLUMNS)
    model_df = model_df.sort_values("timestamp").reset_index(drop=True)

    return model_df


def train_model(model_df: pd.DataFrame) -> None:
    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index]
    test_df = model_df.iloc[split_index:]


    validation_split = int(len(train_df) * 0.8)

    fit_df = train_df.iloc[:validation_split]
    validation_df = train_df.iloc[validation_split:]

    configurations = [
        {"max_depth": None, "min_samples_leaf": 1},
        {"max_depth": 12, "min_samples_leaf": 3},
        {"max_depth": 8, "min_samples_leaf": 5},
    ]

    print("\nHyperparameter tuning results:")

    best_config = None
    best_validation_mae = float("inf")

    for config in configurations:
        candidate = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            **config
        )

        candidate.fit(
            fit_df[FEATURE_COLUMNS],
            fit_df["cnt"]
        )

        validation_predictions = candidate.predict(
            validation_df[FEATURE_COLUMNS]
        )

        validation_mae = mean_absolute_error(
            validation_df["cnt"],
            validation_predictions
        )

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_config = config.copy()

        print(
            f"Config: {config}, "
            f"Validation MAE: {validation_mae:.2f}"
        )

    print(f"\nBest configuration: {best_config}")
    print(f"Best validation MAE: {best_validation_mae:.2f}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["cnt"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["cnt"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        **best_config
)

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)

    train_mae = mean_absolute_error(
        y_train,
        train_predictions
    )

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "bike_demand_model.joblib"

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")

    predictions = model.predict(X_test)

    forest_mae = mean_absolute_error(y_test, predictions)
    baseline_mae = mean_absolute_error(
        y_test,
        test_df["lag_168"]
    )

    improvement = (
        (baseline_mae - forest_mae) / baseline_mae
    ) * 100

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Random Forest MAE: {forest_mae:.2f}")
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"MAE improvement: {improvement:.2f}%")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Testing MAE: {forest_mae:.2f}")
    print(f"MAE gap: {forest_mae - train_mae:.2f}")



if __name__ == "__main__":
    model_df = prepare_data("data/raw/hour.csv")
    train_model(model_df)

  