import pandas as pd


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    result["timestamp"] = (
        pd.to_datetime(result["dteday"])
        + pd.to_timedelta(result["hr"], unit="h")
    )

    result = result.sort_values("timestamp").reset_index(drop=True)

    if result["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found")

    rentals_by_timestamp = result.set_index("timestamp")["cnt"]

    for hours in [1, 24, 168]:
        result[f"lag_{hours}"] = (
            result["timestamp"] - pd.Timedelta(hours=hours)
        ).map(rentals_by_timestamp)

    result["hour"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.day_of_week
    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    return result
