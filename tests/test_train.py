
import pandas as pd

from src.train import prepare_data


def test_prepare_data(tmp_path):
    timestamps = pd.date_range(
        "2011-01-01",
        periods=192,
        freq="h"
    )

    data = pd.DataFrame({
        "dteday": timestamps.strftime("%Y-%m-%d"),
        "hr": timestamps.hour,
        "cnt": range(192),
        "workingday": [1] * 192
    })

    csv_path = tmp_path / "hour.csv"
    data.to_csv(csv_path, index=False)

    result = prepare_data(str(csv_path))

    assert len(result) == 24
    assert result["timestamp"].is_monotonic_increasing
    assert result["lag_168"].notna().all()
    assert result.iloc[0]["lag_168"] == 0
