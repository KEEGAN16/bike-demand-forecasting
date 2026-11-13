
import pandas as pd
import pytest
from src.features import build_features


def test_duplicate_timestamps():
    data = pd.DataFrame({
        "dteday": ["2011-01-01", "2011-01-01"],
        "hr": [8, 8],
        "cnt": [100, 150]
    })

    with pytest.raises(
        ValueError,
        match="Duplicate timestamps found"
    ):
        build_features(data)


def test_weekend_feature():
    data = pd.DataFrame({
        "dteday": [
            "2011-01-07",
            "2011-01-08",
            "2011-01-09"
        ],
        "hr": [12, 12, 12],
        "cnt": [100, 150, 200]
    })

    result = build_features(data)

    assert result["day_of_week"].tolist() == [4, 5, 6]
    assert result["is_weekend"].tolist() == [0, 1, 1]


def test_lag_features():
    data = pd.DataFrame({
        "dteday": [
            "2011-01-01",
            "2011-01-02",
            "2011-01-08"
        ],
        "hr": [0, 0, 0],
        "cnt": [100, 150, 200]
    })

    result = build_features(data)

    jan_2 = result[
        result["timestamp"] == pd.Timestamp("2011-01-02")
    ].iloc[0]

    jan_8 = result[
        result["timestamp"] == pd.Timestamp("2011-01-08")
    ].iloc[0]

    assert jan_2["lag_24"] == 100
    assert jan_8["lag_168"] == 100



def test_missing_previous_hour():
    data = pd.DataFrame({
        "dteday": [
            "2011-01-01",
            "2011-01-01",
            "2011-01-01"
        ],
        "hr": [0, 2, 3],
        "cnt": [10, 20, 30]
    })

    result = build_features(data)

    hour_2 = result[result["hr"] == 2].iloc[0]
    hour_3 = result[result["hr"] == 3].iloc[0]

    assert pd.isna(hour_2["lag_1"])
    assert hour_3["lag_1"] == 20
