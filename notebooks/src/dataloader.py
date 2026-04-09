import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import *
from src.utils import *


def load_bike_data():
    bike_sharing = fetch_ucirepo(id=275)

    # Get features and targets
    X = bike_sharing.data.features
    y = bike_sharing.data.targets

    # Combine into single dataframe (hourly data)
    hour_data = pd.concat([X, y], axis=1)

    print(
        f"Loaded hourly data: {hour_data.shape[0]} rows, {hour_data.shape[1]} columns"
    )
    print(f"  Columns: {', '.join(hour_data.columns.tolist()[:14])}  ")

    # Create daily aggregation
    agg_dict = {
        "season": "first",
        "yr": "first",
        "mnth": "first",
        "holiday": "first",
        "weekday": "first",
        "workingday": "first",
        "weathersit": "mean",
        "temp": "mean",
        "atemp": "mean",
        "hum": "mean",
        "windspeed": "mean",
        "cnt": "sum",
    }

    # Add casual/registered if they exist
    if "casual" in hour_data.columns:
        agg_dict["casual"] = "sum"
    if "registered" in hour_data.columns:
        agg_dict["registered"] = "sum"

    day_data = hour_data.groupby("dteday").agg(agg_dict).reset_index()

    print(f"Created daily aggregation: {day_data.shape[0]} rows")

    return hour_data, day_data
