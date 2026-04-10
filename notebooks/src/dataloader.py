import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import *
from src.utils import *


# Path to local fallback CSV files
def _resolve_data_dir() -> str:
    try:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(src_dir))
    except NameError:
        cwd = os.getcwd()
        if os.path.basename(cwd) == "notebooks":
            repo_root = os.path.dirname(cwd)
        elif os.path.exists(os.path.join(cwd, "notebooks")):
            repo_root = cwd
        else:
            repo_root = cwd
    return os.path.normpath(os.path.join(repo_root, "data", "bike+sharing+dataset"))


_DATA_DIR = _resolve_data_dir()
_HOUR_CSV = os.path.join(_DATA_DIR, "hour.csv")
_DAY_CSV = os.path.join(_DATA_DIR, "day.csv")


def _load_from_local():
    """Load bike sharing data from local CSV files in notebooks/data/."""
    if not os.path.exists(_HOUR_CSV):
        raise FileNotFoundError(f"Local fallback not found: {_HOUR_CSV}")

    hour_data = pd.read_csv(_HOUR_CSV)
    # Drop the index column if present
    if "instant" in hour_data.columns:
        hour_data = hour_data.drop(columns=["instant"])

    print(f"[Fallback] Loaded hourly data from local CSV: {hour_data.shape[0]} rows")
    return hour_data


def load_bike_data():
    hour_data = None
    try:
        bike_sharing = fetch_ucirepo(id=275)

        # Get features and targets
        X = bike_sharing.data.features
        y = bike_sharing.data.targets

        # Combine into single dataframe (hourly data)
        hour_data = pd.concat([X, y], axis=1)

        # Add original data columns (casual, registered)
        original_data = bike_sharing.data.original
        if "casual" in original_data.columns:
            hour_data["casual"] = original_data["casual"].values
        if "registered" in original_data.columns:
            hour_data["registered"] = original_data["registered"].values

        print(
            f"Loaded hourly data from UCI: {hour_data.shape[0]} rows, {hour_data.shape[1]} columns"
        )
    except Exception as e:
        print(f"[Warning] UCI API unavailable ({e}). Falling back to local CSV files.")
        hour_data = _load_from_local()

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


