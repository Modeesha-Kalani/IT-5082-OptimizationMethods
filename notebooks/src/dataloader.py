import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import (
    INSTANCE_SLICES_TEMPLATE,
    MIN_STATION_CAPACITY,
    REAL_STATION_CLUSTERS,
    TARGET_UTILIZATION_RATIO,
)
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


def load_real_instances(hour_data, day_data):
    """
    Args:
        hour_data  (pd.DataFrame): Hourly data from.
        day_data   (pd.DataFrame): Daily aggregated.

    Returns:
        list: Four test instances (small, medium, large, real) as dicts with keys:
              stations, demand, capacity, distance_matrix, metadata, constraints.
    """
    print("\nExtracting mathematical constraints for capacity calculation...")
    extractor = ConstraintExtractor(hour_data, day_data)
    constraints = extractor.extract_all()
    capacity_method = max(constraints.capacity_95pct, constraints.capacity_2sigma)

    hd = hour_data.sort_values(["dteday", "hr"]).reset_index(drop=True)

    # Build deterministic hourly demand profiles from real data.
    # profile[h] = mean(cnt | hr==h, day-type filter), for h in 0..23.

    def _hourly_profile(mask):
        return np.array(
            [
                hd.loc[mask & (hd["hr"] == h), "cnt"].mean()
                if (mask & (hd["hr"] == h)).any()
                else 0.0
                for h in range(24)
            ],
            dtype=float,
        )

    workday_mask = hd["workingday"] == 1
    weekend_mask = hd["workingday"] == 0
    profiles = {
        "workday": np.maximum(_hourly_profile(workday_mask), 1e-6),
        "weekend": np.maximum(_hourly_profile(weekend_mask), 1e-6),
    }
    profiles["mixed"] = (profiles["workday"] + profiles["weekend"]) / 2.0

    print("  Hourly profiles computed from real data:")
    for dtype, prof in profiles.items():
        peak_h = int(np.argmax(prof))
        print(
            f"    {dtype:7s}: peak hour={peak_h:02d}:00, max={prof.max():.0f}, min={prof.min():.0f}"
        )

    # Representative date windows (deterministic selection).

    daily = (
        hd.groupby("dteday")
        .agg(cnt=("cnt", "sum"), workingday=("workingday", "first"))
        .reset_index()
    )
    workdays = daily[daily["workingday"] == 1].copy()
    median_cnt = workdays["cnt"].median()
    # 5th-closest to median avoids edge dates at start/end of dataset
    typical_date = workdays.iloc[
        (workdays["cnt"] - median_cnt).abs().argsort().values[4]
    ]["dteday"]
    small_start = int(hd[hd["dteday"] == typical_date].index[0])
    medium_start = small_start + 7 * 24
    typical_weekday = pd.Timestamp(typical_date).weekday()  # 0=Monday
    large_start = max(0, small_start - typical_weekday * 24)

    # Build instance slices from config template
    instance_slices = [
        (
            0
            if n_time is None
            else (
                small_start
                if size == "small"
                else medium_start
                if size == "medium"
                else large_start
            ),
            len(hd) if n_time is None else n_time,
            n_stations,
            size,
        )
        for n_time, n_stations, size in INSTANCE_SLICES_TEMPLATE
    ]

    test_instances = []
    for idx, (slice_start, n_time, n_stations, size_label) in enumerate(
        instance_slices
    ):
        # Real station definitions
        clusters = REAL_STATION_CLUSTERS[:n_stations]
        station_names = [c[0] for c in clusters]
        coords = np.array([[c[1], c[2]] for c in clusters])  # [lat, lon]
        demand_types = [c[3] for c in clusters]

        # Real distances (haversine, km)
        distance_matrix = station_distance_matrix_km(coords)

        # Real time window from the dataset
        slice_end = min(slice_start + n_time, len(hd))
        actual_slice = hd.iloc[slice_start:slice_end].reset_index(drop=True)
        actual_cnt = np.zeros(n_time)
        actual_hrs = np.zeros(n_time, dtype=int)
        actual_cnt[: len(actual_slice)] = actual_slice["cnt"].values
        actual_hrs[: len(actual_slice)] = actual_slice["hr"].values.astype(int)
        for t in range(len(actual_slice), n_time):
            actual_hrs[t] = t % 24

        # Distribute citywide cnt to stations using real demand-type profiles.
        # At each hour h, station s receives a share proportional to
        # profiles[demand_type_s][h].  Fully deterministic — no random values.

        demand_matrix = np.zeros((n_time, n_stations))
        for t in range(n_time):
            h = actual_hrs[t]
            weights = np.array([profiles[dt][h] for dt in demand_types])
            weights /= weights.sum()
            demand_matrix[t, :] = actual_cnt[t] * weights

        # Capacity from real constraint methods (max of P_95 and μ+2σ)
        peak_demand_per_station = np.max(demand_matrix, axis=0)
        avg_peak = np.mean(peak_demand_per_station)
        capacity = (
            peak_demand_per_station * capacity_method / max(avg_peak, 1e-6)
        ).astype(int)
        capacity = np.maximum(capacity, MIN_STATION_CAPACITY)

        # Scale to realistic daily utilization ratio.
        # The LP models net outflows; bikes are returned in reality so raw
        # citywide cnt >> net station demand.  Scaling preserves temporal and
        # spatial patterns while creating genuine inter-station imbalance.
        total_cap = float(np.sum(capacity))
        total_dem = float(np.sum(demand_matrix))
        days = n_time / 24.0
        if total_dem > 0:
            demand_matrix *= (total_cap * TARGET_UTILIZATION_RATIO * days) / total_dem
        demand_matrix = np.maximum(demand_matrix, 0)
        peak_demand_per_station = np.max(demand_matrix, axis=0)

        slice_dates = (
            actual_slice["dteday"].unique().tolist() if len(actual_slice) else []
        )
        # Peak demand = busiest moment across all stations combined
        demand_by_time = demand_matrix.sum(axis=1)
        peak_demand_system = float(np.max(demand_by_time))

        metadata = {
            "instance_id": idx,
            "size": size_label,
            "n_stations": n_stations,
            "n_time_periods": n_time,
            "total_demand": float(np.sum(demand_matrix)),
            "avg_demand_per_station": float(np.sum(demand_matrix) / n_stations),
            "peak_demand": peak_demand_system,
            "peak_demand_per_station": peak_demand_per_station.tolist(),
            "station_names": station_names,
            "station_demand_types": demand_types,
            "station_coords": coords.tolist(),
            "real_data_dates": slice_dates,
            "citywide_cnt_min": float(actual_cnt.min()),
            "citywide_cnt_max": float(actual_cnt.max()),
            "constraint_extraction": {
                "capacity_method": (
                    f"max(P_95={constraints.capacity_95pct:.1f},"
                    f" μ+2σ={constraints.capacity_2sigma:.1f})"
                ),
            },
        }

        instance = {
            "stations": list(range(n_stations)),
            "demand": demand_matrix,
            "capacity": capacity,
            "distance_matrix": distance_matrix,
            "metadata": metadata,
            "constraints": constraints,
        }
        test_instances.append(instance)

        print(f"Instance {idx} ({size_label}): {n_stations} stations, {n_time}h")
        print(f"  Real dates : {', '.join(str(d) for d in slice_dates)}")
        print(f"  Cnt range  : {actual_cnt.min():.0f}-{actual_cnt.max():.0f}")
        for s, (name, dt) in enumerate(zip(station_names, demand_types)):
            print(f"  Station {s} [{dt:7s}]: {name}")
        print(f"  Capacity   : min={np.min(capacity)}, max={np.max(capacity)}")
        print(
            f"  Total demand (scaled to {TARGET_UTILIZATION_RATIO:.0%} utilization):"
            f" {int(np.sum(demand_matrix))}"
        )

    return test_instances
