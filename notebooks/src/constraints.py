from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import CAPACITY_PERCENTILE, CAPACITY_STDDEV_FACTOR


@dataclass
class ConstraintSet:
    """Constraint parameters consumed by the current solver pipeline."""

    capacity_95pct: float
    capacity_2sigma: float
    max_daily_swing: float
    weekly_std_bounds: Tuple[float, float]
    dp_return_rates: Dict[int, float]


class ConstraintExtractor:
    """Extract only the constraint parameters used by the current workflow."""

    def __init__(self, hour_data: pd.DataFrame, day_data: pd.DataFrame):
        """Initialize with hourly and daily demand data."""
        self.hour_data = hour_data
        self.day_data = day_data
        self._validate_data_format()

    def _hourly_capacity_percentile(
        self, percentile: int = CAPACITY_PERCENTILE
    ) -> float:
        """Capacity computed from the selected hourly demand percentile."""
        return float(np.percentile(self.hour_data["cnt"], percentile))

    def _hourly_capacity_stddev(self, factor: float = CAPACITY_STDDEV_FACTOR) -> float:
        """Capacity computed from hourly mean plus a variability buffer."""
        mu_hourly = float(self.hour_data["cnt"].mean())
        sigma_hourly = float(self.hour_data["cnt"].std())
        return mu_hourly + factor * sigma_hourly

    def _dp_hourly_demand_profile(self) -> np.ndarray:
        """Average hourly demand profile used by the return-rate model."""
        hourly_profile = (
            self.hour_data.groupby("hr")["cnt"]
            .mean()
            .reindex(range(24), fill_value=0.0)
        )
        return hourly_profile.to_numpy(dtype=float)

    def _dp_return_rates(
        self,
        hourly_demand: np.ndarray,
        r_min: float = 0.40,
        r_max: float = 0.90,
    ) -> np.ndarray:
        """Demand-inverse hourly return rates used by the MIP and GA models."""
        d_min = float(hourly_demand.min())
        d_max = float(hourly_demand.max())

        if np.isclose(d_max, d_min):
            midpoint = (r_min + r_max) / 2.0
            return np.full(hourly_demand.shape, midpoint, dtype=float)

        return r_min + (r_max - r_min) * (
            1.0 - (hourly_demand - d_min) / (d_max - d_min)
        )

    def _max_daily_swing(self) -> float:
        """Maximum absolute change in city-wide demand between consecutive days."""
        daily_demand = self.day_data.sort_values("dteday")["cnt"].to_numpy(dtype=float)
        if len(daily_demand) < 2:
            return 0.0
        return float(np.abs(np.diff(daily_demand)).max())

    def _weekly_std_bounds(self) -> Tuple[float, float]:
        """Observed rolling 7-day demand variability bounds."""
        rolling_7d_std = (
            self.day_data.sort_values("dteday")["cnt"].rolling(window=7).std().dropna()
        )
        if rolling_7d_std.empty:
            return (0.0, 0.0)
        return (float(rolling_7d_std.min()), float(rolling_7d_std.max()))

    def extract_all(self) -> ConstraintSet:
        """Extract the constraint parameters used by the current models."""
        hourly_demand = self._dp_hourly_demand_profile()
        return_rates = self._dp_return_rates(hourly_demand)

        return ConstraintSet(
            capacity_95pct=self._hourly_capacity_percentile(),
            capacity_2sigma=self._hourly_capacity_stddev(),
            max_daily_swing=self._max_daily_swing(),
            weekly_std_bounds=self._weekly_std_bounds(),
            dp_return_rates={
                int(hour): float(value) for hour, value in enumerate(return_rates)
            },
        )

    def _validate_data_format(self):
        """Validate that the columns required by the active extractors exist."""
        required_hourly = ["cnt", "hr"]
        required_daily = ["cnt", "dteday"]

        missing_hourly = [c for c in required_hourly if c not in self.hour_data.columns]
        missing_daily = [c for c in required_daily if c not in self.day_data.columns]

        if missing_hourly or missing_daily:
            raise ValueError(
                f"Missing columns - Hourly: {missing_hourly}, Daily: {missing_daily}"
            )
