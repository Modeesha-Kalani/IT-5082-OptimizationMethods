import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from ortools.linear_solver import pywraplp

import src.config as config_module
from src.config import COST_PER_KM, INITIAL_INVENTORY_RATIO, PENALTY_PER_BIKE
from src.constraints import ConstraintSet


@dataclass
class Solution:
    """Solution to the bike redistribution problem."""

    cost: float
    redistribution_cost: float
    unmet_demand_cost: float
    total_unmet_bikes: int
    inventory_sequence: np.ndarray  # shape [T, S], end-of-hour inventory
    redistribution_matrices: np.ndarray  # shape [T-1, S, S]
    status: str
    solve_time: float
    imbalance_cost: float = 0.0


class MIPSolver:
    """MIP solver for bike-sharing redistribution."""

    def __init__(
        self,
        demand_sequence: np.ndarray,
        capacity: np.ndarray,
        distance_matrix: np.ndarray,
        initial_inventory: Optional[np.ndarray] = None,
        constraints: Optional[ConstraintSet] = None,
        cost_per_km: float = COST_PER_KM,
        penalty_per_bike: float = PENALTY_PER_BIKE,
    ):
        """
        Parameters
        ----------
        demand_sequence   : [T, S] demand per period per station
        capacity          : [S]    station capacities
        distance_matrix   : [S, S] inter-station distances
        initial_inventory : [S]    inventory at t=0; defaults to INITIAL_INVENTORY_RATIO x capacity
        constraints       : ConstraintSet extracted from real data
        cost_per_km       : redistribution cost coefficient
        penalty_per_bike  : penalty per unmet bike
        """
        self.demand_sequence = np.array(demand_sequence, dtype=float)
        self.capacity = np.array(capacity, dtype=float)
        self.distance_matrix = np.array(distance_matrix, dtype=float)
        self.constraints = constraints
        self.cost_per_km = cost_per_km
        self.penalty_per_bike = penalty_per_bike

        self.T, self.S = self.demand_sequence.shape
        self.return_lag_hours = max(int(config_module.RETURN_LAG_HOURS), 1)
        self.return_rate_scale = max(float(config_module.RETURN_RATE_SCALE), 0.0)
        self.return_distance_decay_km = max(
            float(config_module.RETURN_DISTANCE_DECAY_KM), 1e-6
        )
        self.max_redistribution_per_period = float(
            config_module.MAX_REDISTRIBUTION_CAPACITY_RATIO * np.sum(self.capacity)
        )
        self.inventory_imbalance_penalty = max(
            float(config_module.INVENTORY_IMBALANCE_PENALTY), 0.0
        )

        if initial_inventory is None:
            # Capacity-proportional initial inventory: each station starts with
            # INITIAL_INVENTORY_RATIO fraction of its own capacity.
            # This creates spatial imbalance: high-cap workday stations start with more bikes,
            # low-cap weekend stations start with fewer bikes -> forces redistribution.
            self.initial_inventory = self.capacity * INITIAL_INVENTORY_RATIO
        else:
            self.initial_inventory = np.array(initial_inventory, dtype=float)

        self.return_allocation_matrix = self._build_return_allocation_matrix()
        self.inventory_targets = self._build_inventory_targets()

    def _capped_weighted_target(
        self, total_bikes: float, weights: np.ndarray
    ) -> np.ndarray:
        """Project a weighted station target onto capacity limits."""
        total_bikes = min(float(total_bikes), float(np.sum(self.capacity)))
        weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
        if np.sum(weights) <= 1e-12:
            weights = self.capacity.copy()

        target = np.zeros(self.S, dtype=float)
        active = np.ones(self.S, dtype=bool)

        while active.any():
            active_idx = np.where(active)[0]
            remaining = total_bikes - float(np.sum(target))
            if remaining <= 1e-9:
                break

            active_weights = weights[active_idx]
            if np.sum(active_weights) <= 1e-12:
                active_weights = np.ones(len(active_idx), dtype=float)

            room = np.maximum(self.capacity[active_idx] - target[active_idx], 0.0)
            proposed = remaining * active_weights / np.sum(active_weights)
            assigned = np.minimum(proposed, room)
            target[active_idx] += assigned

            newly_full = room - assigned <= 1e-9
            if not np.any(newly_full):
                break
            active[active_idx[newly_full]] = False

        return target

    def _build_inventory_targets(self) -> np.ndarray:
        """Demand-weighted end-of-hour inventory targets for imbalance penalties."""
        total_bikes = float(np.sum(self.initial_inventory))
        targets = np.zeros((self.T, self.S), dtype=float)
        for t in range(self.T):
            reference_idx = min(t + 1, self.T - 1)
            weights = self.demand_sequence[reference_idx]
            targets[t] = self._capped_weighted_target(total_bikes, weights)
        return targets

    def _build_return_allocation_matrix(self) -> np.ndarray:
        """Allocate lagged returns across nearby destination stations."""
        if self.S == 1:
            return np.ones((1, 1), dtype=float)

        destination_pull = np.mean(self.demand_sequence, axis=0)
        if np.sum(destination_pull) <= 1e-12:
            destination_pull = self.capacity.copy()

        proximity = np.exp(-self.distance_matrix / self.return_distance_decay_km)
        np.fill_diagonal(proximity, 0.0)
        allocation = proximity * destination_pull[np.newaxis, :]

        for origin in range(self.S):
            row_sum = float(np.sum(allocation[origin]))
            if row_sum <= 1e-12:
                fallback = np.ones(self.S, dtype=float)
                fallback[origin] = 0.0
                allocation[origin] = fallback / np.sum(fallback)
            else:
                allocation[origin] /= row_sum

        return allocation

    def _hourly_return_rate(self, hour: int) -> float:
        """Return the data-driven hourly bike return rate for a given hour."""
        if self.constraints is None:
            return 0.0
        return_rates = getattr(self.constraints, "dp_return_rates", {})
        return float(return_rates.get(hour, 0.0))

    def _lagged_return_expression(self, t: int, s: int, u) -> float:
        """Return lagged bikes for station s that become available in period t."""
        lagged_t = t - self.return_lag_hours
        if lagged_t < 0 or self.return_rate_scale <= 0.0:
            return 0.0

        scaled_rate = self.return_rate_scale * self._hourly_return_rate(lagged_t % 24)
        if scaled_rate <= 0.0:
            return 0.0

        return scaled_rate * sum(
            self.return_allocation_matrix[origin, s]
            * (self.demand_sequence[lagged_t, origin] - u[lagged_t, origin])
            for origin in range(self.S)
        )

    # Public interface

    def solve(self) -> Solution:
        """Solve the bike redistribution MIP."""
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            raise RuntimeError(
                "OR-Tools SCIP solver could not be created. "
                "Ensure ortools is installed: pip install ortools"
            )

        x, r, u, imbalance = self._build_variables(solver)
        self._build_objective(solver, x, r, u, imbalance)
        self._build_core_constraints(solver, x, r, u, imbalance)
        if self.constraints is not None:
            self._build_data_constraints(solver, x, u)

        t0 = time.time()
        status = solver.Solve()
        solve_time = time.time() - t0

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            print(
                f"Warning: MIP solver returned status {status}. Returning empty solution."
            )
            return Solution(
                cost=float("inf"),
                redistribution_cost=0.0,
                unmet_demand_cost=float("inf"),
                total_unmet_bikes=0.0,
                inventory_sequence=np.zeros((self.T, self.S)),
                redistribution_matrices=np.zeros((self.T - 1, self.S, self.S)),
                status="INFEASIBLE"
                if status == pywraplp.Solver.INFEASIBLE
                else f"STATUS_{status}",
                solve_time=float(solve_time),
            )

        return self._extract_solution(solver, x, r, u, imbalance, status, solve_time)

    def _build_variables(self, solver):
        # End-of-hour inventory per station [0, cap[s]]
        x = {
            (t, s): solver.NumVar(0.0, self.capacity[s], f"x_{t}_{s}")
            for t in range(self.T)
            for s in range(self.S)
        }

        # Redistribution flows (i != j) per period
        r = {
            (t, i, j): solver.NumVar(0.0, solver.infinity(), f"r_{t}_{i}_{j}")
            for t in range(self.T - 1)
            for i in range(self.S)
            for j in range(self.S)
            if i != j
        }

        # Unmet demand slack variables
        u = {
            (t, s): solver.NumVar(0.0, solver.infinity(), f"u_{t}_{s}")
            for t in range(self.T)
            for s in range(self.S)
        }

        # Absolute deviation from the demand-weighted inventory target.
        imbalance = {
            (t, s): solver.NumVar(0.0, solver.infinity(), f"imb_{t}_{s}")
            for t in range(1, self.T)
            for s in range(self.S)
        }

        return x, r, u, imbalance

    def _build_objective(self, solver, x, r, u, imbalance):
        obj = solver.Objective()

        for t in range(self.T - 1):
            for i in range(self.S):
                for j in range(self.S):
                    if i != j:
                        obj.SetCoefficient(
                            r[t, i, j], self.distance_matrix[i, j] * self.cost_per_km
                        )

        for t in range(self.T):
            for s in range(self.S):
                obj.SetCoefficient(u[t, s], self.penalty_per_bike)

        for t in range(1, self.T):
            for s in range(self.S):
                obj.SetCoefficient(imbalance[t, s], self.inventory_imbalance_penalty)

        obj.SetMinimization()

    def _build_core_constraints(self, solver, x, r, u, imbalance):
        # Unmet demand cannot exceed realized demand.
        for t in range(self.T):
            for s in range(self.S):
                solver.Add(u[t, s] <= self.demand_sequence[t, s])

        # t=0: balance from initial inventory
        for s in range(self.S):
            solver.Add(
                x[0, s]
                == self.initial_inventory[s] - self.demand_sequence[0, s] + u[0, s]
            )

        # Bikes can only be shipped out after demand is served.
        for t in range(self.T - 1):
            for s in range(self.S):
                outflow = sum(r[t, s, j] for j in range(self.S) if j != s)
                solver.Add(outflow <= x[t, s])

        # Redistribution capacity is bounded each period by operational limits.
        for t in range(self.T - 1):
            total_redistributed = sum(
                r[t, i, j] for i in range(self.S) for j in range(self.S) if i != j
            )
            solver.Add(total_redistributed <= self.max_redistribution_per_period)

        # t>=1: balance from previous inventory + redistribution + lagged hourly returns
        for t in range(1, self.T):
            for s in range(self.S):
                inflow = sum(r[t - 1, i, s] for i in range(self.S) if i != s)
                outflow = sum(r[t - 1, s, j] for j in range(self.S) if j != s)
                returned = self._lagged_return_expression(t, s, u)
                solver.Add(
                    x[t, s]
                    == x[t - 1, s]
                    + inflow
                    - outflow
                    + returned
                    - self.demand_sequence[t, s]
                    + u[t, s]
                )

        # Penalize misalignment between end-of-hour inventory and next-hour demand.
        for t in range(1, self.T):
            for s in range(self.S):
                target = self.inventory_targets[t, s]
                solver.Add(imbalance[t, s] >= x[t, s] - target)
                solver.Add(imbalance[t, s] >= target - x[t, s])

    def _build_data_constraints(self, solver, x, u):
        """Enforce constraints derived from real UCI data."""
        cs = self.constraints
        assert cs is not None

        # Peak-hour full service is NOT enforced as a hard constraint here because
        # over multi-day horizons (e.g. 120h) the redistribution capacity cap makes
        # it geometrically impossible to guarantee zero unmet demand at every peak.
        # The penalty term in the objective already strongly discourages unmet demand.

        # Day-to-day inventory swing bound — scaled to per-station per-hour.
        # cs.max_daily_swing is the city-wide daily swing; divide by stations and
        # by 24 to convert to the hourly granularity used in this model.
        delta_max = max(
            cs.max_daily_swing / self.S / 24.0, float(np.max(self.capacity))
        )
        for t in range(1, self.T):
            for s in range(self.S):
                solver.Add(x[t, s] - x[t - 1, s] <= delta_max)
                solver.Add(x[t - 1, s] - x[t, s] <= delta_max)

        # Weekly variability — only apply a loose upper bound; skip lower bound
        # to avoid forcing inventory to vary in demand-scarce scenarios.
        if self.T >= 2:
            sigma_min, sigma_max = cs.weekly_std_bounds
            scale = self.T / 7.0 / self.S
            tv_ub = sigma_max * scale

            abs_delta = {
                (t, s): solver.NumVar(0.0, solver.infinity(), f"abs_delta_{t}_{s}")
                for t in range(1, self.T)
                for s in range(self.S)
            }
            for t in range(1, self.T):
                for s in range(self.S):
                    solver.Add(abs_delta[t, s] >= x[t, s] - x[t - 1, s])
                    solver.Add(abs_delta[t, s] >= x[t - 1, s] - x[t, s])

            for s in range(self.S):
                total_variation = sum(abs_delta[t, s] for t in range(1, self.T))
                if tv_ub > 0:
                    solver.Add(total_variation <= tv_ub)

    def _extract_solution(
        self, solver, x, r, u, imbalance, status, solve_time
    ) -> Solution:
        inventory_sequence = np.array(
            [[x[t, s].solution_value() for s in range(self.S)] for t in range(self.T)]
        )

        redistribution_matrices = np.zeros((self.T - 1, self.S, self.S))
        for t in range(self.T - 1):
            for i in range(self.S):
                for j in range(self.S):
                    if i != j:
                        redistribution_matrices[t, i, j] = r[t, i, j].solution_value()

        total_unmet = sum(
            u[t, s].solution_value() for t in range(self.T) for s in range(self.S)
        )

        redistribution_cost = float(
            np.sum(redistribution_matrices * self.distance_matrix[np.newaxis, :, :])
            * self.cost_per_km
        )
        unmet_cost = total_unmet * self.penalty_per_bike
        imbalance_cost = (
            sum(
                imbalance[t, s].solution_value()
                for t in range(1, self.T)
                for s in range(self.S)
            )
            * self.inventory_imbalance_penalty
        )

        status_str = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"

        return Solution(
            cost=float(redistribution_cost + unmet_cost + imbalance_cost),
            redistribution_cost=float(redistribution_cost),
            unmet_demand_cost=float(unmet_cost),
            total_unmet_bikes=int(total_unmet),
            inventory_sequence=inventory_sequence,
            redistribution_matrices=redistribution_matrices,
            status=status_str,
            solve_time=float(solve_time),
            imbalance_cost=float(imbalance_cost),
        )


def solve_instance(
    instance: Dict,
    cost_per_km: float = COST_PER_KM,
    penalty_per_bike: float = PENALTY_PER_BIKE,
) -> Solution:
    """Solve a bike-sharing instance from data_loader."""
    return MIPSolver(
        demand_sequence=instance["demand"],
        capacity=instance["capacity"],
        distance_matrix=instance["distance_matrix"],
        initial_inventory=None,
        constraints=instance.get("constraints"),
        cost_per_km=cost_per_km,
        penalty_per_bike=penalty_per_bike,
    ).solve()
