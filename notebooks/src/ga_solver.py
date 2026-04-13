import random
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from deap import algorithms, base, creator, tools

from src.config import COST_PER_KM, INITIAL_INVENTORY_RATIO, PENALTY_PER_BIKE
from src.constraints import ConstraintSet
from src.mipsolver import MIPSolver, Solution


def _flat_index(t: int, i: int, j: int, S: int) -> int:
    """Map (t, i, j) with i != j to a flat chromosome index."""
    pair_idx = i * (S - 1) + (j if j < i else j - 1)
    return t * S * (S - 1) + pair_idx


def _encode(flows: np.ndarray, T: int, S: int) -> List[float]:
    genes: List[float] = []
    for t in range(T - 1):
        for i in range(S):
            for j in range(S):
                if i != j:
                    genes.append(float(flows[t, i, j]))
    return genes


def _decode(individual: List[float], T: int, S: int) -> np.ndarray:
    """Reshape flat chromosome into redistribution tensor [T-1, S, S].

    Genes are clipped to zero before placement because cxBlend can produce
    negative values when the blending coefficient is sampled below zero.
    """
    flows = np.zeros((T - 1, S, S))
    for t in range(T - 1):
        for i in range(S):
            for j in range(S):
                if i != j:
                    flows[t, i, j] = max(0.0, individual[_flat_index(t, i, j, S)])
    return flows


def _greedy_flow_to_target(
    inventory: np.ndarray,
    target_inventory: np.ndarray,
    distance_matrix: np.ndarray,
) -> np.ndarray:
    flow = np.zeros((len(inventory), len(inventory)))
    surplus = np.maximum(inventory - target_inventory, 0.0)
    deficit = np.maximum(target_inventory - inventory, 0.0)

    donors = np.where(surplus > 1e-9)[0]
    receivers = np.where(deficit > 1e-9)[0]
    if len(donors) == 0 or len(receivers) == 0:
        return flow

    candidate_pairs = sorted(
        (
            (float(distance_matrix[i, j]), int(i), int(j))
            for i in donors
            for j in receivers
            if i != j
        ),
        key=lambda item: item[0],
    )

    for _, i, j in candidate_pairs:
        amount = min(surplus[i], deficit[j])
        if amount <= 1e-9:
            continue
        flow[i, j] += amount
        surplus[i] -= amount
        deficit[j] -= amount

    return flow


def _build_heuristic_flows(
    demand: np.ndarray,
    capacity: np.ndarray,
    distance_matrix: np.ndarray,
    initial_inventory: np.ndarray,
    lookahead: int,
    reserve_ratio: float,
    future_scale: float,
) -> np.ndarray:
    T, S = demand.shape
    flows = np.zeros((T - 1, S, S))
    inventory = initial_inventory.copy()

    for t in range(T - 1):
        inventory = np.minimum(capacity, np.maximum(inventory - demand[t], 0.0))

        next_target = np.minimum(capacity, demand[t + 1])
        primary_flow = _greedy_flow_to_target(inventory, next_target, distance_matrix)
        inventory = inventory + primary_flow.sum(axis=0) - primary_flow.sum(axis=1)
        total_flow = primary_flow.copy()

        if lookahead > 1 and t + 2 < T:
            window_end = min(T, t + 1 + lookahead)
            future_window = demand[t + 2 : window_end]
            if len(future_window) > 0:
                weights = np.linspace(len(future_window), 1, len(future_window))
                weighted_future = np.average(future_window, axis=0, weights=weights)
                buffered_target = np.minimum(
                    capacity,
                    np.maximum(
                        next_target,
                        reserve_ratio * capacity + future_scale * weighted_future,
                    ),
                )
                secondary_flow = _greedy_flow_to_target(
                    inventory,
                    buffered_target,
                    distance_matrix,
                )
                inventory = (
                    inventory + secondary_flow.sum(axis=0) - secondary_flow.sum(axis=1)
                )
                total_flow += secondary_flow

        flows[t] = total_flow

    return flows


def _repair(
    flows: np.ndarray,
    inventory: np.ndarray,
    capacity: np.ndarray,
    max_redistribution_per_period: Optional[float] = None,
) -> np.ndarray:
    """Project raw flows onto non-negative inventory and capacity limits."""
    repaired = flows.copy()
    T_minus1, S, _ = repaired.shape
    for t in range(T_minus1):
        period_flow = np.maximum(repaired[t], 0.0)
        np.fill_diagonal(period_flow, 0.0)

        for _ in range(2):
            outflow = period_flow.sum(axis=1)
            row_scale = np.ones(S)
            positive_out = outflow > 1e-9
            row_scale[positive_out] = np.minimum(
                1.0,
                np.maximum(inventory[positive_out], 0.0) / outflow[positive_out],
            )
            period_flow *= row_scale[:, np.newaxis]

            outflow = period_flow.sum(axis=1)
            inflow = period_flow.sum(axis=0)
            available_room = np.maximum(
                capacity - np.maximum(inventory - outflow, 0.0),
                0.0,
            )
            col_scale = np.ones(S)
            positive_in = inflow > 1e-9
            col_scale[positive_in] = np.minimum(
                1.0,
                available_room[positive_in] / inflow[positive_in],
            )
            period_flow *= col_scale[np.newaxis, :]

        if max_redistribution_per_period is not None:
            total_flow = float(np.sum(period_flow))
            if total_flow > max_redistribution_per_period > 0.0:
                period_flow *= max_redistribution_per_period / total_flow

        repaired[t] = period_flow
    return repaired


def _lagged_return_vector(
    t: int,
    served_history: np.ndarray,
    scaled_return_rates: np.ndarray,
    return_allocation_matrix: np.ndarray,
    return_lag_hours: int,
) -> np.ndarray:
    """Return the vector of lagged bikes becoming available in period t."""
    lagged_t = t - return_lag_hours
    n_stations = return_allocation_matrix.shape[0]
    if lagged_t < 0:
        return np.zeros(n_stations, dtype=float)

    scaled_rate = float(scaled_return_rates[lagged_t % len(scaled_return_rates)])
    if scaled_rate <= 0.0:
        return np.zeros(n_stations, dtype=float)

    return scaled_rate * served_history[lagged_t].dot(return_allocation_matrix)


def _simulate_trajectory(
    individual: List[float],
    T: int,
    S: int,
    demand: np.ndarray,
    capacity: np.ndarray,
    distance_matrix: np.ndarray,
    initial_inventory: np.ndarray,
    cost_per_km: float,
    penalty_per_bike: float,
    scaled_return_rates: np.ndarray,
    return_lag_hours: int,
    return_allocation_matrix: np.ndarray,
    inventory_targets: np.ndarray,
    inventory_imbalance_penalty: float,
    max_redistribution_per_period: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, float, float, float, int]:
    """Simulate a chromosome under the same return and imbalance model as the MIP."""
    flows = _decode(individual, T, S)
    inv = initial_inventory.copy()
    inventory_sequence = np.zeros((T, S))
    redistribution_matrices = np.zeros((T - 1, S, S))
    served_history = np.zeros((T, S))
    total_redistrib_cost = 0.0
    total_imbalance_cost = 0.0
    total_unmet = 0.0

    for t in range(T):
        if t > 0:
            raw_flows = flows[t - 1]
            repaired = _repair(
                raw_flows[np.newaxis],
                inv,
                capacity,
                max_redistribution_per_period=max_redistribution_per_period,
            )[0]
            redistribution_matrices[t - 1] = repaired
            inflow = repaired.sum(axis=0)
            outflow = repaired.sum(axis=1)
            total_redistrib_cost += float(
                np.sum(repaired * distance_matrix) * cost_per_km
            )
            inv = inv + inflow - outflow

        inv = inv + _lagged_return_vector(
            t,
            served_history,
            scaled_return_rates,
            return_allocation_matrix,
            return_lag_hours,
        )

        unmet = np.maximum(0.0, demand[t] - inv)
        served = demand[t] - unmet
        inv = np.maximum(0.0, inv - demand[t] + unmet)
        inv = np.minimum(inv, capacity)
        served_history[t] = served
        total_unmet += float(np.sum(unmet))
        inventory_sequence[t] = inv

        if t > 0 and inventory_imbalance_penalty > 0.0:
            total_imbalance_cost += float(
                np.sum(np.abs(inv - inventory_targets[t])) * inventory_imbalance_penalty
            )

    unmet_cost = total_unmet * penalty_per_bike
    return (
        inventory_sequence,
        redistribution_matrices,
        total_redistrib_cost,
        total_imbalance_cost,
        unmet_cost,
        int(total_unmet),
    )


def _evaluate(
    individual: List[float],
    T: int,
    S: int,
    demand: np.ndarray,
    capacity: np.ndarray,
    distance_matrix: np.ndarray,
    initial_inventory: np.ndarray,
    cost_per_km: float,
    penalty_per_bike: float,
    scaled_return_rates: np.ndarray,
    return_lag_hours: int,
    return_allocation_matrix: np.ndarray,
    inventory_targets: np.ndarray,
    inventory_imbalance_penalty: float,
    max_redistribution_per_period: Optional[float],
) -> Tuple[float]:
    _, _, redistrib_cost, imbalance_cost, unmet_cost, _ = _simulate_trajectory(
        individual,
        T,
        S,
        demand,
        capacity,
        distance_matrix,
        initial_inventory,
        cost_per_km,
        penalty_per_bike,
        scaled_return_rates,
        return_lag_hours,
        return_allocation_matrix,
        inventory_targets,
        inventory_imbalance_penalty,
        max_redistribution_per_period,
    )
    return (float(redistrib_cost + imbalance_cost + unmet_cost),)


def _simulate_best(
    individual: List[float],
    T: int,
    S: int,
    demand: np.ndarray,
    capacity: np.ndarray,
    distance_matrix: np.ndarray,
    initial_inventory: np.ndarray,
    cost_per_km: float,
    penalty_per_bike: float,
    scaled_return_rates: np.ndarray,
    return_lag_hours: int,
    return_allocation_matrix: np.ndarray,
    inventory_targets: np.ndarray,
    inventory_imbalance_penalty: float,
    max_redistribution_per_period: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, float, float, float, int]:
    """Simulate the decoded individual and return arrays needed to construct a Solution."""
    return _simulate_trajectory(
        individual,
        T,
        S,
        demand,
        capacity,
        distance_matrix,
        initial_inventory,
        cost_per_km,
        penalty_per_bike,
        scaled_return_rates,
        return_lag_hours,
        return_allocation_matrix,
        inventory_targets,
        inventory_imbalance_penalty,
        max_redistribution_per_period,
    )


def _adaptive_params(T: int, S: int) -> Tuple[int, int]:
    """Return (pop_size, n_gen) scaled by problem size."""
    genome_len = (T - 1) * S * (S - 1)
    if genome_len <= 200:
        return 140, 160
    if genome_len <= 2000:
        return 110, 120
    return 70, 80


class GASolver:
    """Genetic algorithm solver for bike-sharing redistribution.

    Exposes the same `.solve()` interface as MIPSolver and returns a Solution dataclass.
    Uses a direct flow chromosome, demand-aware heuristic seeding, and DEAP evolution.
    """

    def __init__(
        self,
        demand_sequence: np.ndarray,
        capacity: np.ndarray,
        distance_matrix: np.ndarray,
        initial_inventory: Optional[np.ndarray] = None,
        constraints: Optional[ConstraintSet] = None,
        cost_per_km: float = COST_PER_KM,
        penalty_per_bike: float = PENALTY_PER_BIKE,
        pop_size: Optional[int] = None,
        n_gen: Optional[int] = None,
        cx_prob: float = 0.6,
        mut_prob: float = 0.2,
        seed: int = 42,
    ):
        self.demand_sequence = np.array(demand_sequence, dtype=float)
        self.capacity = np.array(capacity, dtype=float)
        self.distance_matrix = np.array(distance_matrix, dtype=float)
        self.constraints = constraints
        self.cost_per_km = cost_per_km
        self.penalty_per_bike = penalty_per_bike
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.seed = seed

        self.T, self.S = self.demand_sequence.shape

        self.initial_inventory = (
            self.capacity * INITIAL_INVENTORY_RATIO
            if initial_inventory is None
            else np.array(initial_inventory, dtype=float)
        )

        reference_solver = MIPSolver(
            demand_sequence=self.demand_sequence,
            capacity=self.capacity,
            distance_matrix=self.distance_matrix,
            initial_inventory=self.initial_inventory,
            constraints=self.constraints,
            cost_per_km=self.cost_per_km,
            penalty_per_bike=self.penalty_per_bike,
        )
        self.return_lag_hours = reference_solver.return_lag_hours
        self.scaled_return_rates = np.array(
            [
                reference_solver.return_rate_scale
                * reference_solver._hourly_return_rate(hour)
                for hour in range(24)
            ],
            dtype=float,
        )
        self.return_allocation_matrix = reference_solver.return_allocation_matrix.copy()
        self.inventory_targets = reference_solver.inventory_targets.copy()
        self.inventory_imbalance_penalty = reference_solver.inventory_imbalance_penalty
        self.max_redistribution_per_period = (
            reference_solver.max_redistribution_per_period
        )

        default_pop, default_gen = _adaptive_params(self.T, self.S)
        self.pop_size = pop_size if pop_size is not None else default_pop
        self.n_gen = n_gen if n_gen is not None else default_gen

        self._genome_len = (self.T - 1) * self.S * (self.S - 1)
        self._gene_upper_bound = float(np.max(self.capacity))
        self._max_gene = float(max(np.percentile(self.demand_sequence, 90), 1.0))

    def _build_toolbox(self) -> base.Toolbox:
        if not hasattr(creator, "FitnessMinGA"):
            creator.create("FitnessMinGA", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "IndividualGA"):
            creator.create("IndividualGA", list, fitness=creator.FitnessMinGA)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0.0, self._max_gene)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.IndividualGA,
            toolbox.attr_float,
            n=self._genome_len,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        eval_kwargs = dict(
            T=self.T,
            S=self.S,
            demand=self.demand_sequence,
            capacity=self.capacity,
            distance_matrix=self.distance_matrix,
            initial_inventory=self.initial_inventory,
            cost_per_km=self.cost_per_km,
            penalty_per_bike=self.penalty_per_bike,
            scaled_return_rates=self.scaled_return_rates,
            return_lag_hours=self.return_lag_hours,
            return_allocation_matrix=self.return_allocation_matrix,
            inventory_targets=self.inventory_targets,
            inventory_imbalance_penalty=self.inventory_imbalance_penalty,
            max_redistribution_per_period=self.max_redistribution_per_period,
        )
        toolbox.register("evaluate", _evaluate, **eval_kwargs)
        toolbox.register("mate", tools.cxBlend, alpha=0.2)
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=max(1.0, self._max_gene * 0.2),
            indpb=0.05,
        )
        toolbox.register("select", tools.selTournament, tournsize=3)
        return toolbox

    def _clip_genes(self, individual):
        for i in range(len(individual)):
            individual[i] = min(
                self._gene_upper_bound,
                max(0.0, individual[i]),
            )
        return individual

    def _build_seed_genomes(self) -> List[List[float]]:
        strategies: Sequence[Tuple[int, float, float]] = (
            (1, 0.00, 0.00),
            (3, 0.05, 0.35),
            (6, 0.08, 0.25),
        )
        seeds: List[List[float]] = []

        for lookahead, reserve_ratio, future_scale in strategies:
            heuristic_flows = _build_heuristic_flows(
                demand=self.demand_sequence,
                capacity=self.capacity,
                distance_matrix=self.distance_matrix,
                initial_inventory=self.initial_inventory,
                lookahead=lookahead,
                reserve_ratio=reserve_ratio,
                future_scale=future_scale,
            )
            genes = _encode(heuristic_flows, self.T, self.S)
            seeds.append(genes)
            noisy = [
                min(
                    self._gene_upper_bound,
                    max(0.0, gene + random.gauss(0.0, max(1.0, self._max_gene * 0.05))),
                )
                for gene in genes
            ]
            seeds.append(noisy)

        return seeds

    def _build_population(self, toolbox: base.Toolbox):
        population = toolbox.population(n=max(self.pop_size, 0))
        seed_genomes = self._build_seed_genomes()
        seed_count = min(len(seed_genomes), len(population))
        for index in range(seed_count):
            population[index][:] = seed_genomes[index]
        return population

    def solve(self) -> Solution:
        random.seed(self.seed)
        np.random.seed(self.seed)

        toolbox = self._build_toolbox()

        pop = self._build_population(toolbox)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)

        t0 = time.time()

        original_mutate = toolbox.mutate

        def mutate_and_clip(individual):
            result = original_mutate(individual)
            self._clip_genes(individual)
            return result

        toolbox.unregister("mutate")
        toolbox.register("mutate", mutate_and_clip)

        original_mate = toolbox.mate

        def mate_and_clip(left, right):
            result = original_mate(left, right)
            self._clip_genes(left)
            self._clip_genes(right)
            return result

        toolbox.unregister("mate")
        toolbox.register("mate", mate_and_clip)

        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=self.pop_size,
            lambda_=self.pop_size,
            cxpb=self.cx_prob,
            mutpb=self.mut_prob,
            ngen=self.n_gen,
            stats=stats,
            halloffame=hof,
            verbose=False,
        )

        solve_time = time.time() - t0
        best = hof[0]

        (
            inventory_sequence,
            redistribution_matrices,
            redistrib_cost,
            imbalance_cost,
            unmet_cost,
            total_unmet,
        ) = _simulate_best(
            best,
            self.T,
            self.S,
            self.demand_sequence,
            self.capacity,
            self.distance_matrix,
            self.initial_inventory,
            self.cost_per_km,
            self.penalty_per_bike,
            self.scaled_return_rates,
            self.return_lag_hours,
            self.return_allocation_matrix,
            self.inventory_targets,
            self.inventory_imbalance_penalty,
            self.max_redistribution_per_period,
        )

        return Solution(
            cost=float(redistrib_cost + imbalance_cost + unmet_cost),
            redistribution_cost=float(redistrib_cost),
            unmet_demand_cost=float(unmet_cost),
            total_unmet_bikes=total_unmet,
            inventory_sequence=inventory_sequence,
            redistribution_matrices=redistribution_matrices,
            status="GA",
            solve_time=float(solve_time),
            imbalance_cost=float(imbalance_cost),
        )
