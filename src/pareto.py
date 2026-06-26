"""
NSGA-II utilities for multi-objective GA.

Objective modes:
  standard  - val_accuracy (max), num_params (min), training_time (min)
  hardware  - val_accuracy (max), inference_latency_ms (min),
              model_size_bytes (min), peak_ram_mb (min)
"""

from typing import List, Tuple, Callable
import math

import config


def _objectives(ind, mode: str = config.OBJECTIVE_MODE_STANDARD) -> Tuple[float, ...]:
    """
    Returns objectives normalised so that LOWER is always BETTER.
    """
    if mode == config.OBJECTIVE_MODE_HARDWARE:
        return (
            1.0 - ind.fitness,
            float(ind.inference_latency_ms),
            float(ind.model_size_bytes),
            float(ind.peak_ram_mb),
        )

    return (
        1.0 - ind.fitness,
        float(ind.num_params),
        float(ind.training_time),
    )


def _crowding_objective_fns(mode: str) -> List[Callable]:
    """Objective value functions for crowding distance (higher = better spread)."""
    if mode == config.OBJECTIVE_MODE_HARDWARE:
        return [
            lambda x: x.fitness,
            lambda x: -float(x.inference_latency_ms),
            lambda x: -float(x.model_size_bytes),
            lambda x: -float(x.peak_ram_mb),
        ]

    return [
        lambda x: x.fitness,
        lambda x: -float(x.num_params),
        lambda x: -float(x.training_time),
    ]


def dominates(a, b, mode: str = config.OBJECTIVE_MODE_STANDARD) -> bool:
    """
    Return True if chromosome `a` Pareto-dominates `b`.
    """
    obj_a = _objectives(a, mode)
    obj_b = _objectives(b, mode)
    at_least_one_better = False
    for va, vb in zip(obj_a, obj_b):
        if va > vb:
            return False
        if va < vb:
            at_least_one_better = True
    return at_least_one_better


def fast_non_dominated_sort(population: list,
                            mode: str = config.OBJECTIVE_MODE_STANDARD) -> List[List[int]]:
    """
    Partition `population` into Pareto fronts F1, F2, …
    """
    n = len(population)
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    domination_count: List[int] = [0] * n

    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i], population[j], mode):
                dominated_by[i].append(j)
            elif dominates(population[j], population[i], mode):
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front: List[int] = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    if fronts and not fronts[-1]:
        fronts.pop()

    return fronts


def crowding_distance(population: list, front_indices: List[int],
                      mode: str = config.OBJECTIVE_MODE_STANDARD) -> List[float]:
    """Compute crowding distance for individuals in a front."""
    m = len(front_indices)
    if m <= 2:
        return [math.inf] * m

    distances = [0.0] * m
    objective_fns = _crowding_objective_fns(mode)

    for obj_fn in objective_fns:
        values = [obj_fn(population[idx]) for idx in front_indices]
        sorted_order = sorted(range(m), key=lambda k: values[k])

        distances[sorted_order[0]] = math.inf
        distances[sorted_order[-1]] = math.inf

        obj_range = values[sorted_order[-1]] - values[sorted_order[0]]
        if obj_range == 0:
            continue

        for k in range(1, m - 1):
            distances[sorted_order[k]] += (
                values[sorted_order[k + 1]] - values[sorted_order[k - 1]]
            ) / obj_range

    return distances


def pareto_select(population: list, n: int,
                  mode: str = config.OBJECTIVE_MODE_STANDARD) -> list:
    """Select `n` individuals using NSGA-II crowded-comparison."""
    fronts = fast_non_dominated_sort(population, mode)

    selected: List = []

    for rank, front in enumerate(fronts):
        if len(selected) >= n:
            break

        distances = crowding_distance(population, front, mode)

        remaining = n - len(selected)
        if len(front) <= remaining:
            for k, idx in enumerate(front):
                selected.append(population[idx])
        else:
            ranked_front = sorted(
                range(len(front)),
                key=lambda k: distances[k],
                reverse=True
            )
            for k in ranked_front[:remaining]:
                idx = front[k]
                selected.append(population[idx])

    return selected


def assign_ranks(population: list,
                 mode: str = config.OBJECTIVE_MODE_STANDARD) -> None:
    """Assign `.pareto_rank` and `.crowding_distance` in-place."""
    fronts = fast_non_dominated_sort(population, mode)
    for rank, front in enumerate(fronts):
        distances = crowding_distance(population, front, mode)
        for k, idx in enumerate(front):
            population[idx].pareto_rank = rank
            population[idx].crowding_distance = distances[k]
