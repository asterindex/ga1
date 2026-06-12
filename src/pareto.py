"""
NSGA-II utilities for multi-objective GA.

Three objectives per Chromosome:
  1. val_accuracy  (maximize)
  2. num_params    (minimize)
  3. training_time (minimize)
"""

from typing import List, Tuple
import math


# ---------------------------------------------------------------------------
# Dominance
# ---------------------------------------------------------------------------

def _objectives(ind) -> Tuple[float, float, float]:
    """
    Returns objectives normalised so that LOWER is always BETTER:
      - accuracy   -> 1 - accuracy
      - num_params -> num_params  (already lower-is-better)
      - time       -> training_time
    """
    return (
        1.0 - ind.fitness,
        float(ind.num_params),
        float(ind.training_time),
    )


def dominates(a, b) -> bool:
    """
    Return True if chromosome `a` Pareto-dominates `b`.
    a dominates b iff a is no worse in all objectives and strictly better
    in at least one.
    """
    obj_a = _objectives(a)
    obj_b = _objectives(b)
    at_least_one_better = False
    for va, vb in zip(obj_a, obj_b):
        if va > vb:       # a is worse in this objective
            return False
        if va < vb:       # a is strictly better in this objective
            at_least_one_better = True
    return at_least_one_better


# ---------------------------------------------------------------------------
# Fast non-dominated sort  (NSGA-II, Deb 2002)
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(population: list) -> List[List[int]]:
    """
    Partition `population` into Pareto fronts F1, F2, …

    Returns:
        List of fronts; each front is a list of indices into `population`.
        Front 0 is the Pareto-optimal (non-dominated) set.
    """
    n = len(population)
    dominated_by: List[List[int]] = [[] for _ in range(n)]   # S_i: indices dominated by i
    domination_count: List[int] = [0] * n                    # n_i: how many dominate i

    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i], population[j]):
                dominated_by[i].append(j)
            elif dominates(population[j], population[i]):
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

    # Remove the trailing empty front
    if fronts and not fronts[-1]:
        fronts.pop()

    return fronts


# ---------------------------------------------------------------------------
# Crowding distance
# ---------------------------------------------------------------------------

def crowding_distance(population: list, front_indices: List[int]) -> List[float]:
    """
    Compute crowding distance for individuals in a front.

    Returns:
        distances[i] = crowding distance for population[front_indices[i]]
    """
    m = len(front_indices)
    if m <= 2:
        return [math.inf] * m

    distances = [0.0] * m

    # Compute per-objective contributions
    for obj_fn in [
        lambda x: x.fitness,                          # accuracy – higher is more spread
        lambda x: -float(x.num_params),               # negate so higher = less params
        lambda x: -float(x.training_time),            # negate so higher = less time
    ]:
        values = [obj_fn(population[idx]) for idx in front_indices]
        sorted_order = sorted(range(m), key=lambda k: values[k])

        # Boundaries get infinite distance
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


# ---------------------------------------------------------------------------
# Pareto-based selection
# ---------------------------------------------------------------------------

def pareto_select(population: list, n: int) -> list:
    """
    Select `n` individuals from `population` using NSGA-II crowded-comparison.

    Selection pressure:
      1. Lower Pareto-front rank is preferred.
      2. Among individuals in the same front, higher crowding distance is preferred
         (promotes diversity).

    Returns:
        List of `n` selected Chromosome objects (copies are NOT made here).
    """
    fronts = fast_non_dominated_sort(population)

    selected: List = []
    rank_map: dict = {}       # individual index -> front rank
    distance_map: dict = {}   # individual index -> crowding distance

    for rank, front in enumerate(fronts):
        if len(selected) >= n:
            break

        distances = crowding_distance(population, front)

        remaining = n - len(selected)
        if len(front) <= remaining:
            # Whole front fits
            for k, idx in enumerate(front):
                selected.append(population[idx])
                rank_map[idx] = rank
                distance_map[idx] = distances[k]
        else:
            # Partial front: pick by descending crowding distance
            ranked_front = sorted(
                range(len(front)),
                key=lambda k: distances[k],
                reverse=True
            )
            for k in ranked_front[:remaining]:
                idx = front[k]
                selected.append(population[idx])
                rank_map[idx] = rank
                distance_map[idx] = distances[k]

    return selected


def assign_ranks(population: list) -> None:
    """
    Assign `.pareto_rank` and `.crowding_distance` attributes to each individual
    in-place (used for logging / history saving).
    """
    fronts = fast_non_dominated_sort(population)
    for rank, front in enumerate(fronts):
        distances = crowding_distance(population, front)
        for k, idx in enumerate(front):
            population[idx].pareto_rank = rank
            population[idx].crowding_distance = distances[k]
