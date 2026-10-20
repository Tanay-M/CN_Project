"""Deterministic candidate-path planning with shared reservations and hysteresis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any

import numpy as np

from .common import finite_nonnegative
from .network import Network, Pair, Route

POLICIES = ("shortest_path", "reactive_carbon", "predictive_te", "predictive_carbon",
            "persistence_carbon", "no_margin", "no_hysteresis")
JOINT_POLICIES = ("joint_gnn", "joint_gru", "joint_persistence")


@dataclass(frozen=True)
class Decision:
    routes: tuple[Route, ...]
    changes: int
    infeasible_pairs: int
    projected_max_utilization: float
    reasons: tuple[str, ...]
    predicted_carbon_cost: float | None = None
    reactive_carbon_budget: float | None = None
    carbon_budget_met: bool | None = None
    joint_assignments_evaluated: int | None = None


class Planner:
    def __init__(self, network: Network, pairs: list[Pair], cfg: dict[str, Any], policy: str):
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy: {policy}")
        self.network, self.pairs, self.cfg, self.policy = network, pairs, cfg, policy
        self.candidates = network.candidates(pairs, cfg["candidate_paths"], cfg["delay_stretch"])
        self.routes = [paths[0] for paths in self.candidates]
        self.last_changed = np.full(len(pairs), -cfg["hold_bins"], dtype=int)
        self.vectors = [{p: network.incidence([p])[:, 0] for p in paths} for paths in self.candidates]

    def decide(self, forecast: np.ndarray, current_carbon: np.ndarray, step: int) -> Decision:
        """Receives predicted demand and current carbon ONLY; no actual future rows."""
        if forecast.shape != (self.cfg["horizon"], 8, 8) or current_carbon.shape != (8,):
            raise ValueError("Forecast/carbon shape mismatch")
        finite_nonnegative(forecast, "forecast")
        finite_nonnegative(current_carbon, "carbon")
        demand = forecast[:, [a for a, b in self.pairs], [b for a, b in self.pairs]]
        loads = demand @ self.network.incidence(self.routes).T
        if self.policy == "shortest_path":
            return Decision(tuple(self.routes), 0, 0,
                            float((loads/self.network.capacities).max()), tuple("fixed" for _ in self.pairs))
        order = sorted(range(len(self.pairs)), key=lambda i: (-float(demand[:, i].max()), i))
        changes, infeasible = 0, 0
        reasons = ["kept"] * len(self.pairs)
        limit = self.cfg["utilization_limit"]
        for i in order:
            old = self.routes[i]
            old_vector = self.vectors[i][old]
            # Remove this flow before evaluating replacements: no double reservation.
            base = loads - demand[:, i, None] * old_vector[None]
            options = []
            for path in self.candidates[i]:
                vector = self.vectors[i][path]
                projected = base + demand[:, i, None] * vector[None]
                utilization = projected / self.network.capacities
                path_max = float(utilization[:, vector > 0].max())
                global_max = float(utilization.max())
                carbon_cost = float(sum(current_carbon[node] for node in path[:-1]))
                objective = global_max if self.policy == "predictive_te" else carbon_cost
                options.append((path, path_max <= limit+1e-9, objective, global_max, path_max))
            feasible = [o for o in options if o[1]]
            if feasible:
                best = min(feasible, key=lambda o: (o[2], o[3], self.network.delay(o[0]), o[0]))
            else:
                infeasible += 1
                best = min(options, key=lambda o: (o[3], o[4], o[2], self.network.delay(o[0]), o[0]))
            current = next(o for o in options if o[0] == old)
            emergency = not current[1]
            if best[0] != old:
                improvement = (current[2]-best[2]) / max(abs(current[2]), 1e-9)
                hold = step - self.last_changed[i] < self.cfg["hold_bins"]
                if self.policy == "no_hysteresis" or emergency or (
                    not hold and improvement >= self.cfg["min_improvement"]
                ):
                    self.routes[i] = best[0]
                    self.last_changed[i] = step
                    changes += 1
                    reasons[i] = "capacity-risk" if emergency else "objective-improvement"
                else:
                    reasons[i] = "hold-time" if hold else "small-improvement"
            if not feasible:
                reasons[i] += ";no-feasible-candidate"
            loads = base + demand[:, i, None] * self.vectors[i][self.routes[i]][None]
        return Decision(tuple(self.routes), changes, infeasible,
                        float((loads/self.network.capacities).max()), tuple(reasons))


class BudgetPlanner:
    """Select a whole-network assignment under a reactive carbon-cost budget.

    The budget uses the *same forecast* for both assignments and a separate,
    causally updated reactive planner as its reference. Costs are predicted
    offered Mbps times forwarding-sender CI, not observed emissions.
    """

    def __init__(self, network: Network, pairs: list[Pair], cfg: dict[str, Any],
                 budget_fraction: float = 0.0):
        if not 0 <= budget_fraction <= 1:
            raise ValueError("Carbon budget fraction must be between 0 and 1")
        self.network, self.pairs, self.cfg = network, pairs, cfg
        self.budget_fraction = budget_fraction
        self.candidates = network.candidates(pairs, cfg["candidate_paths"], cfg["delay_stretch"])
        count = prod(map(len, self.candidates))
        if not 0 < count <= 4096:
            raise ValueError(f"Joint selector supports 1–4096 assignments, found {count}")
        self.choices = np.array(list(product(*(range(len(c)) for c in self.candidates))), dtype=int)
        self.assignments = tuple(tuple(paths[j] for paths, j in zip(self.candidates, choice))
                                 for choice in self.choices)
        self.lookup = {routes: i for i, routes in enumerate(self.assignments)}
        self.incidence = np.stack([network.incidence(list(routes)) for routes in self.assignments])
        self.sender_counts = np.zeros((count, len(pairs), 8))
        self.delay = np.zeros(count)
        for a, routes in enumerate(self.assignments):
            for f, route in enumerate(routes):
                for node in route[:-1]:
                    self.sender_counts[a, f, node] += 1
                self.delay[a] += network.delay(route)
        self.current = 0  # The first combination contains every shortest path.
        self.last_changed = np.full(len(pairs), -cfg["hold_bins"], dtype=int)
        self.reference = Planner(network, pairs, cfg, "reactive_carbon")

    def decide(self, forecast: np.ndarray, current_carbon: np.ndarray, step: int,
               reactive_forecast: np.ndarray) -> Decision:
        horizon = self.cfg["horizon"]
        if (forecast.shape != (horizon, 8, 8) or reactive_forecast.shape != forecast.shape
                or current_carbon.shape != (8,)):
            raise ValueError("Joint forecast/carbon shape mismatch")
        finite_nonnegative(forecast, "joint forecast")
        finite_nonnegative(reactive_forecast, "reactive forecast")
        finite_nonnegative(current_carbon, "carbon")
        reference = self.reference.decide(reactive_forecast, current_carbon, step)
        reference_index = self.lookup[reference.routes]
        demand = forecast[:, [a for a, _ in self.pairs], [b for _, b in self.pairs]]
        loads = np.einsum("hf,alf->ahl", demand, self.incidence)
        utilization = loads / self.network.capacities
        peaks = utilization.max(axis=(1, 2))
        path_costs = np.einsum("afi,i->af", self.sender_counts, current_carbon)
        carbon_costs = path_costs @ demand.mean(axis=0)
        budget = float(carbon_costs[reference_index] * (1 + self.budget_fraction))
        budget_risk = bool(carbon_costs[self.current] > budget + 1e-9)

        eligible = np.ones(len(self.assignments), dtype=bool)
        current_routes = self.assignments[self.current]
        risk = np.zeros(len(self.pairs), dtype=bool)
        for f in range(len(self.pairs)):
            arcs = self.incidence[self.current, :, f] > 0
            risk[f] = bool(utilization[self.current, :, arcs].max() >
                           self.cfg["utilization_limit"] + 1e-9)
            if (step - self.last_changed[f] < self.cfg["hold_bins"]
                    and not risk[f] and not budget_risk):
                eligible &= self.choices[:, f] == self.choices[self.current, f]
        within_budget = eligible & (carbon_costs <= budget + 1e-9)
        if not within_budget.any():
            raise AssertionError("Reactive assignment must remain available within the carbon budget")
        pool = np.flatnonzero(within_budget)
        chosen = int(min(pool, key=lambda i: (peaks[i], carbon_costs[i],
                                               np.count_nonzero(self.choices[i] != self.choices[self.current]),
                                               self.delay[i], i)))
        reasons = ["kept"] * len(self.pairs)
        budget_rescue = budget_risk and carbon_costs[chosen] <= budget + 1e-9
        if chosen != self.current and not risk.any() and not budget_rescue:
            peak_gain = (peaks[self.current]-peaks[chosen]) / max(peaks[self.current], 1e-9)
            carbon_gain = ((carbon_costs[self.current]-carbon_costs[chosen]) /
                           max(carbon_costs[self.current], 1e-9))
            if (peak_gain < self.cfg["min_improvement"] and not
                    (carbon_gain >= self.cfg["min_improvement"] and
                     peaks[chosen] <= peaks[self.current] + 1e-9)):
                chosen = self.current
                reasons = ["small-improvement"] * len(self.pairs)
        changed = self.choices[chosen] != self.choices[self.current]
        for index in np.flatnonzero(changed):
            f = int(index)
            self.last_changed[f] = step
            reasons[f] = ("joint-capacity-risk" if risk[f] else
                          "joint-carbon-budget-risk" if budget_rescue else
                          "joint-objective-improvement")
        self.current = int(chosen)
        infeasible = sum(bool(utilization[chosen, :, self.incidence[chosen, :, f] > 0].max() >
                              self.cfg["utilization_limit"] + 1e-9)
                         for f in range(len(self.pairs)))
        return Decision(current_routes if not changed.any() else self.assignments[chosen],
                        int(changed.sum()), infeasible, float(peaks[chosen]), tuple(reasons),
                        float(carbon_costs[chosen]), budget,
                        bool(carbon_costs[chosen] <= budget + 1e-9), len(self.assignments))
