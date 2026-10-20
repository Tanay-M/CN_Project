"""Causal aggregate replay: commit routes before revealing the target demand."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import finite_nonnegative
from .data import Prepared
from .prediction import Predictor
from .routing import JOINT_POLICIES, BudgetPlanner, Planner
from .service import forwarding_score, power_model_grams, progressive_fill


@dataclass
class ReplayResult:
    summary: dict[str, Any]
    records: list[dict[str, Any]]


def replay(data: Prepared, cfg: dict[str, Any], rates: np.ndarray, carbon: np.ndarray,
           start: int, stop: int, policy: str, predictor: Predictor,
           observation_delay: int = 1, carbon_budget_fraction: float = 0.0) -> ReplayResult:
    """Rates may contain the full realization; only the driver can index target rows.

    delay=1: history ends at k-1. delay=0: everyone gets D[k], with GNN predicting
    k+1 onward. Both use only current-day carbon, held constant over the horizon.
    """
    lookback, horizon = cfg["lookback"], cfg["horizon"]
    if observation_delay not in (0, 1):
        raise ValueError("Supported observation delays are 0 and 1 bins")
    if start < lookback or stop > len(rates) or stop <= start:
        raise ValueError("Invalid replay interval/history")
    if rates.shape != data.rates.shape or carbon.shape != (stop-start, 8):
        raise ValueError("Replay shapes do not match prepared data")
    finite_nonnegative(rates, "replay rates")
    finite_nonnegative(carbon, "replay carbon")
    if np.any(rates[:, data.mask == 0] != 0):
        raise ValueError("Inactive demands must be masked")
    if not np.all(np.diff(data.times[start-lookback:stop]) == np.timedelta64(15, "m")):
        raise ValueError("Replay must not cross missing timestamps")
    joint = policy in JOINT_POLICIES
    planner = (BudgetPlanner(data.network, data.pairs, cfg, carbon_budget_fraction) if joint
               else Planner(data.network, data.pairs, cfg, policy))
    records: list[dict[str, Any]] = []
    source_nodes = np.array([a for a, b in data.network.arcs])
    for k in range(start, stop):
        tick = time.perf_counter()
        observed_stop = k + (observation_delay == 0)
        history = rates[observed_stop-lookback:observed_stop].copy()
        history.setflags(write=False)
        reactive_forecast = np.repeat(history[-1:], horizon, axis=0)
        if policy in ("shortest_path", "reactive_carbon"):
            forecast = reactive_forecast
        else:
            forecast = predictor.predict(history)
            if policy not in ("predictive_te", "no_margin"):
                forecast = forecast + predictor.margin
            if observation_delay == 0:
                forecast = np.concatenate((history[-1:], forecast[:horizon-1]), axis=0)
        ci = carbon[k-start].copy()
        if isinstance(planner, BudgetPlanner):
            decision = planner.decide(forecast, ci, k-start, reactive_forecast)
        else:
            decision = planner.decide(forecast, ci, k-start)
        decision_ms = (time.perf_counter()-tick) * 1000
        # Reveal actual target demand ONLY after assignment is committed.
        offered = rates[k, [a for a, b in data.pairs], [b for a, b in data.pairs]]
        routes = list(decision.routes)
        incidence = data.network.incidence(routes)
        offered_links = incidence @ offered
        served = progressive_fill(offered, incidence, data.network.capacities)
        served_links = incidence @ served
        if np.any(served_links > data.network.capacities+1e-7):
            raise AssertionError("Service violates link capacity")
        utilization = offered_links / data.network.capacities
        costs = incidence.T @ ci[source_nodes]
        accounting = forwarding_score(served, costs, cfg["bin_seconds"])
        outgoing = np.bincount(source_nodes, weights=served_links, minlength=8)
        budget_evidence = ({"predicted_carbon_cost": decision.predicted_carbon_cost,
                            "reactive_carbon_budget": decision.reactive_carbon_budget,
                            "carbon_budget_met": decision.carbon_budget_met,
                            "joint_assignments_evaluated": decision.joint_assignments_evaluated}
                           if joint else {})
        records.append({"step": k-start, "traffic_time": str(data.times[k]),
                        "observed_through": str(data.times[observed_stop-1]),
                        "offered_mbps": offered.tolist(), "served_mbps": served.tolist(),
                        "link_utilization": utilization.tolist(), "carbon": ci.tolist(),
                        "routes": [list(p) for p in routes], "changes": decision.changes,
                        "infeasible_pairs": decision.infeasible_pairs, "reasons": list(decision.reasons),
                        "projected_max_utilization": decision.projected_max_utilization,
                        "decision_ms": decision_ms, **accounting,
                        "offered_mb": float(offered.sum() * cfg["bin_seconds"] / 8),
                        "excess_mbit": float(np.maximum(offered_links-data.network.capacities, 0).sum()
                                              * cfg["bin_seconds"]),
                        "idle_only_g": power_model_grams(outgoing, ci, cfg["bin_seconds"], 50, 0),
                        "idle_dominated_g": power_model_grams(outgoing, ci, cfg["bin_seconds"], 50, .1),
                        "load_sensitive_g": power_model_grams(outgoing, ci, cfg["bin_seconds"], 5, 1),
                        **budget_evidence})
    util = np.array([r["link_utilization"] for r in records])
    served_mb = sum(r["served_mb"] for r in records)
    offered_mb = sum(r["offered_mb"] for r in records)
    weighted = sum(r["carbon_weighted_mb"] for r in records)
    summary = {"policy": policy, "observation_delay_bins": observation_delay,
               "bins": len(records), "offered_mb": offered_mb, "served_mb": served_mb,
               "served_fraction": served_mb/offered_mb if offered_mb else 1.0,
               "carbon_weighted_mb": weighted,
               "forwarding_score": weighted/served_mb if served_mb else None,
               "max_utilization": float(util.max()), "p95_utilization": float(np.quantile(util, .95)),
               "overload_minutes": float((util.max(axis=1) > 1+1e-9).sum()*cfg["bin_seconds"]/60),
               "excess_mbit": sum(r["excess_mbit"] for r in records),
               "route_changes": sum(r["changes"] for r in records),
               "infeasible_pair_bins": sum(r["infeasible_pairs"] for r in records),
               "decision_ms_mean": float(np.mean([r["decision_ms"] for r in records])),
               "decision_ms_p95": float(np.quantile([r["decision_ms"] for r in records], .95)),
               "idle_only_g": sum(r["idle_only_g"] for r in records),
               "idle_dominated_g": sum(r["idle_dominated_g"] for r in records),
               "load_sensitive_g": sum(r["load_sensitive_g"] for r in records)}
    if joint:
        summary["carbon_budget_violations"] = sum(not r["carbon_budget_met"] for r in records)
        summary["joint_assignments_evaluated"] = records[0]["joint_assignments_evaluated"]
    return ReplayResult(summary, records)
