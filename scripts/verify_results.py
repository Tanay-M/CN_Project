"""Audit actual saved experiments, independently of the planner's decisions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from carbonroute.common import ROOT, load_json, sha256


def verify(output: Path) -> None:
    manifest = load_json(output / "manifest.json")
    assert manifest["status"] == "complete" and not manifest["failed_runs"]
    assert manifest["schema_version"] == 2
    if manifest.get("include_joint"):
        model_hashes = manifest["checkpoint_sha256_by_model"]
        training = load_json(output / "training_summary.json")
        for name in ("gnn", "gru"):
            assert model_hashes[name] == training["models"][name]["checkpoint_sha256"]
        assert len(model_hashes["persistence"]) == 64
    for year, expected_hash in manifest["alignment_sha256"].items():
        assert sha256(output / f"alignment_{year}.json") == expected_hash
    frame = pd.read_csv(output / "summary.csv")
    assert sha256(output / "summary.csv") == manifest["summary_sha256"]
    assert set(frame.run_id) == {r["run_id"] for r in manifest["runs"]}
    assert not frame.run_id.duplicated().any()
    graph = manifest["graph"]
    arcs = [tuple(a) for a in graph["arcs"]]
    pairs = manifest["traffic_manifest"]["pairs"]
    assert len(graph["node_ids"]) == 8 and len(arcs) == 22
    for _, group in frame.groupby(["window", "scenario", "carbon_year", "observation_delay_bins"]):
        assert group.traffic_sha256.nunique() == 1
        assert group.offered_mb.max()-group.offered_mb.min() < 1e-7
        assert group.idle_only_g.max()-group.idle_only_g.min() < 1e-7
    steps = 0
    for run in manifest["runs"]:
        assert sha256(output / run["file"]) == run["sha256"]
        payload = load_json(output / run["file"])
        records = payload["records"]
        offered_mb = served_mb = carbon_numerator = 0.
        budget_violations = 0
        for row in records:
            if "carbon_budget_met" in row:
                cost = row["predicted_carbon_cost"]
                budget = row["reactive_carbon_budget"]
                assert cost is not None and budget is not None
                assert row["carbon_budget_met"] == (cost <= budget + 1e-9)
                assert row["joint_assignments_evaluated"] > 0
                budget_violations += not row["carbon_budget_met"]
            loads = np.zeros(22)
            for i, route in enumerate(row["routes"]):
                assert (route[0], route[-1]) == tuple(pairs[i])
                assert len(set(route)) == len(route)
                assert 0 <= row["served_mbps"][i] <= row["offered_mbps"][i]+1e-7
                flow_mb = row["served_mbps"][i]*900/8
                for a, b in zip(route, route[1:]):
                    loads[arcs.index((a, b))] += row["served_mbps"][i]
                    carbon_numerator += flow_mb*row["carbon"][a]
            assert loads.max() <= graph["capacity_mbps"]+1e-7
            offered_mb += sum(row["offered_mbps"])*900/8
            served_mb += sum(row["served_mbps"])*900/8
            observed, target = np.datetime64(row["observed_through"]), np.datetime64(row["traffic_time"])
            assert observed == target-np.timedelta64(run["observation_delay_bins"]*15, "m")
            steps += 1
        summary = payload["summary"]
        if "carbon_budget_violations" in summary:
            assert summary["carbon_budget_violations"] == budget_violations
            assert budget_violations == 0, run["run_id"]
        np.testing.assert_allclose([offered_mb, served_mb, carbon_numerator],
                                   [summary["offered_mb"], summary["served_mb"], summary["carbon_weighted_mb"]], rtol=1e-7)
    print(f"PASS: {len(frame)} paired runs, {steps} intervals; input parity, routes, capacity, timing, carbon accounting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "results/paper")
    verify(parser.parse_args().output)
