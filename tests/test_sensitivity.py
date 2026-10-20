"""Unit and invariant tests for Parameter Sensitivity and Pareto Frontier (Milestone 3)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from carbonroute.common import ROOT, config, load_json, sha256

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from carbonroute.data import load_prepared
from carbonroute.prediction import TrainedPredictor, calibrate_margin
from carbonroute.routing import Planner
from carbonroute.sensitivity import (
    DEFAULT_ALPHAS,
    DEFAULT_TAUS,
    get_cached_validation_residuals,
    is_pareto_efficient_tri_objective,
)
from tests.test_e2e_requirements import validate_sensitivity_summary_schema


def test_margin_monotonicity_across_alphas():
    """Mathematical invariant: positive validation residual margin is non-decreasing with alpha."""
    cfg = config()
    prep_manifest = load_json(ROOT / "inputs/processed/manifest.json")
    prep_cfg = dict(cfg, seed=prep_manifest["config"].get("seed", cfg["seed"]))
    data = load_prepared(ROOT, prep_cfg)

    seed_art = ROOT / "artifacts/seeds/seed_42"
    art_dir = seed_art if seed_art.exists() and (seed_art / "gnn.pt").exists() else ROOT / "artifacts"
    predictor = TrainedPredictor(art_dir, data, prep_cfg, seed=42)

    yval, val_pred = get_cached_validation_residuals(data, predictor, prep_cfg)

    margins = [calibrate_margin(yval, val_pred, data.mask, a) for a in DEFAULT_ALPHAS]

    for i in range(len(DEFAULT_ALPHAS) - 1):
        diff = margins[i + 1] - margins[i]
        assert np.all(diff >= -1e-7), (
            f"Margin monotonicity violated between alpha={DEFAULT_ALPHAS[i]} and {DEFAULT_ALPHAS[i+1]}"
        )


def test_hysteresis_route_flaps_monotonicity_across_taus():
    """Behavioral invariant: increasing hysteresis threshold tau monotonically dampens route flaps."""
    cfg = config()
    prep_manifest = load_json(ROOT / "inputs/processed/manifest.json")
    prep_cfg = dict(cfg, seed=prep_manifest["config"].get("seed", cfg["seed"]))
    data = load_prepared(ROOT, prep_cfg)

    ci = np.array([180.0, 350.0, 120.0, 290.0, 95.0, 240.0, 310.0, 110.0])
    forecast = np.zeros((3, 8, 8))
    for a, b in data.pairs:
        forecast[:, a, b] = 2.5

    changes_recorded = []
    for tau in DEFAULT_TAUS:
        step_cfg = dict(prep_cfg, min_improvement=tau)
        planner = Planner(data.network, data.pairs, step_cfg, "predictive_carbon")
        decision = planner.decide(forecast, ci, step=0)
        changes_recorded.append(decision.changes)

    for i in range(len(DEFAULT_TAUS) - 1):
        assert changes_recorded[i] >= changes_recorded[i + 1], (
            f"Route change monotonicity violated: tau={DEFAULT_TAUS[i]} gave {changes_recorded[i]} changes, "
            f"but tau={DEFAULT_TAUS[i+1]} gave {changes_recorded[i+1]} changes"
        )


def test_pareto_frontier_points_are_non_dominated():
    """Verifies that all points in results/sensitivity/pareto_frontier.json form a valid Pareto set."""
    frontier_path = ROOT / "results/sensitivity/pareto_frontier.json"
    assert frontier_path.exists(), "pareto_frontier.json must exist"

    pareto_data = load_json(frontier_path)
    assert len(pareto_data) >= 3, f"Expected at least 3 non-dominated Pareto points, got {len(pareto_data)}"

    costs = np.array([
        [p["forwarding_score"], p["route_changes"], p["overload_minutes"]]
        for p in pareto_data
    ])
    mask = is_pareto_efficient_tri_objective(costs)
    assert mask.all(), f"Pareto frontier contains dominated points at indices: {np.where(~mask)[0]}"


def test_pareto_frontier_non_dominance_edge_cases():
    """Verifies mathematical correctness of tri-objective non-dominated sorting on edge cases."""
    pts = np.array([
        [10.0, 10.0, 10.0],  # A: optimal
        [10.0, 10.0, 10.0],  # A_dup: identical to A (non-dominated)
        [12.0, 10.0, 10.0],  # B: dominated by A
        [10.0, 15.0, 10.0],  # C: dominated by A
        [10.0, 10.0, 15.0],  # D: dominated by A
        [8.0, 12.0, 12.0],   # E: trade-off point (better carbon, worse changes/overload)
        [12.0, 8.0, 12.0],   # F: trade-off point (better changes, worse carbon/overload)
        [12.0, 12.0, 8.0],   # G: trade-off point (better overload, worse carbon/changes)
    ])
    mask = is_pareto_efficient_tri_objective(pts)
    expected = np.array([True, True, False, False, False, True, True, True])
    assert np.array_equal(mask, expected), f"Sorting failed: got {mask}, expected {expected}"


def test_sensitivity_summary_schema_and_bounds():
    """Validates summary.csv schema, column ranges, and physical bounds."""
    summary_path = ROOT / "results/sensitivity/summary.csv"
    assert summary_path.exists(), "results/sensitivity/summary.csv must exist"

    df = pd.read_csv(summary_path)
    validate_sensitivity_summary_schema(df)

    assert len(df) == 196, f"Expected 196 sensitivity sweep runs, found {len(df)}"
    assert (df["served_fraction"] >= 0.0).all() and (df["served_fraction"] <= 1.00001).all()
    assert (df["overload_minutes"] >= 0.0).all()
    assert (df["forwarding_score"] > 0.0).all()
    assert (df["route_changes"] >= 0).all()


def test_manifest_v2_and_checksum_integrity():
    """Validates Schema v2 manifest compliance and sha256 checksums."""
    manifest_path = ROOT / "results/sensitivity/manifest.json"
    assert manifest_path.exists(), "results/sensitivity/manifest.json must exist"

    manifest = load_json(manifest_path)
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "complete"
    assert not manifest["failed_runs"]
    assert len(manifest["runs"]) == 196

    summary_path = ROOT / "results/sensitivity/summary.csv"
    assert sha256(summary_path) == manifest["summary_sha256"]

    alignment_file = ROOT / "results/sensitivity/alignment_2022.json"
    assert alignment_file.exists()
    assert sha256(alignment_file) == manifest["alignment_sha256"]["2022"]


def test_publication_figures_exist_and_non_empty():
    """Verifies that all 4 required publication figures exist in PNG and PDF formats."""
    expected_figures = [
        "sensitivity_alpha_curves.png",
        "sensitivity_alpha_curves.pdf",
        "sensitivity_tau_curves.png",
        "sensitivity_tau_curves.pdf",
        "pareto_carbon_overload.png",
        "pareto_carbon_overload.pdf",
        "pareto_3d_frontier.png",
        "pareto_3d_frontier.pdf",
    ]

    for d in [ROOT / "results/sensitivity/figures", ROOT / "figures"]:
        assert d.exists(), f"Figure directory {d} missing"
        for fig_name in expected_figures:
            fig_path = d / fig_name
            assert fig_path.exists(), f"Missing publication figure: {fig_path}"
            assert fig_path.stat().st_size > 1000, f"Publication figure {fig_path} suspiciously small"
