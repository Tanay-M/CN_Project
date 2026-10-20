"""Unit tests for multi-seed campaign helpers and statistical significance engine."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from carbonroute.common import ROOT, config
from carbonroute.experiments import compute_paired_significance, compute_statistical_tests

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tests.test_e2e_requirements import validate_statistical_significance_schema


def test_compute_statistical_tests_zero_variance_handles_gracefully():
    """Verifies that identical target and baseline return p-value=1.0 without throwing exceptions."""
    # Synthetic frame where predictive_carbon and reactive_carbon have identical metrics
    rows = []
    for s in [42, 43, 44, 45, 46]:
        for w in range(5):
            for y in [2022, 2023, 2024]:
                for pol in ["predictive_carbon", "reactive_carbon", "persistence_carbon"]:
                    rows.append({
                        "model_seed": s,
                        "window": w,
                        "carbon_year": y,
                        "scenario": "congested",
                        "observation_delay_bins": 1,
                        "policy": pol,
                        "overload_minutes": 100.0,
                        "forwarding_score": 330.0,
                        "served_fraction": 0.92,
                        "route_changes": 15.0,
                    })

    df = pd.DataFrame(rows)
    sig_df = compute_paired_significance(df)
    assert not sig_df.empty

    for _, row in sig_df.iterrows():
        assert row["mean_diff"] == 0.0
        assert row["std_diff"] == 0.0
        assert row["t_statistic"] == 0.0
        assert row["p_value_ttest"] == 1.0
        assert row["wilcoxon_stat"] == 0.0
        assert row["p_value_wilcoxon"] == 1.0

    sig_dict = compute_statistical_tests(df)
    validate_statistical_significance_schema(sig_dict)


def test_compute_statistical_tests_non_zero_differences():
    """Verifies that realistic differences produce valid p-values and statistics."""
    rng = np.random.default_rng(12345)
    rows = []
    for s in [42, 43, 44, 45, 46]:
        for w in range(5):
            for y in [2022, 2023, 2024]:
                base_overload = 126.0 + rng.normal(0, 5)
                # Predictive reduces overload by ~12 minutes
                pred_overload = base_overload - 12.0 + rng.normal(0, 3)
                # Persistence has slightly higher overload
                pers_overload = base_overload - 10.0 + rng.normal(0, 3)

                for pol, ov, fwd in [
                    ("predictive_carbon", pred_overload, 333.0),
                    ("reactive_carbon", base_overload, 323.0),
                    ("persistence_carbon", pers_overload, 338.0),
                ]:
                    rows.append({
                        "model_seed": s,
                        "window": w,
                        "carbon_year": y,
                        "scenario": "congested",
                        "observation_delay_bins": 1,
                        "policy": pol,
                        "overload_minutes": ov,
                        "forwarding_score": fwd,
                        "served_fraction": 0.92,
                        "route_changes": 18.0,
                    })

    df = pd.DataFrame(rows)
    sig_dict = compute_statistical_tests(df)
    validate_statistical_significance_schema(sig_dict)

    vs_reactive = sig_dict["vs_reactive"]
    assert vs_reactive["overload_minutes"]["mean_diff"] < 0
    assert 0.0 <= vs_reactive["overload_minutes"]["p_value_ttest"] <= 0.01
    assert 0.0 <= vs_reactive["overload_minutes"]["p_value_wilcoxon"] <= 0.10


def test_trained_predictor_accepts_output_dir_and_seed():
    """Verifies that TrainedPredictor initializes correctly with seed and output_dir."""
    from carbonroute.data import load_prepared
    from carbonroute.prediction import TrainedPredictor

    cfg = config()
    prep_cfg = dict(cfg, seed=42)
    data = load_prepared(ROOT, prep_cfg)

    seed_dir = ROOT / "artifacts/seeds/seed_42"
    if not seed_dir.exists():
        pytest.skip("Seed 42 artifact directory not yet trained")

    # Initialize from seed_dir with explicit seed
    pred = TrainedPredictor(seed_dir, data, cfg, kind="gnn", seed=42)
    assert pred.model is not None
    assert pred.margin.shape == (cfg["horizon"], 8, 8)

    # Initialize using output_dir parameter
    pred_out = TrainedPredictor(output_dir=seed_dir, prepared=data, cfg=cfg, kind="gnn", seed=42)
    assert pred_out.model is not None
    assert pred_out.margin.shape == (cfg["horizon"], 8, 8)
