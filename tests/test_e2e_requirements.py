"""E2E Requirements Verification Suite for CarbonRoute.

Implements the 4-tier verification methodology defined in TEST_INFRA.md:
- Tier 1: Feature Coverage (Multi-seed schema, significance format, sensitivity grid, manuscript sections, Windows CPU environment)
- Tier 2: Boundary & Corner Cases (Zero overload minutes, p-value bounds [0, 1], quantile monotonicity, tau hysteresis monotonicity, non-negative metrics)
- Tier 3: Cross-Feature Interactions (Paired significance testing, Pareto frontier non-dominated sorting, manuscript-to-artifact reconciliation)
- Tier 4: Real-World Acceptance Scenarios (8-stage independent auditor, full unit test suite 100% pass, summary-manifest reconciliation)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import ttest_rel, wilcoxon

from carbonroute.common import ROOT, load_json, sha256
from carbonroute.data import load_prepared
from carbonroute.prediction import calibrate_margin
from carbonroute.routing import Planner

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.verify_results import verify as auditor_verify

# ==============================================================================
# Helper & Schema Validator Functions (Authoritative Interface Contracts)
# ==============================================================================

def validate_multiseed_summary_schema(df: pd.DataFrame, min_seeds: int = 5) -> None:
    """Validates the schema for multi-seed campaign summary data."""
    required_policies = {"predictive_carbon", "reactive_carbon"}
    policies_present = set(df["policy"].unique()) if "policy" in df.columns else set()
    missing_policies = required_policies - policies_present
    assert not missing_policies, f"Summary missing required policies: {missing_policies}"

    if "seed" in df.columns:
        seeds = df["seed"].unique()
        assert len(seeds) >= min_seeds, (
            f"Expected at least {min_seeds} distinct random seeds, found {len(seeds)}: {seeds}"
        )

    # Check that key metrics exist either as raw columns or aggregated mean/std
    core_metrics = ["forwarding_score", "overload_minutes", "served_fraction"]
    for metric in core_metrics:
        has_raw = metric in df.columns
        has_agg = f"{metric}_mean" in df.columns and f"{metric}_std" in df.columns
        assert has_raw or has_agg, f"Metric '{metric}' missing raw or aggregated (mean/std) columns"

        # If aggregated std is present, verify non-negativity
        if f"{metric}_std" in df.columns:
            assert (df[f"{metric}_std"].dropna() >= 0).all(), f"Negative std found in {metric}_std"


def validate_statistical_significance_schema(sig: dict[str, Any]) -> None:
    """Validates the schema and bounds of statistical significance JSON artifacts."""
    assert isinstance(sig, dict), "Statistical significance artifact must be a dictionary"
    # Must contain comparisons against reactive_carbon and persistence_carbon
    comp_keys = [str(k).lower() for k in sig.keys()]
    has_reactive = any("reactive" in k for k in comp_keys)
    has_persistence = any("persistence" in k for k in comp_keys)
    assert has_reactive, "Significance data must include comparison against reactive_carbon"
    assert has_persistence, "Significance data must include comparison against persistence_carbon"

    for comp_name, comp_data in sig.items():
        assert isinstance(comp_data, dict), f"Comparison '{comp_name}' payload must be a dict"
        for metric_name, mstats in comp_data.items():
            assert isinstance(mstats, dict), f"Metric '{metric_name}' in '{comp_name}' must be a dict"
            required_keys = {"p_value_ttest", "p_value_wilcoxon", "t_statistic", "wilcoxon_stat", "mean_diff"}
            missing = required_keys - set(mstats.keys())
            assert not missing, f"Missing required keys {missing} in {comp_name}.{metric_name}"

            p_t = mstats["p_value_ttest"]
            p_w = mstats["p_value_wilcoxon"]
            assert p_t is not None and 0.0 <= p_t <= 1.0, f"Invalid t-test p-value {p_t} in {comp_name}.{metric_name}"
            assert p_w is not None and 0.0 <= p_w <= 1.0, f"Invalid Wilcoxon p-value {p_w} in {comp_name}.{metric_name}"
            assert np.isfinite(mstats["mean_diff"]), f"Non-finite mean difference in {comp_name}.{metric_name}"


def validate_sensitivity_summary_schema(df: pd.DataFrame) -> None:
    """Validates the schema for parameter sensitivity sweep outputs."""
    # Column aliases support: alpha/margin_quantile, tau/min_improvement
    has_alpha = "alpha" in df.columns or "margin_quantile" in df.columns
    has_tau = "tau" in df.columns or "min_improvement" in df.columns
    assert has_alpha, "Sensitivity summary must contain 'alpha' or 'margin_quantile'"
    assert has_tau, "Sensitivity summary must contain 'tau' or 'min_improvement'"

    alpha_col = "alpha" if "alpha" in df.columns else "margin_quantile"
    tau_col = "tau" if "tau" in df.columns else "min_improvement"

    assert (df[alpha_col] >= 0.70).all() and (df[alpha_col] <= 0.99).all(), (
        f"Alpha values out of expected bounds [0.70, 0.99]: {df[alpha_col].unique()}"
    )
    assert (df[tau_col] >= 0.0).all(), f"Tau values must be non-negative: {df[tau_col].unique()}"

    for col in ["forwarding_score", "overload_minutes", "route_changes"]:
        assert col in df.columns, f"Missing required metric column '{col}' in sensitivity summary"
        assert (df[col] >= 0.0).all(), f"Negative values found in metric column '{col}'"


def validate_threats_to_validity(content: str) -> None:
    """Validates presence and categories of Threats to Validity in manuscript."""
    assert re.search(r"#+\s*(?:\d+\.?\s*)?Threats to Validity", content, re.IGNORECASE), (
        "Manuscript missing 'Threats to Validity' section header"
    )
    categories = ["Internal", "External", "Construct", "Conclusion"]
    found = [cat for cat in categories if re.search(rf"\b{cat}\b", content, re.IGNORECASE)]
    assert len(found) >= 3, (
        f"Expected at least 3 standard validity dimensions (Internal, External, Construct, Conclusion), found: {found}"
    )


def validate_simulation_fidelity(content: str) -> None:
    """Validates presence and technical discussion of Simulation Fidelity in manuscript."""
    assert re.search(r"#+\s*(?:\d+\.?\s*)?Simulation Fidelity", content, re.IGNORECASE), (
        "Manuscript missing 'Simulation Fidelity' section header"
    )
    # Check for technical discussion contrasting fluid model with queuing/packet dynamics
    has_fluid = bool(re.search(r"fluid|progressive[\s-]filling", content, re.IGNORECASE))
    has_queuing = bool(re.search(r"packet|queu(?:e|ing)|tcp", content, re.IGNORECASE))
    assert has_fluid, "Simulation Fidelity section must discuss fluid / progressive filling model"
    assert has_queuing, "Simulation Fidelity section must contrast against packet queuing / TCP dynamics"


def compute_paired_hypothesis_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Computes paired t-test and Wilcoxon signed-rank test with boundary handling."""
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0

    if np.allclose(diff, 0.0):
        return {
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "t_statistic": 0.0,
            "p_value_ttest": 1.0,
            "wilcoxon_stat": 0.0,
            "p_value_wilcoxon": 1.0,
        }

    t_res = ttest_rel(a, b)
    t_stat = float(t_res.statistic) if np.isfinite(t_res.statistic) else 0.0
    p_t = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else 1.0

    try:
        w_res = wilcoxon(diff, alternative="two-sided")
        w_stat = float(w_res.statistic)
        p_w = float(w_res.pvalue)
    except Exception:
        w_stat = 0.0
        p_w = 1.0

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": t_stat,
        "p_value_ttest": max(0.0, min(1.0, p_t)),
        "wilcoxon_stat": w_stat,
        "p_value_wilcoxon": max(0.0, min(1.0, p_w)),
    }


def is_pareto_efficient_tri_objective(costs: np.ndarray) -> np.ndarray:
    """Finds non-dominated points where all 3 objectives are minimized.
    
    costs shape: (N, 3) where columns correspond to:
    (forwarding_score, route_changes, overload_minutes)
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # A point is dominated if another point is <= on all dimensions and < on at least one
            dominated = np.all(costs <= c, axis=1) & np.any(costs < c, axis=1)
            if np.any(dominated):
                is_efficient[i] = False
    return is_efficient


def parse_manuscript_table(content: str) -> dict[str, dict[str, float]]:
    """Parses Table from PAPER_DRAFT.md Section 6."""
    pattern = (
        r"\|\s*Policy\s*\|\s*Carbon forwarding score\s*\|\s*Served demand\s*\|\s*"
        r"Overload min/day\s*\|\s*Changes/day\s*\|\s*\|[-:| ]+\|\s*((?:\|.*\|\s*)+)"
    )
    match = re.search(pattern, content)
    assert match, "Failed to match Table in PAPER_DRAFT.md"
    rows = match.group(1).strip().split("\n")
    parsed: dict[str, dict[str, float]] = {}

    def map_policy(lbl: str) -> str:
        s = lbl.lower()
        if "shortest" in s:
            return "shortest_path"
        if "reactive" in s:
            return "reactive_carbon"
        if "engineering" in s or "predictive traffic" in s:
            return "predictive_te"
        if "persistence" in s:
            return "persistence_carbon"
        if "without margin" in s or "no_margin" in s:
            return "no_margin"
        if "without hysteresis" in s or "no_hysteresis" in s:
            return "no_hysteresis"
        if "proposed" in s or "gcn" in s:
            return "predictive_carbon"
        return s

    for row in rows:
        parts = [p.strip() for p in row.split("|")[1:-1]]
        if len(parts) >= 5:
            policy_id = map_policy(parts[0])
            score = float(parts[1])
            served = float(parts[2].replace("%", ""))
            overload = float(parts[3])
            changes = float(parts[4])
            parsed[policy_id] = {
                "forwarding_score": score,
                "served_demand": served,
                "overload_minutes": overload,
                "changes_day": changes,
            }
    return parsed


# ==============================================================================
# Tier 1: Feature Coverage Tests
# ==============================================================================

def test_tier1_multiseed_output_schema():
    """Tier 1: Verifies multi-seed schema contract, seed count >= 5, and mean/std columns."""
    # 1. Self-verify validator logic with synthetic valid and invalid frames
    synth_valid = pd.DataFrame({
        "seed": [42, 43, 44, 45, 46] * 2,
        "policy": ["predictive_carbon"] * 5 + ["reactive_carbon"] * 5,
        "forwarding_score": [330.0, 332.0, 335.0, 331.0, 333.0, 320.0, 322.0, 325.0, 321.0, 324.0],
        "overload_minutes": [110.0, 112.0, 115.0, 113.0, 114.0, 125.0, 127.0, 130.0, 124.0, 126.0],
        "served_fraction": [0.92] * 10,
        "route_changes": [18.0] * 5 + [19.0] * 5,
    })
    validate_multiseed_summary_schema(synth_valid, min_seeds=5)

    with pytest.raises(AssertionError, match="at least 5 distinct random seeds"):
        validate_multiseed_summary_schema(synth_valid[synth_valid["seed"] < 45], min_seeds=5)

    with pytest.raises(AssertionError, match="Summary missing required policies"):
        validate_multiseed_summary_schema(synth_valid[synth_valid["policy"] != "reactive_carbon"], min_seeds=5)

    # 2. Inspect physical file artifacts with progressive testability
    multiseed_path = ROOT / "results/multiseed/summary_multiseed.csv"
    alt_multiseed = ROOT / "results/multi_seed_summary.csv"
    seeds_dir = ROOT / "results/seeds"

    if not (multiseed_path.exists() or alt_multiseed.exists() or seeds_dir.exists()):
        pytest.skip("Pending Milestone 2: multi-seed campaign outputs not yet generated")

    target = multiseed_path if multiseed_path.exists() else alt_multiseed
    if target.exists():
        df = pd.read_csv(target)
        validate_multiseed_summary_schema(df, min_seeds=5)


def test_tier1_statistical_significance_json_format():
    """Tier 1: Verifies paired t-test and Wilcoxon p-value JSON format and bounds."""
    # 1. Self-verify validator logic with contract fixtures
    synth_sig = {
        "vs_reactive": {
            "overload_minutes": {
                "mean_diff": -12.0,
                "std_diff": 11.4,
                "t_statistic": -3.85,
                "p_value_ttest": 0.0013,
                "wilcoxon_stat": 3.0,
                "p_value_wilcoxon": 0.0057,
            },
            "forwarding_score": {
                "mean_diff": 9.35,
                "std_diff": 5.2,
                "t_statistic": 4.12,
                "p_value_ttest": 0.0005,
                "wilcoxon_stat": 2.0,
                "p_value_wilcoxon": 0.0008,
            },
            "served_fraction": {
                "mean_diff": 0.0003,
                "std_diff": 0.0002,
                "t_statistic": 3.20,
                "p_value_ttest": 0.0039,
                "wilcoxon_stat": 4.0,
                "p_value_wilcoxon": 0.0071,
            },
        },
        "vs_persistence": {
            "overload_minutes": {
                "mean_diff": 0.0,
                "std_diff": 1.2,
                "t_statistic": 0.0,
                "p_value_ttest": 1.0,
                "wilcoxon_stat": 10.0,
                "p_value_wilcoxon": 1.0,
            },
            "forwarding_score": {
                "mean_diff": -5.4,
                "std_diff": 3.1,
                "t_statistic": -2.8,
                "p_value_ttest": 0.012,
                "wilcoxon_stat": 5.0,
                "p_value_wilcoxon": 0.015,
            },
            "served_fraction": {
                "mean_diff": -0.0006,
                "std_diff": 0.0004,
                "t_statistic": -2.1,
                "p_value_ttest": 0.045,
                "wilcoxon_stat": 8.0,
                "p_value_wilcoxon": 0.048,
            },
        },
    }
    validate_statistical_significance_schema(synth_sig)

    # Invalid cases
    with pytest.raises(AssertionError, match="reactive_carbon"):
        validate_statistical_significance_schema({"vs_other": {}})

    # 2. Inspect physical disk artifact
    sig_path = ROOT / "results/multiseed/statistical_significance.json"
    alt_sig = ROOT / "results/statistical_significance.json"
    if not (sig_path.exists() or alt_sig.exists()):
        pytest.skip("Pending Milestone 2: statistical significance JSON not yet generated")

    target = sig_path if sig_path.exists() else alt_sig
    data = load_json(target)
    validate_statistical_significance_schema(data)


def test_tier1_sensitivity_sweep_data_format():
    """Tier 1: Verifies alpha and tau sensitivity sweep data schema and ranges."""
    # 1. Self-verify validator logic with contract fixtures
    synth_sens = pd.DataFrame({
        "alpha": [0.70, 0.80, 0.90, 0.99],
        "tau": [0.00, 0.05, 0.10, 0.20],
        "observation_delay_bins": [1, 1, 1, 1],
        "forwarding_score": [320.0, 325.0, 333.0, 340.0],
        "overload_minutes": [135.0, 125.0, 114.0, 100.0],
        "route_changes": [25.0, 18.0, 12.0, 8.0],
    })
    validate_sensitivity_summary_schema(synth_sens)

    with pytest.raises(AssertionError, match="bounds"):
        bad_sens = synth_sens.copy()
        bad_sens.loc[0, "alpha"] = 0.50
        validate_sensitivity_summary_schema(bad_sens)

    # 2. Inspect physical file artifact
    sens_path = ROOT / "results/sensitivity/summary.csv"
    if not sens_path.exists():
        pytest.skip("Pending Milestone 3: sensitivity sweep data not yet generated")

    df = pd.read_csv(sens_path)
    validate_sensitivity_summary_schema(df)


def test_tier1_manuscript_threats_to_validity_presence():
    """Tier 1: Audits Threats to Validity presence in docs/PAPER_DRAFT.md."""
    # 1. Self-verify validator logic
    sample_text = """
    ## 8. Threats to Validity
    ### 8.1 Internal Validity
    Discussing causal observation lag...
    ### 8.2 External Validity
    Discussing WAN topology and traffic scaling...
    ### 8.3 Construct Validity
    Discussing carbon intensity attribution metrics...
    ### 8.4 Conclusion Validity
    Discussing multi-seed sample size and hypothesis testing...
    """
    validate_threats_to_validity(sample_text)

    # 2. Inspect manuscript with progressive testability
    paper_path = ROOT / "docs/PAPER_DRAFT.md"
    assert paper_path.exists(), "docs/PAPER_DRAFT.md must exist"
    content = paper_path.read_text(encoding="utf-8")

    if not re.search(r"Threats to Validity", content, re.IGNORECASE):
        pytest.skip("Pending Milestone 4: 'Threats to Validity' section scheduled for manuscript elevation")

    validate_threats_to_validity(content)


def test_tier1_manuscript_simulation_fidelity_presence():
    """Tier 1: Audits Simulation Fidelity discussion presence in docs/PAPER_DRAFT.md."""
    # 1. Self-verify validator logic
    sample_text = """
    ## 7. Simulation Fidelity vs. Real Queuing Dynamics
    The fluid progressive-filling allocator abstracts discrete packet queuing and TCP rate control.
    """
    validate_simulation_fidelity(sample_text)

    # 2. Inspect manuscript with progressive testability
    paper_path = ROOT / "docs/PAPER_DRAFT.md"
    assert paper_path.exists(), "docs/PAPER_DRAFT.md must exist"
    content = paper_path.read_text(encoding="utf-8")

    if not re.search(r"Simulation Fidelity", content, re.IGNORECASE):
        pytest.skip("Pending Milestone 4: 'Simulation Fidelity' section scheduled for manuscript elevation")

    validate_simulation_fidelity(content)


def test_tier1_windows_cpu_command_compliance():
    """Tier 1: Enforces Windows CPU environment execution compliance and configuration."""
    # 1. PyTorch CPU environment checks
    assert not torch.cuda.is_available(), "CUDA unexpectedly detected in CPU-constrained environment"
    assert not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    # Verify tensor operations run on CPU cleanly
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assert t.device.type == "cpu"

    # 2. pyproject.toml pythonpath check
    pyproject_path = ROOT / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml must exist"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    assert re.search(r'pythonpath\s*=\s*\[[^\]]*"src"[^\]]*\]', pyproject_text) is not None, (
        "pyproject.toml must configure pythonpath with 'src'"
    )

    # 3. Config cpu_threads check
    cfg = load_json(ROOT / "configs/default.json")
    assert "cpu_threads" in cfg
    assert isinstance(cfg["cpu_threads"], int) and cfg["cpu_threads"] > 0

    # 4. Core modules importable
    import carbonroute.prediction
    import carbonroute.routing
    import carbonroute.simulation
    assert hasattr(carbonroute.prediction, "calibrate_margin")
    assert hasattr(carbonroute.routing, "Planner")
    assert hasattr(carbonroute.simulation, "replay")


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================

def test_tier2_zero_overload_minutes_handling():
    """Tier 2: Boundary test ensuring zero overload minutes produces finite, valid metrics."""
    # In traffic engineering, zero overload minutes is a normal boundary condition.
    # Mathematical operations such as percentage differences or ratios must not divide by zero.
    df = pd.read_csv(ROOT / "results/paper/summary.csv")
    zero_overload_runs = df[df["overload_minutes"] == 0.0]
    assert len(zero_overload_runs) > 0, "Expected runs with zero overload minutes in baseline summary"

    # Excess Mbit must strictly be 0.0 when overload minutes is 0.0
    assert (zero_overload_runs["excess_mbit"] == 0.0).all(), (
        "Non-zero excess Mbit detected when overload_minutes == 0.0"
    )

    # Test paired difference calculation on zero vectors
    zero_a = np.zeros(10)
    zero_b = np.zeros(10)
    res = compute_paired_hypothesis_test(zero_a, zero_b)
    assert res["mean_diff"] == 0.0
    assert res["p_value_ttest"] == 1.0
    assert res["p_value_wilcoxon"] == 1.0
    assert np.isfinite(res["t_statistic"])


def test_tier2_p_value_bounds_strict():
    """Tier 2: Invariant test verifying p-values are strictly bounded in [0.0, 1.0]."""
    rng = np.random.default_rng(12345)
    for _ in range(20):
        a = rng.normal(loc=100.0, scale=10.0, size=15)
        b = a + rng.normal(loc=0.5, scale=2.0, size=15)
        res = compute_paired_hypothesis_test(a, b)
        assert 0.0 <= res["p_value_ttest"] <= 1.0, f"Out-of-bounds t-test p-value: {res['p_value_ttest']}"
        assert 0.0 <= res["p_value_wilcoxon"] <= 1.0, f"Out-of-bounds Wilcoxon p-value: {res['p_value_wilcoxon']}"

    # Extreme separation edge case: p-value should approach 0.0
    a_high = np.full(20, 1000.0)
    b_low = np.full(20, 10.0) + rng.normal(0, 0.1, 20)
    res_ext = compute_paired_hypothesis_test(a_high, b_low)
    assert 0.0 <= res_ext["p_value_ttest"] < 1e-10
    assert 0.0 <= res_ext["p_value_wilcoxon"] < 1e-4


def test_tier2_alpha_quantiles_monotonicity():
    """Tier 2: Mathematical invariant: calibrate_margin is monotonic non-decreasing in alpha."""
    rng = np.random.default_rng(42)
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 2] = mask[3, 0] = mask[4, 1] = mask[5, 3] = True

    # Test across multiple residual distributions: exponential, uniform, heavy-tailed
    distributions = [
        ("exponential", rng.exponential(scale=2.0, size=(120, 3, 8, 8)), rng.exponential(scale=1.5, size=(120, 3, 8, 8))),
        ("uniform", rng.uniform(0.0, 5.0, size=(120, 3, 8, 8)), rng.uniform(0.0, 4.0, size=(120, 3, 8, 8))),
        ("heavy_tailed", rng.pareto(a=2.0, size=(120, 3, 8, 8)), rng.uniform(0.0, 2.0, size=(120, 3, 8, 8))),
    ]

    alphas = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

    for name, actual, pred in distributions:
        margins = [calibrate_margin(actual, pred, mask, a) for a in alphas]
        for i in range(len(alphas) - 1):
            diff = margins[i + 1] - margins[i]
            # Must be non-negative everywhere within numerical precision
            assert np.all(diff >= -1e-9), (
                f"Margin quantile monotonicity violated for {name} between alpha={alphas[i]} and {alphas[i+1]}"
            )


def test_tier2_tau_hysteresis_route_change_monotonicity():
    """Tier 2: Behavioral invariant: increasing tau monotonically decreases or preserves route changes."""
    cfg = load_json(ROOT / "configs/default.json")
    data = load_prepared(ROOT, cfg)
    ci = np.array([200.0, 400.0, 150.0, 300.0, 100.0, 250.0, 350.0, 120.0])
    forecast = np.zeros((3, 8, 8))
    for a, b in data.pairs:
        forecast[:, a, b] = 2.0

    tau_values = [0.00, 0.02, 0.05, 0.10, 0.20, 0.50]
    changes_observed = []

    for tau in tau_values:
        step_cfg = dict(cfg)
        step_cfg["min_improvement"] = tau
        planner = Planner(data.network, data.pairs, step_cfg, "predictive_carbon")
        decision = planner.decide(forecast, ci, step=0)
        changes_observed.append(decision.changes)

    # Monotonicity check: changes(tau_i) >= changes(tau_{i+1})
    for i in range(len(tau_values) - 1):
        assert changes_observed[i] >= changes_observed[i + 1], (
            f"Hysteresis route-change monotonicity violated: tau={tau_values[i]} had "
            f"{changes_observed[i]} changes, but tau={tau_values[i+1]} had {changes_observed[i+1]} changes"
        )


def test_tier2_non_negative_metrics_and_physical_bounds():
    """Tier 2: Invariant test verifying physical bounds and non-negativity across all metrics."""
    df = pd.read_csv(ROOT / "results/paper/summary.csv")

    assert (df["forwarding_score"] >= 0.0).all(), "Negative forwarding score detected"
    assert (df["overload_minutes"] >= 0.0).all(), "Negative overload minutes detected"
    assert (df["excess_mbit"] >= 0.0).all(), "Negative excess Mbit detected"
    assert (df["served_fraction"] >= 0.0).all(), "Negative served fraction detected"
    assert (df["served_fraction"] <= 1.00001).all(), "Served fraction exceeded 100%"
    assert (df["max_utilization"] >= 0.0).all(), "Negative link utilization detected"
    assert (df["route_changes"] >= 0).all(), "Negative route changes detected"
    assert (df["infeasible_pair_bins"] >= 0).all(), "Negative infeasible pair bins detected"


# ==============================================================================
# Tier 3: Cross-Feature Interactions
# ==============================================================================

def test_tier3_paired_statistical_significance_computation():
    """Tier 3: Validates paired hypothesis testing between GNN+Safeguards vs Reactive and Persistence."""
    df = pd.read_csv(ROOT / "results/paper/summary.csv")
    # Primary congested delayed scenario
    cg = df[(df["scenario"] == "congested") & (df["observation_delay_bins"] == 1)]

    p_carb = cg[cg["policy"] == "predictive_carbon"].sort_values(["window", "carbon_year"])
    r_carb = cg[cg["policy"] == "reactive_carbon"].sort_values(["window", "carbon_year"])
    persist = cg[cg["policy"] == "persistence_carbon"].sort_values(["window", "carbon_year"])

    assert len(p_carb) == len(r_carb) == len(persist) == 15, (
        f"Expected 15 paired runs (5 windows * 3 carbon years), found: "
        f"{len(p_carb)}, {len(r_carb)}, {len(persist)}"
    )

    # 1. GNN+Safeguards vs Reactive Carbon
    # Overload minutes: GNN reduces overload significantly
    ovld_p = p_carb["overload_minutes"].to_numpy()
    ovld_r = r_carb["overload_minutes"].to_numpy()
    res_ovld_react = compute_paired_hypothesis_test(ovld_p, ovld_r)

    # Mean difference must match manuscript: -12.00 minutes/day
    np.testing.assert_allclose(res_ovld_react["mean_diff"], -12.0, atol=0.01)
    assert res_ovld_react["p_value_ttest"] < 0.01, (
        f"Expected p < 0.01 for overload reduction, got {res_ovld_react['p_value_ttest']}"
    )
    assert res_ovld_react["p_value_wilcoxon"] < 0.01

    # Forwarding score: GNN increases carbon score slightly (+9.35)
    carb_p = p_carb["forwarding_score"].to_numpy()
    carb_r = r_carb["forwarding_score"].to_numpy()
    res_carb_react = compute_paired_hypothesis_test(carb_p, carb_r)
    np.testing.assert_allclose(res_carb_react["mean_diff"], 9.35, atol=0.02)
    assert res_carb_react["p_value_ttest"] < 0.01

    # 2. GNN+Safeguards vs Persistence+Safeguards
    # Overload minutes are identical on average (114 vs 114)
    ovld_pers = persist["overload_minutes"].to_numpy()
    res_ovld_pers = compute_paired_hypothesis_test(ovld_p, ovld_pers)
    np.testing.assert_allclose(res_ovld_pers["mean_diff"], 0.0, atol=0.01)


def test_tier3_sensitivity_pareto_frontier_non_dominated_sorting():
    """Tier 3: Validates tri-objective Pareto non-dominated sorting."""
    # Objective dimensions: (carbon forwarding score, route changes, overload minutes)
    test_points = np.array([
        [10.0, 10.0, 10.0],  # Point A: Pareto optimal
        [15.0, 12.0, 10.0],  # Point B: Dominated by A (15 > 10, 12 > 10, 10 == 10)
        [9.0, 15.0, 12.0],   # Point C: Non-dominated with A (better carbon, worse changes/overload)
        [12.0, 8.0, 15.0],   # Point D: Non-dominated with A, C (better changes)
        [20.0, 20.0, 20.0],  # Point E: Dominated by all
    ])
    mask = is_pareto_efficient_tri_objective(test_points)
    expected_mask = np.array([True, False, True, True, False])
    assert np.array_equal(mask, expected_mask), (
        f"Non-dominated sorting failed: got {mask}, expected {expected_mask}"
    )

    # If physical Pareto frontier file exists, verify non-dominance
    frontier_path = ROOT / "results/sensitivity/pareto_frontier.json"
    if frontier_path.exists():
        pareto_data = load_json(frontier_path)
        costs = np.array([
            [p["forwarding_score"], p["route_changes"], p["overload_minutes"]]
            for p in pareto_data
        ])
        assert is_pareto_efficient_tri_objective(costs).all(), (
            "Pareto frontier artifact contains mutually dominated points"
        )


def test_tier3_manuscript_numerical_claims_match_artifacts():
    """Tier 3: Audits 100% of manuscript table claims against summary run artifacts within 0.01 tolerance."""
    paper_path = ROOT / "docs/PAPER_DRAFT.md"
    assert paper_path.exists(), "docs/PAPER_DRAFT.md must exist"
    content = paper_path.read_text(encoding="utf-8")

    manuscript_table = parse_manuscript_table(content)
    assert len(manuscript_table) == 7, f"Expected 7 policies in manuscript table, found {len(manuscript_table)}"

    # Authoritative ground truth derived from results/paper/summary.csv (congested lag 1)
    df = pd.read_csv(ROOT / "results/paper/summary.csv")
    cg = df[(df["scenario"] == "congested") & (df["observation_delay_bins"] == 1)]
    grouped = cg.groupby("policy")[["forwarding_score", "served_fraction", "overload_minutes", "route_changes"]].mean()

    for policy_id, paper_vals in manuscript_table.items():
        assert policy_id in grouped.index, f"Policy '{policy_id}' from manuscript not found in simulation summary"
        data_row = grouped.loc[policy_id]

        # 1. Forwarding score within +- 0.01
        score_diff = abs(paper_vals["forwarding_score"] - data_row["forwarding_score"])
        assert score_diff <= 0.01, (
            f"Policy {policy_id} forwarding score mismatch: paper={paper_vals['forwarding_score']}, "
            f"data={data_row['forwarding_score']:.2f} (diff={score_diff:.4f})"
        )

        # 2. Served demand % within +- 0.01%
        served_diff = abs(paper_vals["served_demand"] - data_row["served_fraction"] * 100)
        assert served_diff <= 0.01, (
            f"Policy {policy_id} served demand mismatch: paper={paper_vals['served_demand']}%, "
            f"data={data_row['served_fraction']*100:.2f}% (diff={served_diff:.4f})"
        )

        # 3. Overload minutes within +- 0.01
        ovld_diff = abs(paper_vals["overload_minutes"] - data_row["overload_minutes"])
        assert ovld_diff <= 0.01, (
            f"Policy {policy_id} overload minutes mismatch: paper={paper_vals['overload_minutes']}, "
            f"data={data_row['overload_minutes']:.2f} (diff={ovld_diff:.4f})"
        )

        # 4. Route changes within +- 0.01
        chg_diff = abs(paper_vals["changes_day"] - data_row["route_changes"])
        assert chg_diff <= 0.01, (
            f"Policy {policy_id} route changes mismatch: paper={paper_vals['changes_day']}, "
            f"data={data_row['route_changes']:.2f} (diff={chg_diff:.4f})"
        )


# ==============================================================================
# Tier 4: Real-World Acceptance Scenarios
# ==============================================================================

def test_tier4_verify_results_auditor_all_directories():
    """Tier 4: Validates that verify_results.py succeeds without error on all generated output dirs."""
    # 1. Canonical paper campaign directory
    paper_dir = ROOT / "results/paper"
    assert paper_dir.exists(), "results/paper directory must exist"
    auditor_verify(paper_dir)

    # 2. Demo campaign directory
    demo_dir = ROOT / "results/demo"
    if demo_dir.exists():
        auditor_verify(demo_dir)

    # 3. Dynamically audit all seed campaign directories if present
    seeds_parent = ROOT / "results/seeds"
    if seeds_parent.exists():
        for sdir in sorted(seeds_parent.iterdir()):
            if sdir.is_dir() and (sdir / "manifest.json").exists():
                auditor_verify(sdir)

    # 4. Dynamically audit sensitivity campaign directory if present and complete
    sens_dir = ROOT / "results/sensitivity"
    if sens_dir.exists() and (sens_dir / "manifest.json").exists():
        manifest = load_json(sens_dir / "manifest.json")
        if manifest.get("status") == "complete":
            auditor_verify(sens_dir)


def test_tier4_baseline_unit_test_suite_passes():
    """Tier 4: Validates that existing unit test suite passes 100% with no regressions."""
    baseline_modules = [
        "tests/test_causality.py",
        "tests/test_data.py",
        "tests/test_prediction.py",
        "tests/test_routing.py",
        "tests/test_service.py",
    ]
    for mod in baseline_modules:
        assert (ROOT / mod).exists(), f"Missing unit test module: {mod}"

    cmd = [sys.executable, "-m", "pytest"] + baseline_modules + ["-q"]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Baseline unit tests failed:\n{res.stdout}\n{res.stderr}"


def test_tier4_summary_reports_reconcile():
    """Tier 4: Validates reconciliation between manifest.json and summary.csv."""
    dirs_to_check = [ROOT / "results/paper"]
    seeds_parent = ROOT / "results/seeds"
    if seeds_parent.exists():
        for sdir in seeds_parent.iterdir():
            if sdir.is_dir() and (sdir / "manifest.json").exists():
                dirs_to_check.append(sdir)

    for campaign_dir in dirs_to_check:
        manifest_path = campaign_dir / "manifest.json"
        summary_path = campaign_dir / "summary.csv"
        assert manifest_path.exists() and summary_path.exists()

        manifest = load_json(manifest_path)
        summary = pd.read_csv(summary_path)

        # Check cryptographic hash
        assert sha256(summary_path) == manifest["summary_sha256"], (
            f"Summary SHA-256 mismatch in {campaign_dir}"
        )
        # Check run count parity
        assert len(manifest["runs"]) == len(summary), (
            f"Run count mismatch: manifest has {len(manifest['runs'])}, summary has {len(summary)}"
        )
        # Check run ID parity
        assert set(summary["run_id"]) == {r["run_id"] for r in manifest["runs"]}
        # Check no duplicate run IDs
        assert not summary["run_id"].duplicated().any()
