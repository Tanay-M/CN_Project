"""Parameter Sensitivity Sweep, Tri-Objective Pareto Frontier, and Publication Figures.

Implements Milestone 3 (F5-F8) for CarbonRoute:
- F5: Safety margin quantile sweep alpha in [0.70, 0.99] without GNN retraining.
- F6: Route flap hysteresis threshold sweep tau in [0.00, 0.20].
- F7: Empirical Pareto frontier characterization across (Forwarding Score, Route Changes, Overload Minutes).
- F8: Publication-quality figures (alpha curves, tau curves, 2D Pareto convex hull, 3D surface).
"""

from __future__ import annotations

import copy
import hashlib
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .common import ROOT, config, fingerprint, load_json, save_json, sha256
from .data import CarbonTable, Prepared, load_prepared
from .experiments import evaluation_windows, workload
from .prediction import TrainedPredictor, batch_prediction, calibrate_margin, make_windows
from .simulation import replay

DEFAULT_ALPHAS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
DEFAULT_TAUS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
DEFAULT_DELAYS = [1, 0]
DEFAULT_SCENARIOS = ["congested", "normal"]


def get_cached_validation_residuals(
    data: Prepared,
    predictor: TrainedPredictor,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts validation targets and computes base predictions without retraining.
    
    Returns:
        yval: Actual demand targets on validation split, shape (N_val, horizon, 8, 8).
        val_pred: Model predicted demand on validation split, shape (N_val, horizon, 8, 8).
    """
    xval, yval, _ = make_windows(data, cfg, "validation")
    val_pred = batch_prediction(predictor.model, xval, data.scale, cfg["batch_size"])
    return yval, val_pred


def is_pareto_efficient_tri_objective(costs: np.ndarray) -> np.ndarray:
    """Finds non-dominated points where all 3 objectives are minimized.

    costs shape: (N, 3) where columns correspond to:
    (forwarding_score, route_changes, overload_minutes)
    
    A point c is dominated if another point is <= c on all 3 dimensions
    and < c on at least one dimension.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            dominated = np.all(costs <= c, axis=1) & np.any(costs < c, axis=1)
            if np.any(dominated):
                is_efficient[i] = False
    return is_efficient


def extract_pareto_frontier(
    df: pd.DataFrame,
    scenario: str | None = "congested",
    delay: int | None = 1,
) -> list[dict[str, Any]]:
    """Extracts non-dominated configurations from the sweep summary DataFrame.
    
    Filters by scenario and delay if specified, extracts the non-dominated subset,
    and returns a structured list of dictionaries for pareto_frontier.json.
    """
    subset = df.copy()
    if scenario is not None and "scenario" in subset.columns:
        subset = subset[subset["scenario"] == scenario]
    if delay is not None and "observation_delay_bins" in subset.columns:
        subset = subset[subset["observation_delay_bins"] == delay]

    if subset.empty:
        return []

    costs = subset[["forwarding_score", "route_changes", "overload_minutes"]].to_numpy()
    eff_mask = is_pareto_efficient_tri_objective(costs)
    eff_df = subset[eff_mask].copy()

    # Sort by forwarding score ascending, then overload minutes ascending
    eff_df = eff_df.sort_values(["forwarding_score", "overload_minutes", "route_changes"])

    results: list[dict[str, Any]] = []
    seen_costs = set()
    for _, row in eff_df.iterrows():
        cost_key = (
            round(float(row["forwarding_score"]), 4),
            round(float(row["route_changes"]), 2),
            round(float(row["overload_minutes"]), 2),
        )
        if cost_key in seen_costs:
            continue
        seen_costs.add(cost_key)

        entry = {
            "alpha": float(row.get("alpha", row.get("margin_quantile", 0.90))),
            "tau": float(row.get("tau", row.get("min_improvement", 0.05))),
            "observation_delay_bins": int(row.get("observation_delay_bins", 1)),
            "scenario": str(row.get("scenario", "congested")),
            "forwarding_score": float(row["forwarding_score"]),
            "route_changes": float(row["route_changes"]),
            "overload_minutes": float(row["overload_minutes"]),
            "served_fraction": float(row.get("served_fraction", 1.0)),
            "excess_mbit": float(row.get("excess_mbit", 0.0)),
            "run_id": str(row.get("run_id", "")),
        }
        results.append(entry)

    return results


def plot_sensitivity_alpha(
    df: pd.DataFrame,
    output_dir: Path,
    secondary_dir: Path | None = None,
) -> None:
    """Plots alpha sensitivity curves: alpha vs overload minutes and forwarding score."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if secondary_dir:
        secondary_dir.mkdir(parents=True, exist_ok=True)

    # Use default tau (0.05) or median tau to isolate the alpha effect
    tau_val = 0.05
    tau_col = "tau" if "tau" in df.columns else "min_improvement"
    alpha_col = "alpha" if "alpha" in df.columns else "margin_quantile"

    subset = df[np.isclose(df[tau_col], tau_val)].copy()
    if subset.empty:
        subset = df[df[tau_col] == df[tau_col].min()].copy()

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)

    # Group by alpha for congested delay=1 and delay=0
    congested_lag1 = subset[(subset["scenario"] == "congested") & (subset["observation_delay_bins"] == 1)]
    congested_lag0 = subset[(subset["scenario"] == "congested") & (subset["observation_delay_bins"] == 0)]

    c1 = congested_lag1.groupby(alpha_col)[["overload_minutes", "forwarding_score"]].mean().reset_index()
    c0 = congested_lag0.groupby(alpha_col)[["overload_minutes", "forwarding_score"]].mean().reset_index()

    color_ovld = "#d95f02"
    color_carb = "#1b9e77"

    ax1.set_xlabel(r"Safety Margin Quantile ($\alpha$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Overload Duration (min/day)", color=color_ovld, fontsize=11, fontweight="bold")
    line1 = ax1.plot(c1[alpha_col], c1["overload_minutes"], color=color_ovld, marker="o",
                     linewidth=2.2, label=r"Overload ($d=1$, 15-min lag)")
    if not c0.empty:
        line2 = ax1.plot(c0[alpha_col], c0["overload_minutes"], color=color_ovld, marker="s",
                         linestyle="--", linewidth=1.8, label=r"Overload ($d=0$, immediate)")
    ax1.tick_params(axis="y", labelcolor=color_ovld)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Forwarding Carbon Score (g $\mathrm{CO_2}$/kWh $\cdot$ hop)", color=color_carb, fontsize=11, fontweight="bold")
    line3 = ax2.plot(c1[alpha_col], c1["forwarding_score"], color=color_carb, marker="^",
                     linewidth=2.2, label=r"Carbon Score ($d=1$, 15-min lag)")
    if not c0.empty:
        line4 = ax2.plot(c0[alpha_col], c0["forwarding_score"], color=color_carb, marker="v",
                         linestyle="--", linewidth=1.8, label=r"Carbon Score ($d=0$, immediate)")
    ax2.tick_params(axis="y", labelcolor=color_carb)

    # Combine legends
    lines = line1 + (line2 if not c0.empty else []) + line3 + (line4 if not c0.empty else [])
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=True, fontsize=9)

    plt.title(r"Parameter Sensitivity: Safety Margin Quantile ($\alpha$)", fontsize=12, pad=25, fontweight="bold")
    plt.tight_layout()

    for out in [output_dir / "sensitivity_alpha_curves.png", output_dir / "sensitivity_alpha_curves.pdf"]:
        fig.savefig(out, bbox_inches="tight")
    if secondary_dir:
        for out in [secondary_dir / "sensitivity_alpha_curves.png", secondary_dir / "sensitivity_alpha_curves.pdf"]:
            fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_tau(
    df: pd.DataFrame,
    output_dir: Path,
    secondary_dir: Path | None = None,
) -> None:
    """Plots tau sensitivity curves: tau vs route changes and forwarding score."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if secondary_dir:
        secondary_dir.mkdir(parents=True, exist_ok=True)

    tau_col = "tau" if "tau" in df.columns else "min_improvement"
    alpha_col = "alpha" if "alpha" in df.columns else "margin_quantile"

    # Fix alpha = 0.90 (canonical baseline)
    alpha_val = 0.90
    subset = df[np.isclose(df[alpha_col], alpha_val)].copy()
    if subset.empty:
        subset = df[df[alpha_col] == df[alpha_col].max()].copy()

    congested_lag1 = subset[(subset["scenario"] == "congested") & (subset["observation_delay_bins"] == 1)]
    congested_lag0 = subset[(subset["scenario"] == "congested") & (subset["observation_delay_bins"] == 0)]

    c1 = congested_lag1.groupby(tau_col)[["route_changes", "forwarding_score"]].mean().reset_index()
    c0 = congested_lag0.groupby(tau_col)[["route_changes", "forwarding_score"]].mean().reset_index()

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)

    color_flaps = "#7570b3"
    color_carb = "#1b9e77"

    ax1.set_xlabel(r"Route Flap Hysteresis Threshold ($\tau$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Route Changes / Day", color=color_flaps, fontsize=11, fontweight="bold")
    line1 = ax1.plot(c1[tau_col], c1["route_changes"], color=color_flaps, marker="o",
                     linewidth=2.2, label=r"Route Changes ($d=1$)")
    if not c0.empty:
        line2 = ax1.plot(c0[tau_col], c0["route_changes"], color=color_flaps, marker="s",
                         linestyle="--", linewidth=1.8, label=r"Route Changes ($d=0$)")
    ax1.tick_params(axis="y", labelcolor=color_flaps)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Forwarding Carbon Score (g $\mathrm{CO_2}$/kWh $\cdot$ hop)", color=color_carb, fontsize=11, fontweight="bold")
    line3 = ax2.plot(c1[tau_col], c1["forwarding_score"], color=color_carb, marker="^",
                     linewidth=2.2, label=r"Carbon Score ($d=1$)")
    if not c0.empty:
        line4 = ax2.plot(c0[tau_col], c0["forwarding_score"], color=color_carb, marker="v",
                         linestyle="--", linewidth=1.8, label=r"Carbon Score ($d=0$)")
    ax2.tick_params(axis="y", labelcolor=color_carb)

    lines = line1 + (line2 if not c0.empty else []) + line3 + (line4 if not c0.empty else [])
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=True, fontsize=9)

    plt.title(r"Parameter Sensitivity: Route Flap Hysteresis Threshold ($\tau$)", fontsize=12, pad=25, fontweight="bold")
    plt.tight_layout()

    for out in [output_dir / "sensitivity_tau_curves.png", output_dir / "sensitivity_tau_curves.pdf"]:
        fig.savefig(out, bbox_inches="tight")
    if secondary_dir:
        for out in [secondary_dir / "sensitivity_tau_curves.png", secondary_dir / "sensitivity_tau_curves.pdf"]:
            fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_carbon_overload(
    df: pd.DataFrame,
    pareto_points: list[dict[str, Any]],
    output_dir: Path,
    secondary_dir: Path | None = None,
) -> None:
    """Plots 2D Pareto trade-off: Forwarding Carbon Score vs Overload Minutes with empirical convex hull."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if secondary_dir:
        secondary_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)

    # Focus on primary evaluation condition: congested, lag 1
    cg = df[(df["scenario"] == "congested") & (df["observation_delay_bins"] == 1)].copy()
    if cg.empty:
        cg = df.copy()

    alpha_col = "alpha" if "alpha" in cg.columns else "margin_quantile"
    tau_col = "tau" if "tau" in cg.columns else "min_improvement"

    # Scatter of all evaluated configurations
    scatter = ax.scatter(
        cg["overload_minutes"],
        cg["forwarding_score"],
        c=cg[alpha_col],
        cmap="viridis",
        s=45,
        alpha=0.65,
        edgecolors="none",
        label=r"Evaluated $(\alpha, \tau)$ points",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r"Safety Margin Quantile ($\alpha$)", fontsize=10)

    # Extract non-dominated points in this 2D projection
    costs_2d = cg[["overload_minutes", "forwarding_score"]].to_numpy()
    # 2D non-dominated sorting
    is_eff_2d = np.ones(len(costs_2d), dtype=bool)
    for i, c in enumerate(costs_2d):
        dominated = np.all(costs_2d <= c, axis=1) & np.any(costs_2d < c, axis=1)
        if np.any(dominated):
            is_eff_2d[i] = False

    p2d = cg[is_eff_2d].sort_values("overload_minutes")
    ax.plot(
        p2d["overload_minutes"],
        p2d["forwarding_score"],
        color="#e41a1c",
        linestyle="-",
        linewidth=2.5,
        marker="D",
        markersize=6,
        label="Empirical Pareto Frontier",
        zorder=5,
    )

    # Annotate canonical operating point (alpha=0.90, tau=0.05) if present
    canonical = cg[np.isclose(cg[alpha_col], 0.90) & np.isclose(cg[tau_col], 0.05)]
    if not canonical.empty:
        can_row = canonical.iloc[0]
        ax.scatter(
            [can_row["overload_minutes"]],
            [can_row["forwarding_score"]],
            color="#377eb8",
            edgecolors="black",
            s=120,
            linewidth=1.5,
            marker="*",
            label=r"Default Operating Knee ($\alpha=0.90, \tau=0.05$)",
            zorder=6,
        )
        ax.annotate(
            r"Knee Point ($\alpha=0.90$)",
            (can_row["overload_minutes"], can_row["forwarding_score"]),
            textcoords="offset points",
            xytext=(15, 10),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Overload Duration (minutes/day)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Forwarding Carbon Score (g $\mathrm{CO_2}$/kWh $\cdot$ hop)", fontsize=11, fontweight="bold")
    ax.set_title("Empirical Pareto Trade-off: Carbon vs. Overload Duration", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()

    for out in [output_dir / "pareto_carbon_overload.png", output_dir / "pareto_carbon_overload.pdf"]:
        fig.savefig(out, bbox_inches="tight")
    if secondary_dir:
        for out in [secondary_dir / "pareto_carbon_overload.png", secondary_dir / "pareto_carbon_overload.pdf"]:
            fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_3d_frontier(
    df: pd.DataFrame,
    pareto_points: list[dict[str, Any]],
    output_dir: Path,
    secondary_dir: Path | None = None,
) -> None:
    """Plots 3D Pareto trade-off surface across (Forwarding Score, Route Changes, Overload Minutes)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if secondary_dir:
        secondary_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 6.5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    cg = df[(df["scenario"] == "congested") & (df["observation_delay_bins"] == 1)].copy()
    if cg.empty:
        cg = df.copy()

    # All evaluated points in muted gray
    ax.scatter(
        cg["forwarding_score"],
        cg["route_changes"],
        cg["overload_minutes"],
        c="#999999",
        s=25,
        alpha=0.35,
        label=r"Dominated Configs",
    )

    # Pareto non-dominated points in vibrant red/orange
    if pareto_points:
        p_carb = [p["forwarding_score"] for p in pareto_points]
        p_flaps = [p["route_changes"] for p in pareto_points]
        p_ovld = [p["overload_minutes"] for p in pareto_points]

        ax.scatter(
            p_carb,
            p_flaps,
            p_ovld,
            c="#e41a1c",
            s=80,
            edgecolors="black",
            linewidth=1.2,
            marker="o",
            label="Non-Dominated Pareto Frontier",
            zorder=10,
        )

        # Plot projection drop lines to XY floor for visual clarity
        min_ovld = min(p_ovld) if p_ovld else 0.0
        for x, y, z in zip(p_carb, p_flaps, p_ovld):
            ax.plot([x, x], [y, y], [min_ovld, z], color="#e41a1c", linestyle=":", alpha=0.4)

    ax.set_xlabel(r"Carbon Score ($\mathcal{S}$)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel(r"Route Changes ($\mathcal{R}$)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_zlabel(r"Overload Minutes ($\mathcal{T}$)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_title("Tri-Objective Pareto Frontier: (Carbon, Route Flaps, Overload)", fontsize=12, pad=15, fontweight="bold")
    ax.view_init(elev=24, azim=130)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()

    for out in [output_dir / "pareto_3d_frontier.png", output_dir / "pareto_3d_frontier.pdf"]:
        fig.savefig(out, bbox_inches="tight")
    if secondary_dir:
        for out in [secondary_dir / "pareto_3d_frontier.png", secondary_dir / "pareto_3d_frontier.pdf"]:
            fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def run_sensitivity_sweep(
    root: Path | None = None,
    base_cfg: dict[str, Any] | None = None,
    quantiles: list[float] | None = None,
    tau_thresholds: list[float] | None = None,
    delays: list[int] | None = None,
    scenarios: list[str] | None = None,
    output_dir: Path | None = None,
    artifact_dir: Path | None = None,
    seed: int | None = None,
    window_idx: int = 0,
    carbon_year: int = 2022,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Orchestrates parameter sensitivity sweep, non-dominated sorting, and figure generation."""
    if root is None:
        root = ROOT
    if base_cfg is None:
        base_cfg = config()
    if quantiles is None:
        quantiles = DEFAULT_ALPHAS
    if tau_thresholds is None:
        tau_thresholds = DEFAULT_TAUS
    if delays is None:
        delays = DEFAULT_DELAYS
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if output_dir is None:
        output_dir = root / "results" / "sensitivity"
    if seed is None:
        seed = base_cfg.get("seed", 42)

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    shared_fig_dir = root / "figures"
    shared_fig_dir.mkdir(parents=True, exist_ok=True)

    # Locate trained model checkpoint: default to reference artifacts/ directory
    if artifact_dir is None:
        art_dir = root / "artifacts"
    else:
        art_dir = Path(artifact_dir)

    # Load data with config matching preparation
    prep_manifest_path = root / "inputs/processed/manifest.json"
    if prep_manifest_path.exists():
        prep_seed = load_json(prep_manifest_path).get("config", {}).get("seed", seed)
        prep_cfg = dict(base_cfg, seed=prep_seed)
    else:
        prep_cfg = dict(base_cfg, seed=seed)

    data = load_prepared(root, prep_cfg)
    carbon_table = CarbonTable(root / "inputs/raw/carbon.csv")
    lo, hi = data.manifest["splits"]["test"]
    ci, alignment = carbon_table.pair(data.times[lo:hi], carbon_year)

    # Save alignment file for verify_results.py audit
    alignment_file = output_dir / f"alignment_{carbon_year}.json"
    save_json(alignment_file, alignment)
    alignment_sha = sha256(alignment_file)

    windows = evaluation_windows(data, prep_cfg, 5)
    start, stop = windows[window_idx]

    # Load GNN predictor
    gnn = TrainedPredictor(art_dir, data, prep_cfg, seed=seed)

    # Precompute validation residuals to calibrate alpha quantiles instantaneously without retraining
    print("Precomputing validation residuals for decoupled alpha calibration...", flush=True)
    yval, val_pred = get_cached_validation_residuals(data, gnn, prep_cfg)
    residual_cache = {}
    for a in quantiles:
        residual_cache[a] = calibrate_margin(yval, val_pred, data.mask, a)

    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "sensitivity",
        "status": "running",
        "config": prep_cfg,
        "config_hash": fingerprint(prep_cfg),
        "source_revision": revision,
        "source_files": {
            str(p.relative_to(root)).replace("\\", "/"): sha256(p)
            for p in sorted((root / "src/carbonroute").glob("*")) if p.is_file()
        },
        "graph": data.network.metadata(),
        "graph_hash": data.network.graph_hash,
        "traffic_manifest": data.manifest,
        "checkpoint_sha256": sha256(art_dir / "gnn.pt"),
        "carbon_sha256": carbon_table.source_hash,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "runs": [],
        "failed_runs": [],
        "alignment_sha256": {str(carbon_year): alignment_sha},
        "interpretation": "Parameter sensitivity sweep over alpha quantiles, tau thresholds, and delays.",
    }
    save_json(output_dir / "manifest.json", manifest)

    summaries: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    total_runs = len(scenarios) * len(delays) * len(quantiles) * len(tau_thresholds)
    print(f"Beginning sensitivity sweep: {total_runs} combinations...", flush=True)
    run_idx = 0

    for scenario in scenarios:
        traffic_seed = seed + window_idx * 101
        rates = workload(data, prep_cfg, start, stop, scenario, traffic_seed)
        traffic_hash = hashlib.sha256(rates[start - prep_cfg["lookback"]:stop].tobytes()).hexdigest()
        day_ci = ci[start - lo:stop - lo]

        for delay in delays:
            for alpha in quantiles:
                # Decoupled margin assignment
                gnn.margin = residual_cache[alpha]

                for tau in tau_thresholds:
                    run_idx += 1
                    # Pass min_improvement via run-specific config copy
                    run_cfg = copy.deepcopy(prep_cfg)
                    run_cfg["min_improvement"] = tau
                    run_cfg["margin_quantile"] = alpha

                    run_id = f"sens_w{window_idx}_{carbon_year}_{scenario}_lag{delay}_a{int(round(alpha*100))}_t{int(round(tau*100))}"
                    run_file = f"runs/{run_id}.json"
                    run_path = output_dir / run_file

                    fields = {
                        "run_id": run_id,
                        "window": window_idx,
                        "scenario": scenario,
                        "carbon_year": carbon_year,
                        "seed": traffic_seed,
                        "start": start,
                        "stop": stop,
                        "traffic_start": str(data.times[start]),
                        "traffic_sha256": traffic_hash,
                        "alpha": float(alpha),
                        "margin_quantile": float(alpha),
                        "tau": float(tau),
                        "min_improvement": float(tau),
                        "observation_delay_bins": int(delay),
                        "policy": "predictive_carbon",
                    }

                    try:
                        result = replay(data, run_cfg, rates, day_ci, start, stop, "predictive_carbon", gnn, delay)
                        summary = {**fields, **result.summary}
                        save_json(run_path, {"summary": summary, "records": result.records})
                        summaries.append(summary)

                        manifest["runs"].append({
                            **fields,
                            "file": run_file,
                            "sha256": sha256(run_path),
                        })

                        if run_idx % 20 == 0 or run_idx == total_runs:
                            print(f"Progress: [{run_idx}/{total_runs}] completed ({time.perf_counter()-t_start:.1f}s)", flush=True)

                    except Exception as exc:
                        manifest["status"] = "failed"
                        manifest["failed_runs"].append({**fields, "error": repr(exc)})
                        save_json(output_dir / "manifest.json", manifest)
                        raise

    elapsed = time.perf_counter() - t_start
    manifest["runtime_seconds"] = elapsed
    print(f"Finished {len(summaries)} sweep runs in {elapsed:.2f} seconds.", flush=True)

    summary_df = pd.DataFrame(summaries)
    summary_csv = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    manifest["summary_sha256"] = sha256(summary_csv)
    manifest["status"] = "complete"
    save_json(output_dir / "manifest.json", manifest)

    # Extract non-dominated Pareto frontier points
    print("Extracting empirical Pareto frontier...", flush=True)
    pareto_points = extract_pareto_frontier(summary_df, scenario="congested", delay=1)
    pareto_json = output_dir / "pareto_frontier.json"
    save_json(pareto_json, pareto_points)
    print(f"Extracted {len(pareto_points)} non-dominated Pareto points to {pareto_json}.", flush=True)

    # Generate publication figures
    print("Generating publication figures...", flush=True)
    plot_sensitivity_alpha(summary_df, fig_dir, shared_fig_dir)
    plot_sensitivity_tau(summary_df, fig_dir, shared_fig_dir)
    plot_pareto_carbon_overload(summary_df, pareto_points, fig_dir, shared_fig_dir)
    plot_pareto_3d_frontier(summary_df, pareto_points, fig_dir, shared_fig_dir)
    print(f"Figures saved in {fig_dir} and {shared_fig_dir}.", flush=True)

    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "total_runs": len(summaries),
        "runtime_seconds": elapsed,
        "pareto_points_count": len(pareto_points),
        "summary_csv": str(summary_csv),
        "pareto_frontier_json": str(pareto_json),
    }
