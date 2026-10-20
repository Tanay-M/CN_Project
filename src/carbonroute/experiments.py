"""Paired trace/carbon campaigns and window-clustered descriptive uncertainty."""

from __future__ import annotations

import copy
import hashlib
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import fingerprint, load_json, save_json, sha256
from .data import CarbonTable, Prepared, load_prepared
from .prediction import Persistence, TrainedPredictor
from .routing import JOINT_POLICIES, POLICIES
from .simulation import replay

MAIN_POLICIES = POLICIES[:4]


def evaluation_windows(data: Prepared, cfg: dict[str, Any], count: int,
                       split: str = "test") -> list[tuple[int, int]]:
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    lo, hi = data.manifest["splits"][split]
    candidates = []
    for start in range(lo+cfg["lookback"], hi-96+1):
        if data.times[start] != data.times[start].astype("datetime64[D]"):
            continue
        if np.all(np.diff(data.times[start-cfg["lookback"]:start+96]) == np.timedelta64(15, "m")):
            candidates.append((start, start+96))
    if len(candidates) < count:
        raise ValueError(f"Need {count} complete {split} days, found {len(candidates)}")
    return candidates[:count]


def workload(data: Prepared, cfg: dict[str, Any], start: int, stop: int,
             scenario: str, seed: int) -> np.ndarray:
    factor = cfg["congested_multiplier"] if scenario == "congested" else cfg["normal_multiplier"]
    rates = data.rates.copy()*factor
    if scenario == "bursty":
        rng = np.random.default_rng(seed)
        lo = start-cfg["lookback"]
        # Bursts are unannounced, per-flow events; their past portions become observable.
        for a, b in data.pairs:
            events = np.flatnonzero(rng.random(stop-lo) < cfg["burst_probability"]) + lo
            multiplier = np.ones(stop-lo)
            for k in events:
                multiplier[k-lo:min(k-lo+cfg["burst_bins"], stop-lo)] = cfg["burst_multiplier"]
            rates[lo:stop, a, b] *= multiplier
    elif scenario not in ("normal", "congested"):
        raise ValueError(f"Unknown scenario {scenario}")
    return rates


def paired_differences(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Cluster bootstrap over DAYS; carbon mappings of a day are not independent.

    Five windows yield preliminary, wide uncertainty, not publication-ready proof.
    Single-window pilots intentionally have no confidence interval.
    """
    metrics = ("forwarding_score", "served_fraction", "overload_minutes", "route_changes")
    rows = []
    for (scenario, delay), subset in frame.groupby(["scenario", "observation_delay_bins"]):
        baseline = subset[subset.policy == "reactive_carbon"]
        for policy in subset.policy.unique():
            if policy == "reactive_carbon":
                continue
            joined = subset[subset.policy == policy].merge(
                baseline, on=["window", "carbon_year"], suffixes=("", "_reactive"), validate="one_to_one")
            for metric in metrics:
                joined["difference"] = joined[metric]-joined[f"{metric}_reactive"]
                clusters = joined.groupby("window").difference.mean().to_numpy()
                if not len(clusters):
                    continue
                low = high = None
                if len(clusters) >= 2:
                    rng = np.random.default_rng(seed)
                    boot = rng.choice(clusters, (2000, len(clusters)), replace=True).mean(axis=1)
                    low, high = np.quantile(boot, [.025, .975]).tolist()
                rows.append({"policy": policy, "scenario": scenario, "observation_delay_bins": delay,
                             "metric": metric, "difference_vs_reactive": float(clusters.mean()),
                             "ci95_low": low, "ci95_high": high, "independent_day_clusters": len(clusters)})
    return pd.DataFrame(rows)


def campaign(
    root: Path,
    cfg: dict[str, Any],
    output: Path,
    paper: bool = False,
    overwrite: bool = False,
    split: str = "test",
    include_joint: bool = False,
    carbon_budget_fraction: float = 0.0,
    artifact_dir: Path | None = None,
    seed: int | None = None,
) -> Path:
    if (output / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"Results already exist in {output}. Choose another --output or use --overwrite.")
    if seed is not None:
        cfg = copy.deepcopy(cfg)
        cfg["seed"] = seed
    art_dir = Path(artifact_dir) if artifact_dir is not None else (root / "artifacts")
    prep_manifest_path = root / "inputs/processed/manifest.json"
    if prep_manifest_path.exists():
        prep_seed = load_json(prep_manifest_path).get("config", {}).get("seed", cfg["seed"])
        prep_cfg = dict(cfg, seed=prep_seed)
    else:
        prep_cfg = cfg
    data = load_prepared(root, prep_cfg)
    carbon = CarbonTable(root / "inputs/raw/carbon.csv")
    expected_carbon = data.manifest["sources"]["carbon.csv"]["sha256"]
    if carbon.source_hash != expected_carbon:
        raise ValueError("Carbon input changed after preparation")
    gnn = TrainedPredictor(art_dir, data, cfg, seed=cfg["seed"])
    gru = TrainedPredictor(art_dir, data, cfg, "gru", seed=cfg["seed"]) if include_joint else None
    persist_meta = load_json(art_dir / "persistence.json")
    if persist_meta["config_hash"] != fingerprint(cfg) or persist_meta["traffic_sha256"] != data.manifest["traffic_sha256"]:
        raise ValueError("Persistence margin does not match this data/configuration")
    persistence = Persistence(cfg["horizon"], data.mask, np.array(persist_meta["margin_mbps"]))
    day_count = (7 if split == "validation" else 5) if paper else 1
    windows = evaluation_windows(data, cfg, day_count, split)
    years = cfg["carbon_years"] if paper else cfg["carbon_years"][:1]
    scenarios = ("normal", "congested", "bursty") if paper else ("normal", "congested")
    delays = (1, 0) if paper and not include_joint else (1,)
    output.mkdir(parents=True, exist_ok=True)
    run_folder = output / "runs"
    run_folder.mkdir(exist_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": ("joint-development" if split == "validation" else
                 "joint-test-exploratory" if paper else "joint-test-pilot") if include_joint
                else ("paper" if paper else "pilot"),
        "evaluation_split": split, "include_joint": include_joint,
        "carbon_budget_fraction": carbon_budget_fraction,
        "status": "running",
        "config": cfg, "config_hash": fingerprint(cfg), "source_revision": revision,
        "source_files": {str(p.relative_to(root)).replace("\\", "/"): sha256(p)
                         for p in sorted((root / "src/carbonroute").glob("*")) if p.is_file()},
        "graph": data.network.metadata(), "graph_hash": data.network.graph_hash,
        "traffic_manifest": data.manifest,
        "checkpoint_sha256": sha256(art_dir / "gnn.pt"),
        "checkpoint_sha256_by_model": {
            "gnn": sha256(art_dir / "gnn.pt"),
            "gru": sha256(art_dir / "gru.pt"),
            "persistence": sha256(art_dir / "persistence.json"),
        } if include_joint else {},
        "carbon_sha256": carbon.source_hash, "python": platform.python_version(),
        "platform": platform.platform(), "runs": [], "failed_runs": [],
        "alignment_sha256": {},
        "interpretation": "Constructed scenario: 2005 traffic and later daily country carbon; aggregate service, no packet or physical energy measurements",
        "power_sensitivity": {"idle_only": {"idle_w": 50, "beta_w_per_mbps": 0},
                              "idle_dominated": {"idle_w": 50, "beta_w_per_mbps": .1},
                              "load_sensitive": {"idle_w": 5, "beta_w_per_mbps": 1}},
    }
    # Freeze forecast evidence with this campaign so later retraining cannot change its report.
    for name in ("forecast_metrics.csv", "training_summary.json"):
        shutil.copyfile(art_dir / name, output / name)
    manifest["forecast_metrics_sha256"] = sha256(output / "forecast_metrics.csv")
    manifest["training_summary_sha256"] = sha256(output / "training_summary.json")
    save_json(output / "manifest.json", manifest)
    lo, hi = data.manifest["splits"][split]
    carbon_cache = {}
    for year in years:
        ci, alignment = carbon.pair(data.times[lo:hi], year)
        carbon_cache[year] = ci
        alignment_path = output / f"alignment_{year}.json"
        save_json(alignment_path, alignment)
        manifest["alignment_sha256"][str(year)] = sha256(alignment_path)
    summaries = []
    started = time.perf_counter()
    for window_id, (start, stop) in enumerate(windows):
        for scenario in scenarios:
            seed = cfg["seed"] + window_id * 101
            rates = workload(data, cfg, start, stop, scenario, seed)
            traffic_hash = hashlib.sha256(rates[start-cfg["lookback"]:stop].tobytes()).hexdigest()
            for year in years:
                ci = carbon_cache[year][start-lo:stop-lo]
                for delay in delays:
                    policies = (POLICIES + JOINT_POLICIES if include_joint else POLICIES) if delay == 1 else MAIN_POLICIES
                    for policy in policies:
                        run_id = f"w{window_id}_{year}_{scenario}_lag{delay}_{policy}"
                        fields = {"run_id": run_id, "window": window_id, "scenario": scenario,
                                  "carbon_year": year, "seed": seed, "start": start, "stop": stop,
                                  "traffic_start": str(data.times[start]), "traffic_sha256": traffic_hash}
                        try:
                            predictor = (persistence if policy in ("persistence_carbon", "joint_persistence")
                                         else gru if policy == "joint_gru" else gnn)
                            assert predictor is not None
                            result = replay(data, cfg, rates, ci, start, stop, policy, predictor, delay,
                                            carbon_budget_fraction)
                            summary = {**fields, **result.summary}
                            run_path = run_folder / f"{run_id}.json"
                            save_json(run_path, {"summary": summary, "records": result.records})
                            summaries.append(summary)
                            manifest["runs"].append({**fields, "policy": policy, "observation_delay_bins": delay,
                                                     "file": f"runs/{run_id}.json", "sha256": sha256(run_path)})
                        except Exception as exc:
                            manifest["status"] = "failed"
                            manifest["failed_runs"].append({**fields, "error": repr(exc)})
                            save_json(output / "manifest.json", manifest)
                            raise
            print(f"Completed window {window_id+1}/{len(windows)}, {scenario}; {len(summaries)} paired runs", flush=True)
    manifest["runtime_seconds"] = time.perf_counter()-started
    frame = pd.DataFrame(summaries)
    frame.to_csv(output / "summary.csv", index=False)
    paired_differences(frame, cfg["seed"]).to_csv(output / "paired_differences.csv", index=False)
    manifest["summary_sha256"] = sha256(output / "summary.csv")
    manifest["status"] = "complete"
    save_json(output / "manifest.json", manifest)
    print(f"Saved {len(frame)} runs in {manifest['runtime_seconds']:.1f}s to {output}", flush=True)
    return output


def compute_paired_significance(frame: pd.DataFrame) -> pd.DataFrame:
    """Computes mean, std, paired Student's t-test, and Wilcoxon signed-rank test.

    Compares GNN+Safeguards (predictive_carbon) against:
      - Reactive Carbon (reactive_carbon)
      - Persistence+Safeguards (persistence_carbon)
    across overload_minutes, forwarding_score, served_fraction, and route_changes.
    Gracefully handles zero-variance and tied differences (p-value = 1.0).
    """
    from scipy import stats

    comparisons = [
        ("predictive_carbon", "reactive_carbon", "vs_reactive"),
        ("predictive_carbon", "persistence_carbon", "vs_persistence"),
    ]
    metrics = ["overload_minutes", "forwarding_score", "served_fraction", "route_changes"]
    results: list[dict[str, Any]] = []

    # Pairing keys across realizations: strictly enforce model_seed if present
    merge_keys = ["window", "carbon_year"]
    if "model_seed" in frame.columns:
        merge_keys.append("model_seed")
    elif "seed" in frame.columns:
        merge_keys.append("seed")

    for (scenario, delay), group in frame.groupby(["scenario", "observation_delay_bins"]):
        for target, baseline, comp_name in comparisons:
            tgt_df = group[group.policy == target]
            base_df = group[group.policy == baseline]
            if tgt_df.empty or base_df.empty:
                continue

            merged = tgt_df.merge(base_df, on=merge_keys, suffixes=("_tgt", "_base"))
            if merged.empty:
                continue

            for m in metrics:
                if f"{m}_tgt" not in merged.columns or f"{m}_base" not in merged.columns:
                    continue

                # Aggregate across repeated realizations of the same traffic window (day)
                # to avoid pseudoreplication across identical traffic demand inputs.
                if "window" in merged.columns and merged["window"].nunique() > 1:
                    day_tgt = merged.groupby("window")[f"{m}_tgt"].mean()
                    day_base = merged.groupby("window")[f"{m}_base"].mean()
                    diff = (day_tgt - day_base).to_numpy(dtype=float)
                else:
                    diff = (merged[f"{m}_tgt"] - merged[f"{m}_base"]).to_numpy(dtype=float)

                n_obs = len(diff)
                mean_diff = float(np.mean(diff))
                std_diff = float(np.std(diff, ddof=1)) if n_obs > 1 else 0.0

                if np.allclose(diff, 0.0) or (n_obs > 1 and np.isclose(std_diff, 0.0)):
                    t_stat, p_t = 0.0, 1.0
                    w_stat, p_w = 0.0, 1.0
                else:
                    try:
                        t_res = stats.ttest_1samp(diff, 0.0)
                        t_stat = float(t_res.statistic)
                        p_t = float(t_res.pvalue)
                        if np.isnan(p_t):
                            t_stat, p_t = 0.0, 1.0
                    except Exception:
                        t_stat, p_t = 0.0, 1.0

                    try:
                        w_res = stats.wilcoxon(diff, alternative="two-sided")
                        w_stat = float(w_res.statistic)
                        p_w = float(w_res.pvalue)
                        if np.isnan(p_w):
                            w_stat, p_w = 0.0, 1.0
                    except Exception:
                        w_stat, p_w = 0.0, 1.0

                results.append({
                    "scenario": scenario,
                    "observation_delay_bins": int(delay),
                    "comparison": comp_name,
                    "target_policy": target,
                    "baseline_policy": baseline,
                    "metric": m,
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                    "t_statistic": t_stat,
                    "p_value_ttest": p_t,
                    "wilcoxon_stat": w_stat,
                    "p_value_wilcoxon": p_w,
                    "n_pairs": int(len(merged)),
                    "n_days": int(n_obs),
                })

    return pd.DataFrame(results)


def compute_statistical_tests(frame: pd.DataFrame) -> dict[str, Any]:
    """Computes paired statistical tests and returns a structured dictionary matching PROJECT.md interface contracts."""
    sig_df = compute_paired_significance(frame)
    out: dict[str, Any] = {
        "vs_reactive": {},
        "vs_persistence": {},
    }

    for row in sig_df.to_dict(orient="records"):
        comp = row["comparison"]
        scen_lag = f"{row['scenario']}_lag{row['observation_delay_bins']}"
        metric = row["metric"]

        entry = {
            "mean_diff": row["mean_diff"],
            "std_diff": row["std_diff"],
            "t_statistic": row["t_statistic"],
            "p_value_ttest": row["p_value_ttest"],
            "wilcoxon_stat": row["wilcoxon_stat"],
            "p_value_wilcoxon": row["p_value_wilcoxon"],
            "n_pairs": row["n_pairs"],
            "n_days": row.get("n_days", row["n_pairs"]),
        }

        # Top-level direct mapping for primary paper condition (congested_lag1)
        if scen_lag == "congested_lag1":
            if comp not in out:
                out[comp] = {}
            out[comp][metric] = entry

        # Scenario-specific comparison key
        full_comp_key = f"{comp}_{scen_lag}"
        if full_comp_key not in out:
            out[full_comp_key] = {}
        out[full_comp_key][metric] = entry

    return out


def run_multiseed(
    seeds: list[int] | None = None,
    root: Path | None = None,
    base_cfg: dict[str, Any] | None = None,
    paper: bool = True,
    overwrite: bool = True,
    skip_completed: bool = False,
) -> dict[str, Any]:
    """Orchestrates multi-seed training, campaign runs, aggregation, and statistical significance testing."""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    if root is None:
        from .common import ROOT
        root = ROOT
    if base_cfg is None:
        from .common import config
        base_cfg = config()

    from .prediction import train

    multiseed_results_dir = root / "results" / "multiseed"
    multiseed_results_dir.mkdir(parents=True, exist_ok=True)
    seeds_results_dir = root / "results" / "seeds"
    seeds_results_dir.mkdir(parents=True, exist_ok=True)
    seeds_artifacts_dir = root / "artifacts" / "seeds"
    seeds_artifacts_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries: list[pd.DataFrame] = []
    forecast_summaries: list[pd.DataFrame] = []

    for s in seeds:
        print("\n==========================================", flush=True)
        print(f"  Starting Multi-Seed Pipeline for Seed {s}", flush=True)
        print("==========================================", flush=True)

        out_dir_s = seeds_results_dir / f"seed_{s}"
        art_dir_s = seeds_artifacts_dir / f"seed_{s}"
        is_completed = (
            skip_completed
            and (out_dir_s / "manifest.json").exists()
            and (out_dir_s / "summary.csv").exists()
            and (art_dir_s / "gnn.pt").exists()
        )

        if not is_completed:
            cfg_s = copy.deepcopy(base_cfg)
            cfg_s["seed"] = s
            train(root, cfg_s, seed=s, output_dir=art_dir_s)
            campaign(root, cfg_s, output=out_dir_s, paper=paper, overwrite=overwrite, artifact_dir=art_dir_s, seed=s)
        else:
            print(f"  Seed {s} already completed, loading existing outputs...", flush=True)

        s_summary = pd.read_csv(out_dir_s / "summary.csv")
        s_summary["model_seed"] = s
        seed_summaries.append(s_summary)

        if (art_dir_s / "forecast_metrics.csv").exists():
            fc = pd.read_csv(art_dir_s / "forecast_metrics.csv")
            fc["seed"] = s
            forecast_summaries.append(fc)

    all_runs_df = pd.concat(seed_summaries, ignore_index=True)
    all_runs_df.to_csv(multiseed_results_dir / "all_runs.csv", index=False)

    metric_cols = [
        "overload_minutes", "forwarding_score", "served_fraction", "route_changes",
        "max_utilization", "p95_utilization", "excess_mbit", "decision_ms_mean",
        "idle_only_g", "idle_dominated_g", "load_sensitive_g"
    ]
    avail_metrics = [c for c in metric_cols if c in all_runs_df.columns]

    # Per-seed average per policy, scenario, delay
    per_seed_grouped = all_runs_df.groupby(["model_seed", "policy", "scenario", "observation_delay_bins"])[avail_metrics].mean().reset_index()
    per_seed_grouped.rename(columns={"model_seed": "seed"}, inplace=True)
    per_seed_grouped.to_csv(multiseed_results_dir / "seed_breakdown.csv", index=False)

    # Aggregate across seeds
    agg_dict = {col: ["mean", "std", "min", "max"] for col in avail_metrics}
    agg_df = per_seed_grouped.groupby(["policy", "scenario", "observation_delay_bins"]).agg(agg_dict)

    # Compute overall mean and std maps
    stats_map = {}
    for (pol, scen, dly), row in agg_df.iterrows():
        stats_map[(pol, scen, dly)] = {}
        for col in avail_metrics:
            stats_map[(pol, scen, dly)][f"{col}_mean"] = float(row[(col, "mean")])
            stats_map[(pol, scen, dly)][f"{col}_std"] = float(row[(col, "std")]) if not np.isnan(row[(col, "std")]) else 0.0
            stats_map[(pol, scen, dly)][f"{col}_min"] = float(row[(col, "min")])
            stats_map[(pol, scen, dly)][f"{col}_max"] = float(row[(col, "max")])

    # Enhance per-seed rows with mean and std columns
    enhanced_rows = []
    for _, row in per_seed_grouped.iterrows():
        key = (row["policy"], row["scenario"], row["observation_delay_bins"])
        sm = stats_map.get(key, {})
        enhanced_rows.append({**row.to_dict(), **sm})

    # Also add "all" seed rows representing the aggregate
    for (pol, scen, dly), sm in stats_map.items():
        all_row = {
            "seed": "all",
            "policy": pol,
            "scenario": scen,
            "observation_delay_bins": dly,
        }
        for col in avail_metrics:
            all_row[col] = sm[f"{col}_mean"]
        all_row.update(sm)
        enhanced_rows.append(all_row)

    summary_multiseed = pd.DataFrame(enhanced_rows)
    summary_multiseed.to_csv(multiseed_results_dir / "summary_multiseed.csv", index=False)
    summary_multiseed.to_csv(root / "results" / "multi_seed_summary.csv", index=False)

    stat_tests = compute_statistical_tests(all_runs_df)
    save_json(multiseed_results_dir / "statistical_significance.json", stat_tests)
    save_json(root / "results" / "statistical_significance.json", stat_tests)

    stat_df = compute_paired_significance(all_runs_df)
    stat_df.to_csv(multiseed_results_dir / "statistical_tests.csv", index=False)
    stat_df.to_csv(root / "results" / "statistical_tests.csv", index=False)

    if forecast_summaries:
        all_fc = pd.concat(forecast_summaries, ignore_index=True)
        fc_agg = all_fc.groupby(["model", "split", "horizon_bins"])[["mae_mbps", "rmse_mbps"]].agg(["mean", "std"]).reset_index()
        fc_flat = pd.DataFrame()
        fc_flat["model"] = fc_agg["model"]
        fc_flat["split"] = fc_agg["split"]
        fc_flat["horizon_bins"] = fc_agg["horizon_bins"]
        fc_flat["mae_mbps_mean"] = fc_agg[("mae_mbps", "mean")]
        fc_flat["mae_mbps_std"] = fc_agg[("mae_mbps", "std")].fillna(0.0)
        fc_flat["rmse_mbps_mean"] = fc_agg[("rmse_mbps", "mean")]
        fc_flat["rmse_mbps_std"] = fc_agg[("rmse_mbps", "std")].fillna(0.0)
        fc_flat["mae_mbps"] = fc_flat["mae_mbps_mean"]
        fc_flat["rmse_mbps"] = fc_flat["rmse_mbps_mean"]
        fc_flat.to_csv(multiseed_results_dir / "forecast_multiseed_summary.csv", index=False)

    print(f"\nMulti-Seed Campaign complete across seeds {seeds}. Results saved to {multiseed_results_dir}.", flush=True)
    return {
        "status": "complete",
        "seeds": seeds,
        "results_dir": str(multiseed_results_dir),
        "statistical_tests": stat_tests,
    }
