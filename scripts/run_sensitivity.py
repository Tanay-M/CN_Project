"""CLI Driver for Milestone 3 Parameter Sensitivity & Robustness Ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root and src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carbonroute.common import config
from carbonroute.sensitivity import (
    DEFAULT_ALPHAS,
    DEFAULT_DELAYS,
    DEFAULT_SCENARIOS,
    DEFAULT_TAUS,
    run_sensitivity_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CarbonRoute Parameter Sensitivity & Robustness Sweep")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root directory")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "sensitivity", help="Output directory")
    parser.add_argument("--artifact-dir", type=Path, default=None, help="Trained model artifact directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--window", type=int, default=0, help="Test evaluation window index")
    parser.add_argument("--year", type=int, default=2022, help="Carbon year")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing results")
    args = parser.parse_args()

    cfg = config()
    result = run_sensitivity_sweep(
        root=args.root,
        base_cfg=cfg,
        quantiles=DEFAULT_ALPHAS,
        tau_thresholds=DEFAULT_TAUS,
        delays=DEFAULT_DELAYS,
        scenarios=DEFAULT_SCENARIOS,
        output_dir=args.output,
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        window_idx=args.window,
        carbon_year=args.year,
        overwrite=args.overwrite,
    )

    print("\n" + "=" * 60)
    print("Parameter Sensitivity Sweep Completed Successfully!")
    print(f"Total Runs: {result['total_runs']}")
    print(f"Runtime: {result['runtime_seconds']:.2f} seconds")
    print(f"Non-Dominated Pareto Points: {result['pareto_points_count']}")
    print(f"Output Directory: {result['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
