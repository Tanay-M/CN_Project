"""Multi-seed training, campaign execution, statistical testing, and audit runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure ROOT and src are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from carbonroute.common import ROOT, config
from carbonroute.experiments import run_multiseed
from scripts.verify_results import verify as auditor_verify


def main() -> None:
    parser = argparse.ArgumentParser(description="CarbonRoute Multi-Seed & Statistical Significance Campaign Runner")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="List of random seeds to evaluate (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Run full paper campaign (495 runs per seed across 5 days, 3 years, 3 scenarios) (default: True)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing seed campaign results if present (default: True)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify all generated seed campaign directories using scripts/verify_results.py (default: True)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=False,
        help="Skip re-training and re-running campaigns if seed artifacts already exist (default: False)",
    )
    args = parser.parse_args()

    cfg = config()
    print("=" * 70)
    print("CarbonRoute Multi-Seed Campaign Runner")
    print(f"Seeds: {args.seeds}")
    print(f"Paper Mode: {args.paper}")
    print(f"Root: {ROOT}")
    print("=" * 70)

    run_multiseed(
        seeds=args.seeds,
        root=ROOT,
        base_cfg=cfg,
        paper=args.paper,
        overwrite=args.overwrite,
        skip_completed=args.skip_completed,
    )

    if args.verify:
        print("\n" + "=" * 70)
        print("Auditing Generated Seed Campaign Directories with scripts/verify_results.py")
        print("=" * 70)
        seeds_parent = ROOT / "results/seeds"
        for s in args.seeds:
            s_dir = seeds_parent / f"seed_{s}"
            print(f"\nVerifying {s_dir} ...")
            auditor_verify(s_dir)

    print("\n" + "=" * 70)
    print("Multi-Seed Campaign Completed and Verified Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
