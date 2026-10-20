"""One entry point for preparation, training, replay, reporting and handoff."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import webbrowser
import zipfile
from datetime import datetime
from pathlib import Path

from .common import ROOT, config


def bundle(root: Path) -> Path:
    """Source, trained models and completed reports; no original third-party files."""
    destination = root / "dist" / f"geant-project-{datetime.now():%Y%m%d-%H%M%S}.zip"
    destination.parent.mkdir(exist_ok=True)
    folders = ("src", "configs", "docs", "tests", "scripts", "artifacts", "results/demo", "results/paper")
    root_files = ("README.md", "IMPLEMENTATION_PLAN.md", "pyproject.toml", "requirements-lock.txt",
                  "requirements-torch.txt", "SETUP.ps1", "RUN_DEMO.ps1", "RUN_PAPER.ps1", "DEMO.html", ".gitignore")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in folders:
            for path in (root / folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and ".egg-info" not in str(path):
                    archive.write(path, path.relative_to(root))
        for name in root_files:
            if (root / name).is_file():
                archive.write(root / name, name)
    print(f"Portable source + trained models + offline demo and paper results: {destination}")
    print("Original source data and the Python environment are not redistributed. Run prepare after setup to reproduce.")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Free, offline-capable GEANT carbon-aware routing research")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root containing configs/ and inputs/")
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "doctor", "bundle"):
        sub.add_parser(command)
    for command in ("run", "report", "demo"):
        child = sub.add_parser(command)
        child.add_argument("--output", type=Path, default=None)
        if command in ("run", "demo"):
            child.add_argument("--overwrite", action="store_true")
        if command == "run":
            child.add_argument("--paper", action="store_true")
            child.add_argument("--joint", action="store_true", help="Include the joint carbon-budgeted selectors")
            child.add_argument("--split", choices=("validation", "test"), default="test")
            child.add_argument("--carbon-budget-fraction", type=float, default=0.0,
                               help="Allowed predicted carbon cost above the reactive assignment")
        if command in ("report", "demo"):
            child.add_argument("--open", action="store_true", help="Open the local HTML report in your browser")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        cfg = config(args.config or root / "configs/default.json")
        if args.command == "doctor":
            versions = {name: importlib.metadata.version(name) for name in
                        ("torch", "numpy", "pandas", "networkx", "matplotlib")}
            print(json.dumps({"python": sys.version, "packages": versions, "project": str(root),
                              "prepared": (root / "inputs/processed/manifest.json").exists(),
                              "trained": (root / "artifacts/gnn.pt").exists()}, indent=2))
        elif args.command == "bundle":
            bundle(root)
        elif args.command == "prepare":
            from .data import prepare
            prepare(root, cfg)
        elif args.command == "train":
            from .prediction import train
            train(root, cfg)
        else:
            name = (f"joint-{args.split}" if args.command == "run" and args.joint else
                    "paper" if getattr(args, "paper", False) else "demo")
            output = (args.output or root / "results" / name).resolve()
            if args.command in ("run", "demo"):
                from .experiments import campaign
                if args.command == "demo":
                    if not (root / "inputs/processed/manifest.json").exists():
                        from .data import prepare
                        prepare(root, cfg)
                    if not (root / "artifacts/gnn.pt").exists():
                        from .prediction import train
                        train(root, cfg)
                campaign(root, cfg, output, getattr(args, "paper", False), args.overwrite,
                         getattr(args, "split", "test"), getattr(args, "joint", False),
                         getattr(args, "carbon_budget_fraction", 0.0))
            from .reporting import report
            path = report(root, output)
            if getattr(args, "open", False):
                webbrowser.open(path.as_uri())
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
