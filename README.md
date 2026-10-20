# GÉANT predictive carbon-aware routing

A working, free, native Windows research simulation with real historical inputs, a trained GCN–GRU, seven reference policies, three experimental joint selectors, and an offline replay viewer.

**Start here: double-click `DEMO.html`.** The saved demo works without Python or internet. `results/paper/index.html` shows the larger comparison. This is aggregate trace-driven simulation; Mininet and OpenFlow actuation are not part of the completed build.

For a new teammate or coding agent, read [docs/HANDOFF.md](docs/HANDOFF.md) for the current state, file map, results, verification commands and next work. This repository includes the validated reference checkpoints and saved campaigns. The original third-party datasets and local Python environment are deliberately excluded.

## What is implemented

- Exact eight-node, eleven-link GÉANT subgraph; 22 directed capacity constraints.
- Verified SNDlib traffic archive and ECON-PowerCI daily country carbon downloads.
- 3,972 matrices from the first 42 calendar days; six training-selected directed demands, chronological splits, gap-safe windows, training-only scaling.
- A 7,672-parameter GCN–GRU and a matched GRU, trained on CPU; persistence comparison and validation-calibrated forecast margins.
- Causal routing, shared reservations, propagation guards, hysteresis, deterministic max-min service, and explicit carbon attribution.
- **Multi-Seed Robustness Campaign:** 5 random seeds ($S=42..46$), 2,475 simulation runs, with paired Student's t-tests and Wilcoxon signed-rank tests (`scripts/run_multiseed.py`).
- **Parameter Sensitivity & Pareto Frontier:** 196-run sweep across safety quantiles $\alpha \in [0.70, 0.99]$ and hysteresis $\tau \in [0.00, 0.20]$, extracting 6 non-dominated Pareto configurations (`scripts/run_sensitivity.py`).
- **Joint Carbon-Budgeted Routing:** 64-state combinatorial routing selector subject to contemporaneous reactive reference carbon budgets (`docs/JOINT_SELECTOR.md`).
- Four main policies and three component comparisons. Five test days × three carbon years × three workloads × (seven delayed policies + four immediate policies) = **495 runs**.
- Self-contained HTML replay, CSV/JSON results, 4 publication-quality PNG/PDF figures, source/config/model hashes, 56 pytest tests, and an upgraded formal paper draft (`docs/PAPER_DRAFT.md`).

## Run on this laptop

PowerShell from the project folder:

```powershell
# Open the existing saved demo. No training, downloads or Python required.
.\RUN_DEMO.ps1

# Re-run the pilot from the prepared data and trained checkpoint.
.\RUN_DEMO.ps1 -Rebuild

# Produce a new dated folder with the full 495-run comparison.
.\RUN_PAPER.ps1
```

If Windows blocks a script, use the equivalent Python command below; no machine-wide execution-policy change is needed. Paths containing spaces are supported.

```powershell
.\.venv\Scripts\python.exe -m carbonroute doctor
.\.venv\Scripts\python.exe -m carbonroute demo --overwrite --open
.\.venv\Scripts\python.exe -m carbonroute run --paper --output results/my-experiment
```

The first measured pilot took about 1.3 seconds and the 495-run campaign about 47 seconds here, excluding setup/training/reporting. Times vary by CPU. The model uses two CPU threads; no GPU, VM, paid service, API key or account is needed.

## Set up another Windows laptop

1. Clone this repository or extract the ZIP produced by `python -m carbonroute bundle`. Both include trained models and both saved campaigns, so **the friend can immediately open `DEMO.html` and `results/paper/index.html`**.
2. For reproduction, install free **64-bit Python 3.12** from [python.org](https://www.python.org/downloads/windows/). Then run `SETUP.ps1` to install dependencies. Alternatively pass `-PythonExe 'C:\path\to\python.exe'`.
3. Run `RUN_DEMO.ps1 -Rebuild`. With a new source checkout, this downloads and prepares inputs, trains if checkpoints are absent, runs comparisons, and creates the report. With the transfer ZIP, the supplied checkpoint is reused after input preparation verifies compatibility.
4. Run `RUN_PAPER.ps1` for the larger campaign. Internet is required only for installing dependencies and the first input download.

Do not copy `.venv` between computers. Original third-party datasets are excluded from the bundle and Git; `RUN_DEMO.ps1 -Rebuild` downloads and verifies them when absent. A copied **HTML report alone** is already a portable offline presentation, including the animation and its data.

Manual setup, also usable on Linux/macOS (use `python` from a Python 3.12 venv):

```text
python -m venv .venv
# Windows interpreter: .venv\Scripts\python.exe
# Linux/macOS interpreter: .venv/bin/python
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps --no-build-isolation
python -m carbonroute prepare
python -m carbonroute train
python -m carbonroute demo --open
```

The exact lock was tested on Windows x64 / Python 3.12. Other operating systems have not been validated here. If using a shell without an activated venv, replace every `python` above with the venv interpreter's full path.

## Commands and outputs

| Command | Result |
|---|---|
| `python -m carbonroute prepare` | Checks/downloads original inputs; writes `inputs/processed/traffic.npz` and manifest |
| `python -m carbonroute train` | Trains GNN and GRU; saves checkpoints, validation margins and forecast metrics in `artifacts/` |
| `python -m carbonroute demo --overwrite` | Runs 14 paired pilot cases and generates `results/demo/index.html` |
| `python -m carbonroute run --paper --output results/new-run` | Runs 495 cases and generates the full report |
| `python -m carbonroute run --joint --split validation --paper --output results/joint-new` | Runs the 630-case carbon-budgeted development experiment using saved checkpoints |
| `python scripts/run_multiseed.py` | Runs 5-seed training & simulation campaign (2,475 runs) and computes t-test/Wilcoxon tests |
| `python scripts/run_sensitivity.py` | Runs 196-run parameter sensitivity grid sweep across alpha & tau, generates Pareto frontier & figures |
| `python -m carbonroute report --output results/new-run` | Rebuilds presentation from that campaign's frozen evidence |
| `python -m carbonroute bundle` | Writes a source + checkpoint + pilot ZIP under `dist/` |
| `python scripts/verify_results.py results/paper` | Independently audits saved routing, capacity, timing and carbon accounting |

Existing campaigns require `--overwrite` or a different output path. Use a new folder for research runs. Changing configuration deliberately invalidates prepared data/checkpoints: run `prepare` and `train` again before comparing. `--root` and `--config` are global options and precede the command.

Each result folder contains `FINDINGS.md`, `summary.csv`, `paired_differences.csv`, `manifest.json`, daily alignment files, per-bin `runs/*.json`, forecast evidence, `index.html`, and four figure sets. Every run log and date alignment has a recorded SHA-256 hash, verified again before report generation. The paper viewer embeds detailed traces for the first test day and first carbon year; its comparison table covers every run. All other traces remain in `runs/`.

## Current result, without cherry-picking

The first full run produced **mixed evidence**, not a blanket GNN win:

- Mean held-out demand MAE: GNN **0.4893 Mbps**, GRU **0.4834**, persistence **0.4858**.
- Congested, one-bin lag: proposed mean overload **114 min/day**, reactive carbon **126**, shortest path **165**. Proposed forwarding score **333.18**, reactive **323.83**; lower is better. Served fractions were **91.92%**, **91.89%**, and **91.22%**, respectively.
- Persistence with the same safeguards achieved **114 min/day** too. These results support investigating the safeguards, but do not establish the superiority of the GNN component.
- Immediate observations change the comparison; see the paper report. All original results, including unfavorable ones, are retained in the generated findings.

These numbers are provisional, from one training seed and five adjacent test days. The carbon score is **not grams saved** and is not a physical electricity measurement. Before writing a strong paper claim, expand held-out periods/seeds and examine the paired differences and sensitivity results. Do not tune on these test days and then call them unseen evaluation.

For the 20 October Review 1 explanation and source-backed improvement program, see [docs/REVIEW1_RESEARCH.md](docs/REVIEW1_RESEARCH.md). Its first proposed experiment, the joint carbon-budgeted selector, is now implemented; [docs/JOINT_SELECTOR.md](docs/JOINT_SELECTOR.md) gives the decision rule, validation results, honest limits, and commands. The [offline joint validation viewer](docs/reports/joint-validation.html) opens directly from a clone, while full audit logs are generated locally. The frozen `results/paper/` reference above is unchanged.

## Team ownership

| Team member | Main modules | What to understand for the viva |
|---|---|---|
| Ashutosh Gupta | `data.py`, `prediction.py`, `routing.py` | Chronological forecasting, graph mixing, residual margin, shared reservations |
| Tanay Mohta | `network.py`, `simulation.py`, `service.py` | Directed links, observation timing, offered versus served traffic, bottleneck sharing |
| Avaneesh Chavan | Carbon loader, `experiments.py`, `reporting.py` | Country/date alignment, carbon units, fair comparisons, uncertainty and limitations |

Use feature branches, one owner for shared configuration, and small reviewed merges. Run the tests before merging. Everyone works with the same contracts; see `docs/ARCHITECTURE.md`. The friend can run larger batches, but the current small CPU workload already runs on this laptop.

## Checks

```text
python -m pytest -q
python -m mypy src/carbonroute
python -m ruff check src tests scripts
python scripts/verify_results.py results/paper
```

Tests use small independent examples and require no downloads. Actual-result auditing requires a completed campaign. See `docs/METHODOLOGY.md` for equations and assumptions, `docs/DATA_SOURCES.md` for provenance, `docs/DEMO_GUIDE.md` for presentation, `docs/PAPER_DRAFT.md` for the current manuscript, and `docs/REVIEW.md` for the independent code review.
