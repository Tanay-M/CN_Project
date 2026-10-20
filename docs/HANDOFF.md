# Project hand-off — GÉANT carbon-aware routing

**Status (19 October 2026):** The agreed Windows-friendly, trace-driven research simulation is implemented and reproducible locally. A saved offline demo and a 495-run reference campaign are included in this repository. A joint carbon-budgeted route selector has also been implemented and evaluated on seven development validation days; see [JOINT_SELECTOR.md](JOINT_SELECTOR.md). Mininet/OpenFlow deployment and a test on the team's presentation laptop have **not** been completed. Start with [README.md](../README.md) for commands, [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) for the accepted scope, and [CODING_STANDARDS.md](../CODING_STANDARDS.md) before editing.

## What this project does

It studies whether a graph neural network (GCN–GRU) forecast of future origin–destination traffic, combined with a conservative validation margin and guarded carbon-aware path planner, can reduce simulated overload without unacceptable carbon-forwarding trade-offs. An eight-node, eleven-physical-link induced subgraph is taken from the real GÉANT topology. Real historical SNDlib traffic matrices provide demand; free ECON-PowerCI daily European country data provide carbon-intensity scenarios. A deterministic aggregate flow simulator compares identical demands across shortest path, reactive carbon, predictive traffic engineering, the proposed GNN policy, persistence, and two safeguard ablations. The outputs are **simulation results**, not packet-level measurements or measured electricity savings.

Traffic is from 2005 and carbon from 2022–2024. Their calendar/weekday matching constructs scenarios; it does not imply simultaneous real-world observations. Country intensity stays constant across each day. Capacities and propagation times are declared assumptions. No paid API, GPU, Linux VM, or live carbon feed is required. Dataset provenance and citation notes are in [DATA_SOURCES.md](DATA_SOURCES.md); the exact experimental assumptions and equations are in [METHODOLOGY.md](METHODOLOGY.md).

## What is done and where to look

| Area | Main files | Role |
|---|---|---|
| Entry points and setup | `README.md`, `SETUP.ps1`, `RUN_DEMO.ps1`, `RUN_PAPER.ps1`, `src/carbonroute/cli.py` | Windows setup, CLI commands, saved demo and experiment execution |
| Input and graph | `configs/default.json`, `src/carbonroute/data.py`, `network.py`, `common.py` | Verified downloads, chronological preparation, fixed node/OD order, directed graph and units |
| Learning | `src/carbonroute/prediction.py`, `artifacts/gnn.pt`, `artifacts/gru.pt`, companion JSON | CPU GCN–GRU, matched GRU, persistence, validation margins, frozen reference checkpoints |
| Routing and replay | `src/carbonroute/routing.py`, `service.py`, `simulation.py` | Candidate routes, capacity and delay guards, causal decisions, max-min served flows and carbon accounting |
| Campaign and presentation | `src/carbonroute/experiments.py`, `reporting.py`, `report.html`, `results/demo/`, `results/paper/` | Paired policy cases, integrity-checked logs, self-contained HTML, CSV and figures |
| Independent audit | `scripts/verify_results.py`, `tests/` | Recalculates run invariants and tests causality, routing, service, data and forecasting |
| Research writing | `docs/PAPER_DRAFT.md`, `results/paper/FINDINGS.md`, `docs/REVIEW.md`, `docs/DEMO_GUIDE.md` | Manuscript draft, generated findings, completed code review and presentation script |

`routing.py` now also contains `BudgetPlanner`, which enumerates the 64 joint route assignments and keeps a separate reactive reference state. `experiments.py` accepts validation/test splits and `--joint`; `simulation.py` logs the predicted budget and its compliance per bin. The complete development experiment, commands and limitations are in [JOINT_SELECTOR.md](JOINT_SELECTOR.md). A [standalone validation viewer](reports/joint-validation.html) and [exploratory test viewer](reports/joint-test-exploratory.html) are committed for offline presentation. Local full logs under `results/joint-*` are ignored by Git and can be regenerated from the saved checkpoints and verified input download. Neither the reference campaign nor the neural checkpoints were modified.

The detailed module inputs/outputs and replay sequence are in [ARCHITECTURE.md](ARCHITECTURE.md). Its crucial invariant is that prediction sees only historical OD matrices, routing commits before target demand is revealed, and every policy receives the same workload, graph, route candidates and allocator. The source topology's static demand section is not used as training time series. Do not fit pair selection, scaling or margins on the test split. Rates are decimal Mbps; a bin is 900 seconds; carbon intensities are numerically g CO₂/kWh. The reported forwarding score sums sender-country intensity across hops, weighted by **served** bytes; it is neither grams emitted nor a measured energy figure.

## Reference evidence and honest result

The checked-in `results/paper/` campaign has 495 runs: five adjacent held-out days × three carbon-year pairings × three workload levels × eleven policy/timing cases. `results/demo/` has 14 pilot cases. The reference manifests bind each run and country/date alignment to hashes as well as the source/config/model hashes; `scripts/verify_results.py` independently audits 47,520 paper run-bins. `results/paper/index.html` and `results/demo/index.html` are standalone offline viewers; `DEMO.html` opens the demo from the repository root. `summary.csv` has every run, `paired_differences.csv` has paired comparisons, `runs/*.json` holds detailed traces, and `figures/` has shareable plots. Treat the committed campaign as a frozen reference; write new campaigns to a new `results/<name>` folder.

Mean held-out three-horizon demand MAE is **0.4893 Mbps GNN**, **0.4834 GRU**, and **0.4858 persistence**. The GNN does **not** outperform either forecasting baseline on this metric. Under congested workload with a one-bin observation lag, the proposed policy averages **114 overload minutes/day**, versus **126 reactive carbon** and **165 shortest path**. Its forwarding score is **333.18** versus **323.83 reactive carbon** (lower is better), so the overload gain comes with a worse carbon score in that comparison. Persistence with the same safeguards also reaches 114 overload minutes/day. Served fraction is 91.92% proposed versus 91.89% reactive. The complete scenarios, immediate-observation rows, uncertainty and limitations are in [FINDINGS.md](../results/paper/FINDINGS.md) and [PAPER_DRAFT.md](PAPER_DRAFT.md). Do not claim GNN superiority, physical CO₂ savings, measured latency, or publication-ready novelty from these observations.

## Reproduce and edit safely

On Windows, clone the repository, open `DEMO.html` immediately, then install free 64-bit Python 3.12 and run `SETUP.ps1` to install dependencies. Run `RUN_DEMO.ps1 -Rebuild` to download the original data from free primary sources, verify checksums, prepare inputs, check the supplied checkpoints against them and replay the pilot; no API key is needed. Run `RUN_PAPER.ps1` for a new 495-run campaign. Details and equivalent Python commands are in [README.md](../README.md). `inputs/raw/` and `inputs/processed/` are intentionally not redistributed. `dist/` contains local transfer bundles and is not committed; `python -m carbonroute bundle` recreates one. Do not copy `.venv` between machines.

Before merging changes, run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m mypy src/carbonroute
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe scripts/verify_results.py results/paper
```

The last command audits the *frozen reference* and is meaningful only while its source/config/model context remains intact. If you alter parsing, configuration, model, policy, accounting or report schema, regenerate the affected prepared data, checkpoints and reference campaigns deliberately. Do not edit a saved run or alignment JSON in place: hash validation will (correctly) reject it. Use feature branches and review any change that affects the common experiment contract.

## Most useful next work

1. On the teammate's Windows presentation laptop, clone, open `DEMO.html`, run setup and `RUN_DEMO.ps1 -Rebuild`, then record environment and timing. The current transfer test was only on a second local folder.
2. For a stronger paper, repeat with predeclared extra held-out periods and training seeds, retain negative results, and report paired uncertainty. The existing five adjacent test days and one seed limit generalization. Do not choose periods after seeing policy outcomes and then call them unseen tests.
3. Freeze the new [joint selector](JOINT_SELECTOR.md) and evaluate it on a predeclared later, untouched traffic period before making a paper claim. The currently inspected validation and original test periods cannot provide a fresh independent confirmation. Avoid a generic “first predictive carbon routing” claim. The manuscript is a draft, not a submitted paper.
4. Add Mininet/OpenFlow only as a separately labeled extension if a faculty requirement makes packet-level demonstration necessary. No current result should be relabeled as Mininet output.

## Suggested skills for a coding agent

- Use **`implement`** for changes against the accepted plan and module contracts.
- Use **`diagnosing-bugs`** when an input, model compatibility check, replay or report fails.
- Use **`code-review`** before merging a substantial methodology or experiment change; the previous two-axis review is recorded in [REVIEW.md](REVIEW.md).
- Use **`research`** when checking dataset terms, related work, or scientific claims against primary sources.

The local Git history before this hand-off is `c05e3d9` (scope), `9af9d2b` (simulation), and `25e3928` (evidence and draft). The report manifests intentionally refer to the source revision used to generate the frozen campaigns; documentation and packaging commits afterward do not invalidate those source hashes.
