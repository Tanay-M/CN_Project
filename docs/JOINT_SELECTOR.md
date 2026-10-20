# Joint carbon-budgeted routing experiment

This implements the first follow-up experiment proposed in [REVIEW1_RESEARCH.md](REVIEW1_RESEARCH.md): choose all six OD routes together, minimizing the worst **predicted** directed-link utilization while keeping predicted offered-traffic carbon cost at or below a contemporaneous reactive carbon-routing reference. It is a trace-driven simulation, with no Mininet or live control plane.

## Decision rule

At each 15-minute bin, `BudgetPlanner` receives a three-step demand forecast and the currently known country carbon intensities. A separate reactive planner sees the same delayed observation history as the original baseline. For each candidate route assignment, the joint selector computes:

```text
predicted peak = max over forecast horizons and directed links
                 (sum of predicted OD demand using that link / link capacity)
predicted carbon cost = sum over OD pairs
                        (mean predicted demand × sum of sender-node CI along its route)
carbon budget = predicted carbon cost of the reactive assignment × (1 + budget fraction)
```

Both the candidate and reference carbon costs use **the same joint-policy forecast**, so the reference is a route assignment, not a peek at realized traffic or at the reactive policy's eventual observed score. The six active pairs each have two delay-bounded paths, producing 64 assignments. The selector chooses the lowest predicted peak among assignments inside the budget, breaking ties by carbon cost, route changes, delay, and stable index. The original two-bin hold and 5% improvement threshold suppress small changes. **One intentional priority change:** predicted capacity risk or a predicted carbon-budget breach can override a hold and the threshold, because otherwise the supposedly hard budget could fail when the reactive reference becomes cheaper. This can increase route churn. The reactive assignment is always in the search space, so a nonnegative budget remains attainable. The result auditor verifies zero budget breaches.

The cost above is a planning proxy in Mbps × summed g CO₂/kWh. The reported forwarding score is calculated **after** revealing demand, allocating served rates, and weighting by served bytes. A predicted budget is therefore not a guarantee on the observed forwarding score, and neither quantity is measured CO₂ emissions. The utilization objective is also predictive; actual overload can remain or worsen if forecasts miss bursts.

`joint_gnn`, `joint_gru`, and `joint_persistence` use the same selector, margin-calibration procedure, candidate paths, observation lag, carbon budget, and tie rules. Each uses its own already saved predictor and corresponding validation margin. The campaign also retains all seven original delayed policies for paired comparison. This experiment does **not** call `train` or update any checkpoint.

## Reproduce on Windows

From the repository root after `SETUP.ps1` and `python -m carbonroute prepare` (or with the existing prepared data):

```powershell
.\.venv\Scripts\python.exe -m carbonroute run --joint --split validation --paper --output results/joint-validation-new
.\.venv\Scripts\python.exe scripts/verify_results.py results/joint-validation-new
```

`--paper` runs seven complete validation days × three carbon years × three workloads × ten policies = 630 paired runs (60,480 intervals). Without `--paper`, the command runs a one-day, one-year pilot. The default `--carbon-budget-fraction 0` permits no predicted excess over the reactive assignment. A positive fraction, such as `0.05`, permits 5% predicted excess and must be reported as a separate, predeclared experiment. Outputs include the offline `index.html`, full `runs/*.json`, `summary.csv`, `paired_differences.csv`, and source/input/GNN/GRU/persistence hashes in `manifest.json`. New `results/joint-*` folders are local and ignored by Git; the frozen `results/paper` reference is untouched.

For a presentation on another laptop, open [the standalone validation viewer](reports/joint-validation.html) or [the exploratory test viewer](reports/joint-test-exploratory.html) from a clone of this repository. They embed aggregate results and first-day replays. The full per-bin audit logs remain in the local `results/joint-*` folders and can be regenerated with the commands above.

## Development validation result (19 October 2026)

On seven validation days, the zero-budget `joint_gnn` policy had zero predicted-budget violations across 2,016 scenario/carbon intervals per workload. The independent audit passed all 630 runs and 60,480 intervals.

| Workload | Joint GNN forwarding score | Reactive score | Joint overload min/day | Reactive overload min/day | Joint served | Reactive served |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 267.89 | 272.19 | 19.29 | 19.29 | 99.6% | 99.6% |
| Congested | 284.39 | 300.22 | 62.14 | 62.14 | 98.3% | 98.3% |
| Bursty | 277.26 | 287.42 | 105.00 | 105.00 | 97.5% | 97.5% |

The paired score differences versus reactive are −4.30 (normal), −15.83 (congested), and −10.16 (bursty); lower is better. Joint GRU and joint persistence are almost identical to joint GNN on these outcomes, so this result supports the **joint selector**, not a special gain from graph learning. The old `predictive_carbon` policy lowers overload slightly in normal/congested validation but raises the carbon score; the joint selector largely trades that overload benefit away for its carbon constraint. Route changes also increase in congested and bursty cases. Full numbers and descriptive day-cluster bootstrap intervals are in the locally generated `results/joint-validation-final/`.

These are **development results**: the validation period was already used for checkpoint selection and forecast-margin calibration, and the existing five test days were previously inspected during earlier work. The seven adjacent days and one model seed do not support a general performance claim. The scientific next step is to freeze this policy and evaluate on a predeclared later untouched traffic period, with all three predictors and baselines, before claiming a paper result. No repeat of the already viewed test days should be called a fresh independent test.

## Exploratory replay on the previously inspected test days

For a practical robustness check, the same zero-budget policy was replayed on the original five test days (450 runs, 43,200 intervals; independent audit passed; zero predicted-budget violations). It was **not** a fresh test of a predeclared policy, since earlier results from these days informed the joint-selector idea.

| Workload | Joint GNN forwarding score | Reactive score | Joint overload min/day | Reactive overload min/day |
|---|---:|---:|---:|---:|
| Normal | 291.87 | 302.93 | 72 | 72 |
| Congested | 303.45 | 323.83 | 126 | 126 |
| Bursty | 298.16 | 316.72 | 135 | 135 |

The old greedy GNN policy reaches 114 rather than 126 overload minutes on congested test traffic, but its forwarding score is 333.18 rather than 303.45 for the joint selector. Thus the new method improves the carbon proxy versus reactive without an overload improvement; the old method has a different trade-off. GRU and persistence again closely match the GNN. Full local evidence is in `results/joint-test-exploratory-final/`; regenerate it with `--joint --split test --paper` and a new output path.
