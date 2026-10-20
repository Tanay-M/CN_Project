# Module contracts and team integration

All rates are decimal Mbps. One scenario bin is 900 seconds. The fixed node order is FR, DE, BE, NL, CH, IT, AT, LU. Graph identity includes source link IDs and assumed capacities/delays.

```text
SNDlib archive -> data.prepare -> masked/scaled chronological OD traffic
                                     |
                              prediction.train
                                     |
                        checkpoint + validation margin
                                     |
ECON daily CSV -> country/date pairing -> experiments.campaign
                                             |
                                  simulation.replay
                               history -> predictor
                               forecast -> planner
                               commit routes
                               reveal actual demand
                               allocate served rates
                               account and log
                                             |
                                   reporting.report
```

| Interface | Input | Output / invariant |
|---|---|---|
| `data.prepare(root, config)` | Checked native files + JSON config | `traffic.npz` and manifest; no hidden test-fitted statistics |
| `data.load_prepared(...)` | Matching files/config | `Prepared`: timestamps, `[T,8,8]` rates, mask/scaler, graph, ordered OD pairs |
| `Predictor.predict(history)` | Exactly 12 observed `[8,8]` matrices | Nonnegative `[3,8,8]` Mbps; inactive pairs zero |
| `Predictor.margin` | Validation residual calibration | `[3,8,8]` additive Mbps; inactive pairs zero |
| `Planner.decide(forecast, current_carbon, step)` | Conservative forecast, eight current CI values | Immutable `Decision`: ordered routes, changes, infeasibility, reasons |
| `BudgetPlanner.decide(forecast, current_carbon, step)` | Conservative forecast, eight current CI values, reactive reference | Immutable `Decision`: joint 64-state combinatorial selection, min-peak utilization, carbon budget compliant |
| `progressive_fill(demand, incidence, capacities)` | Flow Mbps, link-by-flow binary matrix, directed capacities | Offered-bounded, capacity-feasible served Mbps |
| `replay(...)` | Frozen scenario and predictor | `ReplayResult` containing per-bin records and aggregate metrics |
| `report(root, result_dir)` | Completed campaign with checked frozen evidence, including each run and carbon date mapping | Standalone offline HTML, findings, PNG/PDF figures |

No network service or cross-process API is needed. JSON is used for public metadata; numeric arrays use NPZ without pickle. Checkpoints contain only PyTorch state dictionaries and load with `weights_only=True`. Raw third-party files, prepared inputs, environments and scratch outputs are ignored by Git. The validated reference checkpoints and demo/paper campaigns are committed. `.gitattributes` preserves their exact bytes so manifest hashes survive a checkout.

Keep neural prediction independent of routing policy: it predicts offered OD demand, never policy-dependent link utilization. Keep the replay driver as the only component that can reveal the target demand. Do not pass full scenario arrays into `predict` or `decide`.

The small dense graph avoids PyTorch Geometric/CUDA dependencies. The no-graph GRU uses identity adjacency with the same architecture/parameter count; only the neighbor information is removed.

For handoff, clone the repository or use a generated transfer bundle; see `docs/HANDOFF.md`. Do not merge someone else's `.venv`, temporary inputs or source data into Git. Reprepare/retrain after configuration changes. A cached checkpoint mismatch is an error, not permission to silently use the wrong model.
