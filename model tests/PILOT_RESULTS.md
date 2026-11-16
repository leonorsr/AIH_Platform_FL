# Initial pilots

Eight training runs were completed: four architectures with both unweighted cross-entropy and inverse-frequency class weighting. They used one seed (2024), up to five epochs, and checkpoint selection by development S/V PR-AUC.

Both experiments used the same uniform sample, verified by manifest hashes: 8,192 training beats and 2,048 development beats. Training counts: N=6,314, S=9, V=95, Q=1,774. Development counts: N=1,494, S=17, V=0, Q=537. The absence of development V examples prevents evaluating that class in these pilots; S is the only positive class represented in the grouped S/V metric.

| Experiment | Highest-ranked pilot architecture | Dev S/V PR-AUC | Dev S/V ROC-AUC | S/V miss rate | S/V FAR |
| --- | --- | ---: | ---: | ---: | ---: |
| Unweighted loss | MLP | 0.012209 | 0.647001 | 100% | 0% |
| Inverse-frequency weighting | ResNet 1D | 0.012987 | 0.687520 | 100% | 2.61% |

Miss rate and FAR use a fixed threshold of 0.5. Neither top-ranked configuration detected S/V at that threshold. Areas assess score ordering and do not guarantee a useful operating point. These pilots do not establish a production-ready model or a winner on the full dataset.

Detailed reports, checkpoints, and histories are stored in:

- `results/20261115-172405-pilot-fd21bd/`
- `results/20261115-172702-pilot-balanced-07b46d/`

These are the initial pilots. The current comparison uses bounded smoke tests in `configs/smoke.json`, with results in `ModelTests_results.ipynb`. The held-out test set is not used. Architecture performance under FedAvg will be evaluated separately.
