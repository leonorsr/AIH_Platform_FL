# Neural-network smoke tests in C#

A short comparison of four MLP configurations, a small CNN, a deep CNN, ResNet 1D, and four MobileNetV1 1D widths. The objective is to identify candidates for the FL system. Selection uses grouped S/V PR-AUC on a development sample. This workflow does not run full-data training or evaluate the held-out test set.

## Run

From the repository root:

```sh
dotnet build "model tests/ModelTests.csproj" -c Release
dotnet run --project "model tests" -c Release -- self-test
dotnet run --project "model tests" -c Release -- run
```

`run` defaults to `configs/smoke.json`:

- Eleven configurations, three seeds (2024/2025/2026), and at most three epochs: 33 short runs.
- Up to 128 examples per class/patient for training and 256 for development, without replacement. All available examples are retained when a class has fewer than its quota.
- At most 8,192 training and 2,048 development examples; actual totals can be lower because absent classes are not synthesized. The current sample contains 5,547 training and 1,294 development beats.
- The sample is fixed and shared by all configurations/seeds. Training and development patients remain separate.
- Adam, learning rate 0.001, batch size 128, and inverse-frequency cross-entropy weights calculated only from the training sample.
- CPU for the MLP/CNN/ResNet configurations and MPS for MobileNetV1 on this Mac. Small samples are cached in memory; no new full-dataset cache is built.

Stratification includes rare classes but changes sample prevalence. PR-AUC, precision, and FAR describe this selected sample, not validated full-population performance. Results are intended for architecture screening.

## Results and chart

**`ModelTests_results.ipynb`** is generated in C# with a .NET (C#) kernel and an embedded ScottPlot chart. It contains only the partition explanation, results table, and one PR-AUC bar chart. Saved output is visible without running notebook cells.

Each experiment saves configuration, sample manifests and hashes, class support, per-epoch histories, per-seed results, and checkpoints in `results/`. The ranking uses mean S/V PR-AUC across the three seeds and reports population standard deviation. Each seed contributes its best-development-PR-AUC checkpoint from the bounded training run. The seeds column identifies incomplete comparisons.

To regenerate the notebook without training:

```sh
dotnet run --project "model tests" -c Release -- report --run "model tests/results/RUN_DIRECTORY"
```

The notebook is regenerated after each epoch and completed candidate. Reopen it to see updates. The chart switches to horizontal bars when needed to keep configuration names readable.

## Metrics

- PR-AUC: trapezoidal integration. Average Precision is reported separately in JSON.
- ROC-AUC: calculated from continuous probabilities, processing tied scores together.
- Miss rate: `FN/(TP+FN)`.
- FAR: `FP/(FP+TN)`, a false-positive proportion, not false alarms per hour.
- Grouped S/V: score `p(S)+p(V)` and a fixed threshold of 0.5 for miss rate/FAR. N/Q form the remaining group; Q means unclassifiable, not normal.
- Per-class evaluation: one-vs-rest AUC from scores and argmax decisions for rates. Undefined metrics are `null`, with support reported.

All configurations receive the same data and epoch budget. This is centralized architecture screening; FedAvg performance will be evaluated separately.

## MLP configurations

| MLP | Hidden layers | Input → output |
| --- | --- | --- |
| Simple | 64 | 500 → 4 |
| Medium (existing MLP) | 128 → 64 | 500 → 4 |
| Complex | 256 → 128 → 64 | 500 → 4 |
| Extra-complex | 512 → 256 → 128 → 64 | 500 → 4 |

All MLPs use ReLU and dropout 0.2 after the first hidden layer. The medium architecture is unchanged, allowing its existing results to be reused.

## MobileNetV1 widths

The model adapts [MobileNetV1 (Howard et al., 2017)](https://arxiv.org/abs/1704.04861) to ECG 1D, one input channel, and four classes. Alpha values **0.25 / 0.50 / 0.75 / 1.00** change channel widths while retaining the same 13 depthwise/pointwise blocks, strides, BatchNorm, ReLU, and global average pooling. All variants are trained from scratch without image weights. These are 1D adaptations, not MobileNetV2 models.

The alpha=0.25 variant retains exactly 206,468 trainable parameters. Future FedAvg integration must define a policy for BatchNorm buffers in addition to trainable parameters. CPU/MPS differences may prevent bitwise numerical equivalence; backend assignments are recorded.

To repeat only the alpha=0.25 smoke tests:

```sh
dotnet run --project "model tests" -c Release -- run --config "model tests/configs/smoke-mobilenetv1.json"
```

## Reusing existing results

The expanded comparison reuses the 15 results for the five existing configurations. It adds only **18 new short runs**: three new MLP configurations and three new MobileNet widths, each with three seeds and at most three epochs. Manifest hashes verify identical samples. Provenance is stored in `comparison_sources.json` in the run directory. The held-out test set remains unused.

To run only the additional configurations:

```sh
dotnet run --project "model tests" -c Release -- run --config "model tests/configs/smoke-additional.json"
```

## Interrupted full-data run

The full-data run in `results/full-20261115-181126/` was stopped when the scope was corrected to smoke tests. Existing results were preserved and the test set was not evaluated. The automatic full-data launcher and its configuration were removed. `run` rejects configurations without both training and development sample limits.
