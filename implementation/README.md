# MobileNetV1 1D with FedAvg

Executable C#/.NET 8 local simulation. MobileNetV1 1D alpha **1.00** had the highest mean S/V PR-AUC in the smoke tests (0.4992 ± 0.0336). This chooses the architecture for this experiment; full-data federated performance has not been measured.

The current schedule is **100 local epochs × 15 global rounds × 3 seeds**. Use the new output directory below; the stopped 100-local/25-global run retains its original saved configuration and must not be resumed for this schedule.

## Commands

Run from the repository root with the .NET 8 SDK installed:

```sh
# Build once.
dotnet build implementation/Implementation.csproj -c Release

# Check configuration and aggregation without training.
dotnet run --project implementation -c Release --no-build -- self-test

# Very short end-to-end test: synthetic ECG only; no real patient data.
dotnet run --project implementation -c Release --no-build -- integration-test

# Inspect the full real-data experiment without starting training.
dotnet run --project implementation -c Release --no-build -- plan

# Start the full real-data pipeline: FedAvg, local tuning, then held-out evaluation.
dotnet run --project implementation -c Release --no-build -- pipeline \
  --config implementation/configs/fedavg.json \
  --out artifacts/fl/mobilenetv1-fedavg-100local-15global

# Resume the full pipeline, including tuning/evaluation, after an interruption.
dotnet run --project implementation -c Release --no-build -- pipeline \
  --out artifacts/fl/mobilenetv1-fedavg-100local-15global
```

The full run uses 100 local epochs per round, 15 rounds, and three seeds. **Only the integration test uses tiny settings.** Do not pass `--config` when resuming an existing pipeline. Open [Implementation_results.ipynb](Implementation_results.ipynb) using the .NET (C#) kernel, set `experimentDirectory` to your run directory, then Run All to refresh the report.

## Very short integration test

`integration-test` creates a separate synthetic 16/2/2 dataset with four beats per client (one per class). It executes the actual MobileNetV1/FedAvg code with three seeds, one global round, one local epoch, and two fine-tuning candidates (head-only and all layers; learning rate 0.0001; one epoch each). It covers NPZ/index reading, disk caches, optimizer/backpropagation, weighted aggregation, global/local checkpoint persistence, development selection, synthetic test inference, all metric reports, and a second complete pipeline invocation that must preserve existing checkpoint hashes.

No real training/development/test indexes or signals are opened. Synthetic scores are functional checks, not ECG performance evidence. Production settings in `configs/fedavg.json` are unchanged. Default outputs are under `artifacts/fl/integration-smoke/<timestamp>/`; `verification.json` records pass/fail and runtime, and `results/` has the same report layout as a full experiment. To choose a location, append `--out artifacts/fl/my-integration-test` (the directory must be empty). Generated model checkpoints require roughly 1 GB. The results notebook can inspect the synthetic run by pointing `experimentDirectory` at its `results/` directory.

The integration check passed on 2026-11-16 in **77.3 seconds** on this machine: 48 local checkpoints, 48 completed fine-tuning searches, 204 metric reports, and unchanged checkpoint hashes on resume. These are functional validation results, not model-performance results.

## Experiment

| Setting | Value |
| --- | --- |
| Input / output | Normalized ECG beat `[batch, 1, 500]`; logits for N, S, V, Q |
| Model | MobileNetV1 1D alpha 1.00; 3,180,548 trainable parameters; initialized from scratch per seed |
| Training clients | All 16, `client_00`–`client_15`, every round |
| Development clients | `client_16`, `client_17`; full data, evaluation after every round |
| Test clients | `client_18`, `client_19`; accessed only after all fine-tuning selections are locked, by `evaluate-test` or the final `pipeline` stage |
| Local epochs | **100 per client per global round**, no early stopping |
| Global rounds | **15** per seed |
| Seeds | **2024, 2025, 2026**, independent initializations and experiments |
| Training data | All prepared training examples, natural prevalence, shuffled without replacement each epoch |
| Optimizer | Adam, learning rate 0.001; new optimizer for every client/round |
| Batch size / device | 128 / CPU, four Torch threads; clients execute sequentially |
| Loss | Cross-entropy; fixed inverse-frequency weights computed from all training clients only |
| Aggregation | FedAvg weighted by each client's distinct example count, independent of local epochs |
| Checkpoint selection | Highest pooled development S/V PR-AUC, separately per seed; earliest round on ties |
| Local checkpoints | All 16 final local updates are saved before aggregation in every round |
| Fine-tuning | Per final local model: 3 learning rates × head-only/all layers, up to 20 epochs each; unchanged baseline included |

The requested schedule totals **1,500 local epochs per client per seed**, 720 client training jobs, and 72,000 local epochs across clients and seeds. There is no sample cap or early stopping in this experiment. Batch size, optimizer, learning rate, class weighting, and threshold are explicit in [configs/fedavg.json](configs/fedavg.json).

BatchNorm affine parameters, running means, and running variances use the same sample-weighted average as other floating tensors. Integer `num_batches_tracked` buffers take the maximum across participating clients. This is a defined global-buffer averaging policy, not FedBN or exact recomputation of pooled activation statistics. Floating sums use float64 accumulators and return to float32. All 16 updates must pass identity, round, count, schema, and finite-value validation before the global checkpoint advances. Optimizer states are never averaged.

## Prepare and check without training

Run from the repository root:

```sh
dotnet build implementation/Implementation.csproj -c Release
dotnet run --project implementation -c Release --no-build -- self-test
dotnet run --project implementation -c Release --no-build -- plan
```

`self-test` uses synthetic tensors, verifies weighted aggregation and rejection paths, checks BatchNorm handling, and saves/restores an untrained model. It never executes an optimizer or backpropagation. `plan` validates training/development indexes, records counts and index hashes in its console output, and displays the workload without reading signals, building caches, or starting training. Running with no command prints help.

## Start only when ready

**Training starts only with an explicit `run`, `resume`, `fine-tune`, or `pipeline` command.** The following commands are provided for later use:

```sh
dotnet run --project implementation -c Release --no-build -- run \
  --config implementation/configs/fedavg.json \
  --out artifacts/fl/mobilenetv1-fedavg-100local-15global

dotnet run --project implementation -c Release --no-build -- resume \
  --out artifacts/fl/mobilenetv1-fedavg-100local-15global
```

For the complete experiment (FedAvg → global/local development reports → local fine-tuning search → locked test evaluations), use this command **only when ready to start training**:

```sh
dotnet run --project implementation -c Release --no-build -- pipeline \
  --out artifacts/fl/mobilenetv1-fedavg-100local-15global
```

Repeating `pipeline` with the same output resumes committed rounds, completed client searches, and saved evaluations. `run`/`resume` stop after FedAvg and development reports. `fine-tune --out ...` runs development-only selection; `evaluate-test --out ...` performs inference after all searches are locked.

Use an empty output directory for a new run. The process holds an exclusive run lock. Configuration, patient partition, index hashes, class supports/weights, environment, and seed schedule are recorded. Resume rejects changed configuration, partition, or training/development indexes. Prepared signals are assumed immutable; index hashes do not hash every NPZ payload.

Signals are read in batches, with only one client's metadata held during local training. The default disk cache stores 2,000 bytes per beat in `implementation/cache/`, reuses the same signals across seeds, and checks cached vectors against source NPZ samples. The first training execution builds caches as clients are encountered; planning never does. Disable `UseDiskCache` to read NPZ files directly. Caches and experiment outputs are excluded from Git.

`status.json` identifies the active seed, round, client, epoch, and batch progress. Ctrl+C requests cancellation at a batch/client boundary; an ongoing cache build or evaluation batch must return first. A round is published atomically only after all clients and development evaluation complete. Interrupted rounds restart in full from the previous committed global checkpoint; completed seeds/rounds are reused. No local optimizer continuation within an interrupted round is attempted.

## Results and held-out evaluation

Open [Implementation_results.ipynb](Implementation_results.ipynb) with the **.NET (C#)** kernel to inspect saved results. Set `experimentDirectory` if your output directory differs from the default, then run all cells to refresh. The notebook displays per-seed metrics, mean and population SD, development curves, per-class/patient metrics, confusion matrices, complete round history, and local client loss/timing. It handles missing and partially completed results and never launches training or test evaluation.

Each seed stores `round-NNN/global.bin`, `round-NNN/locals/client_XX.bin`, `metrics.json`, and `rounds.csv`. Retaining all 720 local model checkpoints uses roughly 9.6 GB before caches and other artifacts. Local checkpoints correspond to the final local update before aggregation; the global checkpoint is the weighted aggregate. Metrics include pooled and per-client development results, per-class PR-AUC/ROC-AUC, confusion matrices, accuracy, macro-F1, and grouped S/V PR-AUC, ROC-AUC, miss rate, and FAR. S/V scores are `p(S)+p(V)`, with N/Q as negatives and threshold 0.5 for miss/FAR. PR-AUC uses trapezoidal integration; Average Precision is separate. Undefined metrics are null. The full development prevalence is retained; these scores are not directly comparable to the stratified smoke-test scores.

## Local fine-tuning

The existing patient split remains unchanged. Each of the 16 local models trains only on its own original training patient. **Both existing dev patients evaluate every local model**, and both existing test patients evaluate the locked configurations. These metrics measure transfer to unseen patients; there is no within-client held-out split or training on dev/test patients.

The search starts from each client's final round-15 local checkpoint, independently for each seed. Defaults in `FineTuning` are learning rates `[0.001, 0.0001, 0.00001]`, trainable layers `[head, all]`, and 20 maximum epochs. Every candidate starts with identical source weights, a new Adam optimizer, and the same deterministic epoch shuffles. Head-only tuning freezes feature weights and BatchNorm statistics. All-layer tuning updates the whole model and its BatchNorm statistics. Loss weights remain the recorded training-only FedAvg weights.

Development evaluation occurs after each fine-tuning epoch. Selection maximizes **the local model's pooled dev S/V PR-AUC**, choosing learning rate, layer scope, and epoch. The unchanged local checkpoint is also a candidate; ties keep it, then the earliest candidate/epoch in deterministic grid order. Search completes this bounded grid; it does not guarantee improvement or search indefinitely. Default search is 288 candidate fits across clients/seeds, with at most 5,760 additional local epochs. To run only this stage after FedAvg:

```sh
dotnet run --project implementation -c Release --no-build -- fine-tune \
  --out artifacts/fl/mobilenetv1-fedavg-100local-15global
```

Each completed search publishes `seed-*/fine-tuning/client_*/result.json` and `best.bin` atomically, with candidate configurations, per-epoch metrics, baseline metrics, and selected weights. Resume reuses completed searches and restarts an interrupted client's search. A saved search plan and source-checkpoint hashes reject changed settings or source models.

## Global, local, and test reports

After all three FedAvg seeds complete, `summary.json` records the development-selected global checkpoints and their mean ± population SD. `model-reports/` contains each final global model, development-selected global model, final local model, and selected fine-tuned local model. Every report includes pooled and per-patient metrics; local model scores are measured on the same shared evaluation patients, so the 16 models are not independent evaluation samples. The notebook reports all configurations, paired before/after metric changes, seed summaries, and PR-AUC comparison charts.

Seeds are not selected by performance. No test score selects architecture, round, fine-tuning configuration, epoch, threshold, or seed. Test evaluation requires all 48 local searches to finish and lock their selected checkpoints. It performs **inference only**, without further fitting or gradient updates. To run that stage explicitly:

```sh
dotnet run --project implementation -c Release --no-build -- evaluate-test \
  --out artifacts/fl/mobilenetv1-fedavg-100local-15global
```

This evaluates 34 model entries per seed: final global, development-selected global, 16 final local, and 16 fine-tuned local models. Results are saved individually in `model-reports/test-*.json`; the selected global result is also retained in each seed's `test.json` for compatibility. Saved model evaluations are skipped on repeat invocation. Test results never trigger further configuration search. `test_status.json` records completion.

Training components, signal readers, architecture, and metrics are reused from the existing C# projects. The smoke-test CLI and notebook are not modified by this application. This is an in-process sequential federated simulation; distributed transport is not implemented.
