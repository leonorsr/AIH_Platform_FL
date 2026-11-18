# C# federated ECG pipeline

MobileNetV1 1D alpha 1.00, FedAvg, **three randomly selected clients per round from a pool of 16**, **100 local epochs per round, 15 global rounds, and three seeds** (2024–2026). Training remains stopped until an explicit training command is issued.

## CPU improvements and experiment scope

- Mathematically equivalent vectorized depthwise convolutions and matrix-multiplication pointwise convolutions replace the slow CPU Conv1d paths. Parameter shapes, names, architecture, and checkpoint schema are retained. Numerical output and input/weight-gradient checks cover both implementations.
- On this machine, synthetic batch-128/four-thread training improved from **14.6 to 877.7 examples/s** (about **60×**); evaluation improved from 71.7 to 3,514.6 examples/s. These measurements exclude data loading, metrics, checkpoints, and pauses.
- The default protocol uses the original **3,904,222-beat training pool**, with all data of the **three selected training clients** used in each round. One randomly selected original dev patient supplies development data per round; the other remains eligible in later rounds. `TrainSamplesPerClient` and `DevSamplesPerClient` are both `null`: no training/development sampling caps apply. The production pipeline reads the original prepared ECG signals; synthetic fixtures belong only to the separate optional software integration test.
- FedAvg weights and inverse-frequency loss weights use the **selected training counts**, not the full source counts. `train_samples.csv`, `dev_samples.csv`, and `clients.json` record the exact sample, selected supports, full source counts, and index hashes.
- Each round preselects **one random test patient** and records its identity without opening its signals. All test inference is deferred until training/fine-tuning are locked. Each selected evaluation patient contributes all its examples. No test signal or score selects the sample, checkpoint, epoch, or configuration. Original development and test prevalence are retained; compare before/after metrics within each split.
- Dev/test readers and caches are reused across models/epochs instead of reparsing full source indexes for every evaluation. Batches remain streamed; no million-beat training tensor is required in RAM.

The full-data FedAvg schedule processes **3,345,269,500 beat presentations** for the current reproducible schedule (135 client training jobs). This is about 81% less FedAvg work than full participation, but still a large full-data schedule; the earlier eight-hour bounded-sample estimate does not apply. Fine-tuning and final test inference add work. A synthetic benchmark is not a guaranteed full-pipeline ETA.

## Commands

Run from the repository root using .NET 8:

```sh
dotnet build implementation/Implementation.csproj -c Release

# Numerical equivalence, gradient, aggregation, checkpoint, and test-gate checks.
dotnet run --project implementation -c Release --no-build -- self-test

# Measure the baseline and optimized paths on disposable synthetic batches.
dotnet run --project implementation -c Release --no-build -- benchmark

# Complete short synthetic pipeline, including bounded sampling and full synthetic test evaluation.
dotnet run --project implementation -c Release --no-build -- integration-test

# Inspect exact real-data sample counts and workload without fitting a model.
dotnet run --project implementation -c Release --no-build -- plan

# Start the full original-data pipeline only when ready.
dotnet run --project implementation -c Release --no-build -- pipeline \
  --config implementation/configs/fedavg.json \
  --out artifacts/fl/mobilenetv1-fedavg-cpu-3clients

# Resume all stages; omit --config and use the same output directory.
dotnet run --project implementation -c Release --no-build -- pipeline \
  --out artifacts/fl/mobilenetv1-fedavg-cpu-3clients
```

Use a new output directory for this protocol. Historical stopped and bounded runs retain their saved configuration; use the new full-data output directory instead of resuming them. Configuration/partition/index changes invalidate resume. Prepared NPZ signals are assumed immutable; every payload is not hashed.

The default configuration already sets `TrainSamplesPerClient` and `DevSamplesPerClient` to `null`. Any optional sampled experiment must use a separate, explicitly modified configuration and output directory. `OptimizedCpuConvolutions: false` restores the native numerical path for controlled comparisons. Neither option is changed automatically when resuming.

## Training and selection

Each round draws three distinct training clients without replacement within that round; selected clients execute sequentially. Clients can repeat across rounds and coverage is not guaranteed. The current three seeded 15-round schedules happen to cover all 16 clients at least once. Every selected client starts from that round's global model. Each original local training example is visited once per local epoch, shuffled without replacement. Adam (learning rate 0.001, batch size 128) resets for every client/round. Training-only inverse-frequency weights are fixed across clients/rounds/seeds. There is no early stopping or hidden step cap.

FedAvg uses distinct selected-example counts. Floating parameters and BatchNorm running statistics are sample-weighted; integer BatchNorm counters take the maximum. Float64 aggregation accumulators return to float32. Every update must match the round, client, sample count, tensor schema, and finite-value checks. Optimizer states are never averaged.

Global selection uses S/V PR-AUC on the randomly assigned dev patient for each round, separately for each seed, retaining the earliest round on ties. Dev patients rotate, so observed round scores also reflect patient/prevalence differences; best-observed rotating-dev selection is exploratory, not a constant-population comparison. Every committed round saves global weights and the three participating local updates before aggregation. Publication is atomic; interrupted rounds repeat from the previous complete global checkpoint. The process holds an exclusive run lock. Ctrl+C requests cancellation at a batch/client boundary.

## Local fine-tuning and final evaluation

Each participating client fine-tunes its last available local checkpoint, which may come from an earlier round, using all original training examples from its own client. Clients never selected have no fitted local model and are excluded from fine-tuning. The default finite grid is **3 learning rates (0.001, 0.0001, 0.00001) × head-only/all layers × up to 20 epochs**. Every candidate restarts from the same final local checkpoint with fresh Adam state and identical epoch shuffles. Head-only mode freezes the extractor and its BatchNorm statistics. The best epoch is part of configuration selection. The unchanged local model remains a candidate and wins a tie.

Each local model is selected by S/V PR-AUC on the same **single dev patient assigned to that client's last participation round**. No within-client patient split is introduced. These results measure transfer to unseen patients, not adaptation evaluated on that training patient's held-out data. The 16 models share evaluation patients and are not independent evaluation datasets.

All searches for participating client/seed pairs must finish and lock their chosen configurations/weights before final test inference. Test data receive no gradient updates or configuration/epoch/threshold selection. Test reports cover final global, dev-selected global, each participating client's latest local model, and each selected fine-tuned local model. Each uses the test patient assigned to its source round. Additional `round-tests/` reports evaluate every round's global model and three local updates on that round's preselected test patient. Saved test evaluations are reused on repeat invocation.

To run stages separately:

```sh
# FedAvg and development reports only.
dotnet run --project implementation -c Release --no-build -- run \
  --config implementation/configs/fedavg.json --out artifacts/fl/mobilenetv1-fedavg-cpu-3clients

# Continue interrupted FedAvg only.
dotnet run --project implementation -c Release --no-build -- resume \
  --out artifacts/fl/mobilenetv1-fedavg-cpu-3clients

# Finish/resume local development-only searches.
dotnet run --project implementation -c Release --no-build -- fine-tune \
  --out artifacts/fl/mobilenetv1-fedavg-cpu-3clients

# Evaluate locked models on all test beats, without fitting.
dotnet run --project implementation -c Release --no-build -- evaluate-test \
  --out artifacts/fl/mobilenetv1-fedavg-cpu-3clients
```

## Results and notebook

Open [Implementation_results.ipynb](Implementation_results.ipynb) with the **.NET (C#)** kernel, set `experimentDirectory`, and Run All to refresh. The notebook only reads saved JSON; it never starts training or test evaluation. It labels sample caps, provisional results, and synthetic integration runs. It reports per-seed and across-seed metrics, global/local comparisons, candidate histories, paired before/after comparisons, PR-AUC charts, per-class/patient metrics, and confusion matrices.

Metrics include S/V PR-AUC (trapezoidal), separate Average Precision, ROC-AUC, miss rate, FAR, accuracy, macro-F1, loss, and per-class results. S/V uses `p(S)+p(V)`, with N/Q as negatives and fixed threshold 0.5 for miss/FAR; Q retains its unclassifiable meaning. Undefined values are null. Mean/SD are calculated across available seeds separately for each model/client/split.

Outputs include `status.json`, `fine_tuning_status.json`, `test_status.json`, per-seed `round-NNN/{global.bin,locals/,metrics.json}`, `fine-tuning/client_*/{result.json,best.bin}`, and `model-reports/`. Signal caches in `implementation/cache/` and all experiment outputs are excluded from Git. All 135 local round checkpoints occupy roughly 1.8 GB before caches and other outputs.

The integration test uses a separate synthetic 16/2/2 fixture with two beats per class, four-beat train/dev caps, two global rounds, three participating clients per round, one local epoch, three seeds, and two one-epoch fine-tuning candidates. It exercises the actual optimized kernels, sampling, caches, optimizer, checkpoints, all metric reports, uncapped synthetic test inference, and resume hash checks. It never opens real patient data. Synthetic scores establish functionality only.

`seed-*/schedule.json` records all seeded train/dev/test assignments. `participation.json` records actual coverage and each client's last update. The notebook includes both, plus deferred per-round test reports; test identities/scores never influence later training or tuning.
