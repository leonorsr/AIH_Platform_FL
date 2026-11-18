# Run an ECG experiment

All profiles use **original prepared ECG signals**, three training clients selected once and kept fixed across rounds/seeds, three seeds, and optimized CPU kernels. Training/dev caps are `null`. One random dev patient and one preselected test patient are assigned per round; test inference is deferred until choices are locked.

## Choose the configuration

| Profile / file in `implementation/configs/` | Global rounds | Local epochs | Fine-tuning learning rates | Layers | Max tuning epochs |
| --- | ---: | ---: | --- | --- | ---: |
| `real-check.json` | 1 | 1 | 0.0001 | head, all | 1 |
| `real-pilot.json` | 3 | 1 | 0.0001, 0.00001 | head, all | 3 |
| `fedavg.json` | 10 | 25 | 0.001, 0.0001, 0.00001 | head, all | 20 |

The cohort seed is 2024 in all profiles: **client_01, client_09, client_11**, with **637,703 original training beats**. `real-check` is a smaller real-data experiment, not a synthetic smoke test; it still traverses every beat of these clients.

## Build and run

From the repository root:

```sh
dotnet build implementation/Implementation.csproj -c Release

experiment=real-check  # Or real-pilot or fedavg.
run_dir="artifacts/fl/${experiment}-01"

# Inspect identities, exact workload, and configuration without training.
dotnet run --project implementation -c Release --no-build -- plan \
  --config "implementation/configs/${experiment}.json"

# FedAvg and development reports only: no fine-tuning or test evaluation.
dotnet run --project implementation -c Release --no-build -- run \
  --config "implementation/configs/${experiment}.json" --out "$run_dir"
```

For the **complete pipeline**, replace `run` above with:

```sh
dotnet run --project implementation -c Release --no-build -- pipeline \
  --config "implementation/configs/${experiment}.json" --out "$run_dir"
```

`pipeline` executes FedAvg → global/local dev reports → local fine-tuning search → locked test evaluation. Its fine-tuning budget is the one listed in the selected profile. Use a new output directory for every new experiment. No command is run automatically by opening the README or results notebook.

## Resume or run individual stages

Set `run_dir` to an existing experiment directory. Resume uses its saved configuration; omit `--config`.

```sh
# Resume FedAvg only.
dotnet run --project implementation -c Release --no-build -- resume --out "$run_dir"

# Resume all remaining stages.
dotnet run --project implementation -c Release --no-build -- pipeline --out "$run_dir"

# Run/resume only local fine-tuning after FedAvg has completed.
dotnet run --project implementation -c Release --no-build -- fine-tune --out "$run_dir"

# Evaluate test data after all fine-tuning configurations have been locked.
dotnet run --project implementation -c Release --no-build -- evaluate-test --out "$run_dir"
```

Use Ctrl+C in the training terminal to request a stop. Resume repeats an interrupted global round from its previous complete checkpoint; it does not continue from an unsaved local epoch. Interrupted fine-tuning repeats that client's search. Completed rounds, searches, and test evaluations are reused.

## Customize an experiment

Copy a profile, then edit the copy before starting:

```sh
cp implementation/configs/real-check.json implementation/configs/my-experiment.json

dotnet run --project implementation -c Release --no-build -- plan \
  --config implementation/configs/my-experiment.json

dotnet run --project implementation -c Release --no-build -- run \
  --config implementation/configs/my-experiment.json --out artifacts/fl/my-experiment-01
```

Edit `GlobalRounds`, `LocalEpochs`, and `FineTuning` for training budgets. Change `ClientSelectionSeed` to draw another fixed trio; `ClientSelectionMode: "fixed"` keeps it identical across rounds and all three seeds. `Seeds` requires exactly three distinct training seeds. Keep `TrainSamplesPerClient` and `DevSamplesPerClient` as `null` to retain all original data. Changing a saved experiment's configuration requires a new run directory.

## Results

Open [Implementation_results.ipynb](Implementation_results.ipynb) using the **.NET (C#)** kernel, set `experimentDirectory` to `run_dir`, and Run All. It only reads saved results. Tables include global/local metrics, selected tuning configurations, candidate histories, paired comparisons, participation coverage, and deferred per-round tests. Metrics include PR-AUC, ROC-AUC, miss rate, FAR, accuracy, loss, class/patient results, and confusion matrices.

`status.json`, `fine_tuning_status.json`, and `test_status.json` track each phase. `cohort.json` records the fixed clients; `seed-*/schedule.json` records dev/test assignments. Dev patients rotate, so differences between round scores can reflect patient/prevalence differences. Fine-tuning always compares candidates on the same dev patient assigned to that local model. Test data never select configurations, thresholds, epochs, or seeds.

## Optional software diagnostics

These commands use synthetic tensors/fixtures, **not clinical performance experiments**:

```sh
dotnet run --project implementation -c Release --no-build -- self-test
dotnet run --project implementation -c Release --no-build -- benchmark
dotnet run --project implementation -c Release --no-build -- integration-test
```

The optimized convolutions preserve architecture, parameter schema, and outputs/gradients within checked numerical tolerances. Caches and generated results/checkpoints are excluded from Git. CPU throughput is machine-dependent; a software test passing does not establish model accuracy or a full-experiment completion time.
