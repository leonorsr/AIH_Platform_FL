# AIH Platform FL

ECG analysis and data preparation in C# / .NET 8. All project documentation, notebooks, chart labels, and application messages are maintained in English.

- `data/`: 20 Icentia11k clients — 16 training, 2 development, and 2 test clients.
- `src/Aih.Ecg/`: C# application for analysis, preparation, and validation.
- `architecture/`: proposed federated-learning architecture using FedAvg.
- `model tests/`: neural-network smoke tests with PR-AUC, ROC-AUC, miss rate, and FAR.
- `implementation/`: executable C# FedAvg simulation using MobileNetV1 1D alpha 1.00, a pool of 16 training clients with three randomly selected once and fixed across all rounds and seeds, 25 local epochs per round, 10 rounds, and three seeds. See its [execution guide](implementation/README.md).
- `eda and data prep/`: C# analysis and preparation notebooks (.NET / Polyglot Notebooks kernel).

```sh
dotnet build src/Aih.Ecg -c Release
dotnet run --project src/Aih.Ecg -c Release -- eda
dotnet run --project src/Aih.Ecg -c Release -- self-test
```

The EDA report is saved to `artifacts/eda/eda.html`. To repeat preparation from raw WFDB recordings:

```sh
dotnet run --project src/Aih.Ecg -c Release -- prepare --raw data/raw --out data/reprepared
```

The output directory must be empty. The existing signals are already prepared. See [algorithm and validation details](src/Aih.Ecg/README.md).

[ModelTests_results.ipynb](model%20tests/ModelTests_results.ipynb) contains the data-partition explanation, smoke-test results table, and a single PR-AUC comparison chart.

## Choose an experiment

Run from the repository root. Every profile uses **original ECG data**, the same three training clients selected once, and three seeds. No beat-sampling caps apply.

| Profile | Global rounds | Local epochs per round | Optional fine-tuning grid |
| --- | ---: | ---: | --- |
| `real-check` | 1 | 1 | 1 learning rate × 2 layer scopes × 1 epoch |
| `real-pilot` | 3 | 1 | 2 learning rates × 2 layer scopes × 3 epochs |
| `fedavg` | 10 | 25 | 3 learning rates × 2 layer scopes × 20 epochs |

Build, choose a profile, inspect its plan, and start **FedAvg only**:

```sh
dotnet build implementation/Implementation.csproj -c Release

experiment=real-check  # Choose real-check, real-pilot, or fedavg.
run_dir="artifacts/fl/${experiment}-01"  # Use a new directory for a new run.

dotnet run --project implementation -c Release --no-build -- plan \
  --config "implementation/configs/${experiment}.json"

dotnet run --project implementation -c Release --no-build -- run \
  --config "implementation/configs/${experiment}.json" --out "$run_dir"
```

To include **fine-tuning and final test evaluation**, use `pipeline` **instead of** the `run` command above:

```sh
dotnet run --project implementation -c Release --no-build -- pipeline \
  --config "implementation/configs/${experiment}.json" --out "$run_dir"
```

To resume the same experiment and all remaining pipeline stages, use its saved configuration:

```sh
dotnet run --project implementation -c Release --no-build -- pipeline --out "$run_dir"
```

Even `real-check` uses all beats of the selected real clients; it is not a seconds-long synthetic test. Full pipeline time includes the fine-tuning grid in the table. See [implementation/README.md](implementation/README.md) for individual stages, customization, and the results notebook.
