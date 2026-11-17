# AIH Platform FL

ECG analysis and data preparation in C# / .NET 8. All project documentation, notebooks, chart labels, and application messages are maintained in English.

- `data/`: 20 Icentia11k clients — 16 training, 2 development, and 2 test clients.
- `src/Aih.Ecg/`: C# application for analysis, preparation, and validation.
- `architecture/`: proposed federated-learning architecture using FedAvg.
- `model tests/`: neural-network smoke tests with PR-AUC, ROC-AUC, miss rate, and FAR.
- `implementation/`: executable C# FedAvg simulation using MobileNetV1 1D alpha 1.00, 16 training clients, 100 local epochs per round, 15 rounds, and three seeds. See its [execution guide](implementation/README.md).
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

## Run the federated pipeline

From the repository root:

```sh
dotnet build implementation/Implementation.csproj -c Release

# Very short functional check on synthetic ECG; no real patient data.
dotnet run --project implementation -c Release --no-build -- integration-test

# Full experiment: FedAvg → local fine-tuning → held-out evaluation.
dotnet run --project implementation -c Release --no-build -- pipeline \
  --config implementation/configs/fedavg.json \
  --out artifacts/fl/mobilenetv1-fedavg-100local-15global

# Resume all stages of that experiment.
dotnet run --project implementation -c Release --no-build -- pipeline \
  --out artifacts/fl/mobilenetv1-fedavg-100local-15global
```

The full configuration uses MobileNetV1 1D alpha 1.00, 16 training clients, 100 local epochs per round, 15 global rounds, and three seeds. The integration test uses a separate tiny synthetic fixture. See the [execution guide](implementation/README.md) for planning, stage-specific commands, output paths, and the C# results notebook.
