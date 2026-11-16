# AIH Platform FL

ECG analysis and data preparation in C# / .NET 8. All project documentation, notebooks, chart labels, and application messages are maintained in English.

- `data/`: 20 Icentia11k clients — 16 training, 2 development, and 2 test clients.
- `src/Aih.Ecg/`: C# application for analysis, preparation, and validation.
- `architecture/`: proposed federated-learning architecture using FedAvg.
- `model tests/`: neural-network smoke tests with PR-AUC, ROC-AUC, miss rate, and FAR.
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
