# AIH Platform FL

Análise de ECG e preparação de dados em C# / .NET 8.

- `data/`: 20 clientes Icentia11k — 16 treino, 2 dev e 2 teste.
- `src/Aih.Ecg/`: aplicação C# de análise, preparação e validação.
- `eda and data prep/`: notebooks C# para análise e preparação (kernel .NET / Polyglot Notebooks).

```sh
dotnet build src/Aih.Ecg -c Release
dotnet run --project src/Aih.Ecg -c Release -- eda
dotnet run --project src/Aih.Ecg -c Release -- self-test
```

O relatório fica em `artifacts/eda/eda.html`. Para repetir a preparação a partir dos brutos WFDB:

```sh
dotnet run --project src/Aih.Ecg -c Release -- prepare --raw data/raw --out data/reprepared
```

A saída deve estar vazia. Os dados existentes já estão preparados. Detalhes dos algoritmos e validação em [src/Aih.Ecg/README.md](src/Aih.Ecg/README.md).
