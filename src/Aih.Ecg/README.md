# ECG em C#

Aplicação .NET 8 sem pacotes NuGet ou chamadas a outros runtimes. Os dois notebooks usam esta mesma implementação.

- `Data.cs`: CSV sequencial, configuração dos pacientes e NPZ/NPY.
- `Exploration.cs`: classes, qualidade e exemplos; exporta CSV, SVG e relatório HTML.
- `Wfdb.cs`: leitura dos ECG Icentia11k (um canal, formato 16, 250 Hz) e anotações N/S/V/Q, incluindo SKIP.
- `SignalProcessing.cs`: FIR Kaiser 4/5, filtros de fase zero, detetores de gradiente e Pan–Tompkins, associação temporal 1:1, interpolação, normalização e SQI fuzzy.
- `Preparation.cs`: preparação por registo, índices por cliente/split e validação dos arrays exportados.

Os coeficientes fixos estão em `Coefficients.cs`: Butterworth de 5.ª ordem a 0,5 Hz, notch a 60 Hz com Q=35 e FIR Kaiser de 101 taps para reamostragem 250→200 Hz. Os detetores seguem os algoritmos disponibilizados pelo [NeuroKit2](https://github.com/neuropsychology/NeuroKit/blob/v0.2.11/neurokit2/ecg/ecg_findpeaks.py), com a licença preservada em `licenses/`. O SQI implementa a fusão fuzzy de [Zhao & Zhang](https://doi.org/10.3389/fphys.2018.00727), adaptada aos detetores e a janelas de 1 segundo.

## Utilização

A partir da raiz do repositório:

```sh
dotnet build src/Aih.Ecg -c Release
dotnet run --project src/Aih.Ecg -c Release -- eda
dotnet run --project src/Aih.Ecg -c Release -- validate
dotnet run --project src/Aih.Ecg -c Release -- self-test
```

`validate` verifica os índices, as contagens de ficheiros e um array por cliente; `validate --all` lê todos os arrays. Os notebooks requerem um kernel .NET (C#), por exemplo através de [Polyglot Notebooks](https://github.com/dotnet/interactive), e a aplicação compilada em Release. O comando CLI funciona sem o kernel.

Para uma preparação menor, usar `prepare --raw data/raw --out data/prepared-small --limit 1` (um registo por paciente). Para um único registo, acrescentar `--record data/raw/pXX/pXXXXX/pXXXXX_s00`. Os caminhos são resolvidos a partir da pasta atual; `--record` recebe o prefixo sem extensão. A saída deve estar vazia. Uma execução interrompida não publica um relatório de validação concluída; usar outra pasta para repetir. Não há retoma de checkpoints.

A preparação completa lê as divisões e os IDs em `data/preparation_config.json`, processa um ECG contínuo de cada vez e escreve os índices sequencialmente. Os dados existentes em `data/` não são reprocessados pelo comando `eda`.

## Validação realizada

- Compilação Release sem erros nem avisos.
- Testes de CSV, reamostragem, filtros, deteção R sintética, associação de anotações, fronteiras, normalização, SQI, NPZ e WFDB SKIP.
- Análise dos 20 clientes: 4 865 610 batimentos e exemplos locais das quatro classes.
- Preparação real de `p00290_s00`: 6 339 batimentos. Picos R, anotações, rótulos, cliente e categorias SQI coincidiram com o índice guardado. Nos cinco arrays comparados, a maior diferença absoluta foi `5,96e-8`; nos scores fuzzy do registo, `6,89e-15`.

Esta comparação cobre um registo, não demonstra equivalência numérica em todos os pacientes. Arredondamentos podem alterar decisões junto dos limiares. Q representa não classificável e não é automaticamente uma arritmia. Dev e teste permanecem fora do treino.
