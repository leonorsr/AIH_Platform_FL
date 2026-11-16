# ECG processing in C#

A .NET 8 application without NuGet dependencies or calls to other runtimes. Both ECG notebooks use this implementation.

- `Data.cs`: sequential CSV reading, patient configuration, and NPZ/NPY input/output.
- `Exploration.cs`: class distributions, quality, and examples; exports CSV, SVG, and HTML.
- `Wfdb.cs`: reads Icentia11k ECGs (one channel, format 16, 250 Hz) and N/S/V/Q annotations, including SKIP.
- `SignalProcessing.cs`: Kaiser FIR 4/5 resampling, zero-phase filters, gradient and Pan–Tompkins detectors, one-to-one temporal matching, interpolation, normalization, and fuzzy SQI.
- `Preparation.cs`: processes recordings, writes client/split indexes, and validates exported arrays.

Fixed coefficients are defined in `Coefficients.cs`: a fifth-order Butterworth high-pass filter at 0.5 Hz, a 60 Hz notch with Q=35, and a 101-tap Kaiser FIR for 250→200 Hz resampling. The detectors follow the [NeuroKit2 algorithms](https://github.com/neuropsychology/NeuroKit/blob/v0.2.11/neurokit2/ecg/ecg_findpeaks.py); the license is preserved in `licenses/`. SQI implements the fuzzy fusion method of [Zhao & Zhang](https://doi.org/10.3389/fphys.2018.00727), adapted to these detectors and one-second windows.

## Usage

From the repository root:

```sh
dotnet build src/Aih.Ecg -c Release
dotnet run --project src/Aih.Ecg -c Release -- eda
dotnet run --project src/Aih.Ecg -c Release -- validate
dotnet run --project src/Aih.Ecg -c Release -- self-test
```

`validate` checks indexes, file counts, and one array per client; `validate --all` reads every array. The notebooks require a .NET (C#) kernel, such as [Polyglot Notebooks](https://github.com/dotnet/interactive), and a Release build. The CLI works without a notebook kernel.

For a smaller preparation run, use `prepare --raw data/raw --out data/prepared-small --limit 1` (one recording per patient). To process one recording, add `--record data/raw/pXX/pXXXXX/pXXXXX_s00`. Paths are resolved from the current directory; `--record` takes the prefix without an extension. The output directory must be empty. An interrupted run does not publish a completed validation report; choose another directory to repeat the run. Checkpoint recovery is not implemented for preparation.

Full preparation reads patient partitions and IDs from `data/preparation_config.json`, processes one continuous recording at a time, and writes indexes sequentially. The `eda` command does not reprocess the existing signals in `data/`.

## Completed validation

- Release compilation without errors or warnings.
- Tests for CSV, resampling, filters, synthetic R-peak detection, annotation matching, boundaries, normalization, SQI, NPZ, and WFDB SKIP.
- Analysis of all 20 clients: 4,865,610 beats and local examples from all four classes.
- Preparation of the real recording `p00290_s00`: 6,339 beats. R peaks, annotations, labels, client IDs, and SQI categories matched the saved index. The maximum absolute difference across five compared arrays was `5.96e-8`; the maximum fuzzy-score difference for the recording was `6.89e-15`.

The numerical comparison covers one recording, not every patient. Rounding may affect decisions near thresholds. Q means unclassifiable and is not automatically an arrhythmia. Development and test patients remain excluded from training.
