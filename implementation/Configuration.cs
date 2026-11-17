using System.Text.Json;
using Aih.ModelTests;

namespace Aih.Implementation;

public sealed class Configuration
{
    public string Name { get; set; } = "icentia11k-mobilenetv1-fedavg";
    public string Model { get; set; } = "mobilenetv1-1d-100";
    public int[] Seeds { get; set; } = [2024, 2025, 2026];
    public int GlobalRounds { get; set; } = 15;
    public int LocalEpochs { get; set; } = 100;
    public int BatchSize { get; set; } = 128;
    public double LearningRate { get; set; } = .001;
    public int Threads { get; set; } = 4;
    public string ClassWeighting { get; set; } = "inverse-frequency";
    public double RareThreshold { get; set; } = .5;
    public bool UseDiskCache { get; set; } = true;
    public FineTuningConfig FineTuning { get; set; } = new();

    public static Configuration Load(string path)
    {
        var c = JsonSerializer.Deserialize<Configuration>(File.ReadAllText(path), ExperimentConfig.Json)
            ?? throw new InvalidDataException("Empty configuration.");
        c.Validate();
        return c;
    }

    public void Validate()
    {
        if (Model != "mobilenetv1-1d-100") throw new ArgumentException("This experiment uses the smoke-test winner: MobileNetV1 1D alpha 1.00.");
        if (Seeds.Length != 3 || Seeds.Distinct().Count() != 3 || Seeds.Any(s => s < 0)) throw new ArgumentException("Exactly three distinct nonnegative seeds are required.");
        if (GlobalRounds <= 0 || LocalEpochs <= 0 || BatchSize < 2 || Threads <= 0 || !double.IsFinite(LearningRate) || LearningRate <= 0
            || !double.IsFinite(RareThreshold) || RareThreshold < 0 || RareThreshold > 1) throw new ArgumentException("Invalid training configuration.");
        if (ClassWeighting is not ("none" or "inverse-frequency")) throw new ArgumentException("Invalid class weighting.");
        FineTuning.Validate();
    }
}

public sealed class FineTuningConfig
{
    public double[] LearningRates { get; set; } = [.001, .0001, .00001];
    public string[] TrainableLayers { get; set; } = ["head", "all"];
    public int MaxEpochs { get; set; } = 20;
    public void Validate()
    {
        if (MaxEpochs <= 0 || LearningRates.Length == 0 || LearningRates.Distinct().Count() != LearningRates.Length
            || LearningRates.Any(x => !double.IsFinite(x) || x <= 0)
            || TrainableLayers.Length == 0 || TrainableLayers.Distinct().Count() != TrainableLayers.Length
            || TrainableLayers.Any(x => x is not ("head" or "all"))) throw new ArgumentException("Invalid fine-tuning search grid.");
    }
}
