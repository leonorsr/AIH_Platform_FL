using System.Diagnostics;
using Aih.ModelTests;
using static TorchSharp.torch;

namespace Aih.Implementation;

public sealed record Throughput(string Variant, int BatchSize, int Threads, double TrainingExamplesPerSecond,
    double EvaluationExamplesPerSecond, double SecondsPerTrainingBatch);

public static class PerformanceBenchmark
{
    public static void Run(Configuration c, string output)
    {
        c.Validate(); set_num_threads(c.Threads);
        var measurements = new List<Throughput>();
        foreach (bool optimized in new[] { false, true })
        {
            manual_seed(2024);
            using var model = MobileNetV1.Create(1, optimizedCpu: optimized);
            using var optimizer = optim.Adam(model.parameters(), lr: c.LearningRate);
            using var x = rand(c.BatchSize, 1, 500);
            using var y = zeros(c.BatchSize, dtype: ScalarType.Int64);
            model.train(); double seconds = 0;
            for (int step = 0; step < 3; step++)
            {
                using var scope = NewDisposeScope(); optimizer.zero_grad();
                var watch = Stopwatch.StartNew();
                var loss = nn.functional.cross_entropy(model.forward(x), y);
                loss.backward(); optimizer.step();
                if (!float.IsFinite(loss.item<float>())) throw new InvalidDataException("Non-finite benchmark loss.");
                if (step > 0) seconds += watch.Elapsed.TotalSeconds;
            }
            model.eval(); double evaluationSeconds = 0;
            using (var ng = no_grad())
                for (int step = 0; step < 3; step++)
                {
                    using var scope = NewDisposeScope(); var watch = Stopwatch.StartNew();
                    _ = model.forward(x).sum().item<float>();
                    if (step > 0) evaluationSeconds += watch.Elapsed.TotalSeconds;
                }
            var measurement = new Throughput(optimized ? "vectorized-depthwise-and-matmul-pointwise" : "native-conv1d",
                c.BatchSize, c.Threads, 2 * c.BatchSize / seconds, 2 * c.BatchSize / evaluationSeconds, seconds / 2);
            measurements.Add(measurement);
            Console.WriteLine($"{measurement.Variant}: train {measurement.TrainingExamplesPerSecond:F1} examples/s, eval {measurement.EvaluationExamplesPerSecond:F1} examples/s");
        }
        Directory.CreateDirectory(output);
        Experiment.Json(Path.Combine(output, "throughput.json"), new { Measurements = measurements,
            TrainingSpeedup = measurements[1].TrainingExamplesPerSecond / measurements[0].TrainingExamplesPerSecond,
            Scope = "Synthetic in-memory batches on this machine; excludes data loading, metric calculation, checkpoints, and process pauses.",
            CompletedUtc = DateTime.UtcNow });
    }
}
