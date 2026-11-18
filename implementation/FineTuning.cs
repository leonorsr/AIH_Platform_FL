using System.Globalization;
using Aih.Ecg;
using Aih.ModelTests;
using TorchSharp;
using static TorchSharp.torch;

namespace Aih.Implementation;

public sealed record TuningEpoch(int Epoch, double TrainLoss, Evaluation Dev);
public sealed record TuningCandidate(string Id, string Layers, double LearningRate, int BestEpoch, Evaluation Dev,
    Dictionary<string, Evaluation> ByClient, TuningEpoch[] History);
public sealed record TuningResult(int Seed, string Client, string SourceCheckpoint, string SourceHash,
    string BestConfiguration, string Layers, double LearningRate, int Epochs, Evaluation BaselineDev,
    Evaluation Dev, Dictionary<string, Evaluation> ByClient, string Checkpoint, TuningCandidate[] Candidates, int SourceRound = 0, string DevClient = "", string TestClient = "");

public static class FineTuning
{
    static string Plan(Configuration c) => System.Text.Json.JsonSerializer.Serialize(new { c.Model, c.Seeds, c.GlobalRounds, c.BatchSize,
        c.ClassWeighting, c.RareThreshold, c.FineTuning, c.OptimizedCpuConvolutions,
        c.TrainSamplesPerClient, c.DevSamplesPerClient, c.SamplingSeed, c.ClientsPerRound,
        c.ClientSelectionMode, c.ClientSelectionSeed }, ExperimentConfig.Json);
    // Strict comparison keeps the unchanged model on a tie, then the first
    // candidate/earliest epoch in the recorded deterministic search order.
    public static bool Improves(Evaluation candidate, Evaluation best)
        => Trainer.Selection(candidate, "sv-pr-auc") > Trainer.Selection(best, "sv-pr-auc");

    public static void SetLayers(nn.Module<Tensor, Tensor> model, string layers)
    {
        if (layers is not ("head" or "all")) throw new ArgumentException("Unknown layer scope.");
        foreach (var (name, parameter) in model.named_parameters())
            parameter.requires_grad_(layers == "all" || name.StartsWith("output.", StringComparison.Ordinal));
        // In head-only mode, keep the entire feature extractor and its BatchNorm
        // running statistics fixed. MobileNetV1 has no dropout.
        if (layers == "head") model.eval(); else model.train();
    }

    public static void Run(string root, string run, Configuration c, CancellationToken token)
    {
        var data = new Dataset(Path.Combine(root, "data"));
        set_num_threads(c.Threads);
        using var developmentPool = new EvaluationPool(data, "dev", c, Path.Combine(root, "implementation/cache"));
        string plan = Plan(c);
        string planPath = Path.Combine(run, "fine_tuning_plan.json");
        if (File.Exists(planPath) && File.ReadAllText(planPath) != plan)
            throw new InvalidDataException("Fine-tuning configuration changed. Start a new experiment.");
        if (!File.Exists(planPath)) File.WriteAllText(planPath, plan);
        if (Directory.EnumerateFiles(Path.Combine(run, "model-reports"), "test-*.json").Any()) RequireCompleted(run, c);
        // Reuse exactly the FedAvg training-only loss weights across candidates.
        using var weightsDoc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(run, "class_weights.json")));
        var weightsElement = weightsDoc.RootElement.GetProperty("Weights");
        float[]? weights = weightsElement.ValueKind == System.Text.Json.JsonValueKind.Null ? null
            : weightsElement.EnumerateArray().Select(x => x.GetSingle()).ToArray();
        foreach (int seed in c.Seeds)
        {
            int clientIndex = 0;
            foreach (var sourceInfo in RoundSchedule.LatestLocals(run, c, seed))
            {
                token.ThrowIfCancellationRequested();
                var client = data.Clients.Single(x => x.Id == sourceInfo.Client);
                string destination = Path.Combine(run, $"seed-{seed}", "fine-tuning", client.Id);
                string source = sourceInfo.Checkpoint;
                string sourcePath = Path.Combine(run, source), sourceHash = LocalData.Hash(sourcePath);
                if (File.Exists(Path.Combine(destination, "result.json")))
                {
                    var existing = ModelReports.Read<TuningResult>(Path.Combine(destination, "result.json"));
                    if (existing.SourceHash != sourceHash || !File.Exists(Path.Combine(destination, existing.Checkpoint)))
                        throw new InvalidDataException("Fine-tuning source/checkpoint changed: " + client.Id);
                    Experiment.Json(Path.Combine(run, "model-reports", $"dev-seed-{seed}-local-fine-tuned-{client.Id}.json"),
                        new ModelEvaluation(seed, "local-fine-tuned", client.Id, sourceInfo.Round, "dev",
                            Path.Combine($"seed-{seed}", "fine-tuning", client.Id, existing.Checkpoint), existing.Dev, existing.ByClient));
                    clientIndex++; continue;
                }
                string staging = destination + ".partial";
                if (Directory.Exists(staging)) Directory.Delete(staging, true);
                Directory.CreateDirectory(staging);
                var baseline = ModelReports.Read<ModelEvaluation>(Path.Combine(run, "model-reports", $"dev-seed-{seed}-local-final-{client.Id}.json"));
                _ = Trainer.Selection(baseline.Pooled, "sv-pr-auc");
                Evaluation bestDev = baseline.Pooled;
                var bestByClient = baseline.ByClient;
                string bestId = "unchanged", bestLayers = "none"; double bestRate = 0; int bestEpoch = 0;
                File.Copy(sourcePath, Path.Combine(staging, "best.bin"));
                var candidates = new List<TuningCandidate> {
                    new("unchanged", "none", 0, 0, baseline.Pooled, baseline.ByClient, [])
                };
                using var localData = new LocalData(data, client, c, Path.Combine(root, "implementation/cache"));
                int searchSeed = LocalTraining.Seed(seed, c.GlobalRounds + 1, clientIndex++);
                foreach (string layers in c.FineTuning.TrainableLayers)
                foreach (double rate in c.FineTuning.LearningRates)
                {
                    token.ThrowIfCancellationRequested();
                    // Every candidate restarts from the same final local model,
                    // with the same epoch shuffles and a fresh Adam optimizer.
                    manual_seed(searchSeed);
                    using var model = ModelFactory.Create(c); model.load(sourcePath); SetLayers(model, layers);
                    using var optimizer = optim.Adam(model.parameters().Where(p => p.requires_grad), lr: rate);
                    using var classWeights = weights is null ? null : tensor(weights, dtype: ScalarType.Float32);
                    string id = layers + "-lr-" + rate.ToString("G", CultureInfo.InvariantCulture);
                    var history = new List<TuningEpoch>();
                    Evaluation? candidateBest = null; Dictionary<string, Evaluation>? candidateByClient = null; int candidateEpoch = 0;
                    for (int epoch = 1; epoch <= c.FineTuning.MaxEpochs; epoch++)
                    {
                        SetLayers(model, layers);
                        int[] order = Enumerable.Range(0, localData.Rows.Length).ToArray();
                        new Random(LocalTraining.Seed(searchSeed, 0, 0, epoch)).Shuffle(order);
                        double sumLoss = 0, sumWeight = 0;
                        for (int start = 0; start < order.Length; start += c.BatchSize)
                        {
                            token.ThrowIfCancellationRequested(); using var scope = NewDisposeScope();
                            var rows = order.Skip(start).Take(c.BatchSize).Select(i => localData.Rows[i]).ToArray();
                            var (x, y) = Trainer.Batch(rows, localData.Read);
                            optimizer.zero_grad(); var logits = model.forward(x);
                            var loss = nn.functional.cross_entropy(logits, y, weight: classWeights);
                            double value = loss.item<float>();
                            if (!double.IsFinite(value)) throw new InvalidDataException("Non-finite fine-tuning loss.");
                            loss.backward(); optimizer.step();
                            double weight = weights is null ? rows.Length : rows.Sum(r => (double)weights[r.Label]);
                            sumLoss += value * weight; sumWeight += weight;
                            if ((start / c.BatchSize) % 250 == 0 || start + rows.Length == order.Length)
                                Experiment.Json(Path.Combine(run, "fine_tuning_status.json"), new { Status = "training", Seed = seed, Client = client.Id,
                                    Candidate = id, Epoch = epoch, c.FineTuning.MaxEpochs, Processed = start + rows.Length, Total = order.Length });
                        }
                        var (dev, byClient) = Experiment.Evaluate(model, data, "dev", c, Path.Combine(root, "implementation/cache"), token, developmentPool.For(sourceInfo.DevClient));
                        _ = Trainer.Selection(dev, "sv-pr-auc");
                        history.Add(new(epoch, sumLoss / sumWeight, dev));
                        if (candidateBest is null || Improves(dev, candidateBest))
                        { candidateBest = dev; candidateByClient = byClient; candidateEpoch = epoch; }
                        if (Improves(dev, bestDev))
                        {
                            bestDev = dev; bestByClient = byClient; bestId = id; bestLayers = layers; bestRate = rate; bestEpoch = epoch;
                            model.save(Path.Combine(staging, "best.bin.tmp"));
                            File.Move(Path.Combine(staging, "best.bin.tmp"), Path.Combine(staging, "best.bin"), true);
                        }
                        Experiment.Json(Path.Combine(staging, "active_candidate.json"), new { Id = id, History = history });
                        Console.WriteLine($"Fine-tuning seed {seed}, {client.Id}, {id}, epoch {epoch}: dev S/V PR-AUC {dev.RareSv.PrAuc:F6}");
                    }
                    candidates.Add(new(id, layers, rate, candidateEpoch, candidateBest!, candidateByClient!, history.ToArray()));
                    Experiment.Json(Path.Combine(staging, "candidates.json"), candidates);
                }
                // Lock the chosen configuration and weights before any test access.
                var result = new TuningResult(seed, client.Id, source, sourceHash, bestId, bestLayers, bestRate, bestEpoch,
                    baseline.Pooled, bestDev, bestByClient, "best.bin", candidates.ToArray(), sourceInfo.Round, sourceInfo.DevClient, sourceInfo.TestClient);
                Experiment.Json(Path.Combine(staging, "result.json"), result);
                File.Delete(Path.Combine(staging, "active_candidate.json"));
                Directory.Move(staging, destination);
                Experiment.Json(Path.Combine(run, "model-reports", $"dev-seed-{seed}-local-fine-tuned-{client.Id}.json"),
                    new ModelEvaluation(seed, "local-fine-tuned", client.Id, sourceInfo.Round, "dev",
                        Path.Combine($"seed-{seed}", "fine-tuning", client.Id, "best.bin"), bestDev, bestByClient));
            }
        }
        Experiment.Json(Path.Combine(run, "fine_tuning_status.json"), new { Status = "completed", CompletedUtc = DateTime.UtcNow,
            Selection = "Per participating local model, S/V PR-AUC on the single dev patient assigned to its last participation round", TestRead = false });
    }

    public static void RequireCompleted(string run, Configuration c)
    {
        string planPath = Path.Combine(run, "fine_tuning_plan.json");
        if (!File.Exists(planPath) || File.ReadAllText(planPath) != Plan(c))
            throw new InvalidOperationException("Complete the saved fine-tuning search plan before test evaluation.");
        foreach (int seed in c.Seeds) foreach (var source in RoundSchedule.LatestLocals(run, c, seed))
        {
            string dir = Path.Combine(run, $"seed-{seed}", "fine-tuning", source.Client);
            if (!File.Exists(Path.Combine(dir, "result.json")) || !File.Exists(Path.Combine(dir, "best.bin")))
                throw new InvalidOperationException("Finish and lock all local fine-tuning searches before test evaluation.");
        }
    }
}
