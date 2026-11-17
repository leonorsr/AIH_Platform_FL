using System.Text.Json;
using Aih.Ecg;
using Aih.ModelTests;
using TorchSharp;
using static TorchSharp.torch;

namespace Aih.Implementation;

public sealed record RoundResult(int Seed, int Round, ClientResult[] Clients, Evaluation Dev,
    Dictionary<string, Evaluation> DevByClient, double Seconds);
public sealed record SeedResult(int Seed, int BestRound, Evaluation Dev, string Checkpoint);

public static class Experiment
{
    public static void Json(string path, object value) => Trainer.WriteJson(path, value);
    static T Read<T>(string path) => JsonSerializer.Deserialize<T>(File.ReadAllText(path), ExperimentConfig.Json)
        ?? throw new InvalidDataException("Empty file: " + path);
    static string RoundPath(string seedDir, int round) => Path.Combine(seedDir, $"round-{round:D3}");

    public static ClientInfo[] Inspect(Dataset data)
    {
        // Test indexes and signals are deliberately excluded from preparation.
        var clients = data.Clients.Where(c => c.Split != "test").Select(c => {
            var info = LocalData.Inspect(data, c);
            Console.WriteLine($"{info.Id}: {info.Split}, {info.Count:N0} examples");
            return info;
        }).ToArray();
        if (clients.Where(c => c.Split == "dev").Sum(c => c.Supports[1] + c.Supports[2]) == 0)
            throw new InvalidDataException("S/V development PR-AUC is undefined: no positive examples.");
        return clients;
    }

    public static void Plan(string root, Configuration c)
    {
        c.Validate();
        var clients = Inspect(new Dataset(Path.Combine(root, "data")));
        Console.WriteLine(JsonSerializer.Serialize(new {
            c.Model, c.Seeds, c.GlobalRounds, c.LocalEpochs, c.BatchSize, c.LearningRate,
            Clients = clients,
            TrainExamples = clients.Where(x => x.Split == "train").Sum(x => x.Count),
            DevExamples = clients.Where(x => x.Split == "dev").Sum(x => x.Count),
            ClientTrainingJobs = 16L * c.GlobalRounds * c.Seeds.Length,
            LocalEpochsAcrossAllClients = 16L * c.GlobalRounds * c.LocalEpochs * c.Seeds.Length,
            TrainingExampleVisits = clients.Where(x => x.Split == "train").Sum(x => x.Count) * c.GlobalRounds * c.LocalEpochs * c.Seeds.Length,
            Aggregation = "distinct-example-weighted FedAvg",
            Optimizer = "Adam; reset each client/round",
            FineTuning = c.FineTuning,
            FineTuningTrainingJobs = 16L * c.Seeds.Length * c.FineTuning.LearningRates.Length * c.FineTuning.TrainableLayers.Length,
            CheckpointMetric = "pooled development S/V PR-AUC; earliest round on ties",
            TestRead = false, TrainingStarted = false
        }, ExperimentConfig.Json));
    }

    public static void Run(string root, string configPath, string output, bool resume, CancellationToken token)
    {
        var c = Configuration.Load(configPath);
        string run = Path.GetFullPath(output);
        Directory.CreateDirectory(run);
        // Exclusive ownership prevents two processes from publishing the same round.
        using var runLock = new FileStream(Path.Combine(run, ".run.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        if (!resume && Directory.EnumerateFileSystemEntries(run).Any(p => Path.GetFileName(p) != ".run.lock"))
            throw new IOException("Use an empty output directory, or resume an existing experiment.");
        var data = new Dataset(Path.Combine(root, "data"));
        var infos = Inspect(data);
        string configJson = JsonSerializer.Serialize(c, ExperimentConfig.Json);
        string manifestJson = JsonSerializer.Serialize(infos, ExperimentConfig.Json);
        string partition = Path.Combine(data.Root, "preparation_config.json");
        if (resume)
        {
            if (File.ReadAllText(Path.Combine(run, "config.json")) != configJson
                || File.ReadAllText(Path.Combine(run, "clients.json")) != manifestJson
                || LocalData.Hash(partition) != LocalData.Hash(Path.Combine(run, "dataset_config.json")))
                throw new InvalidDataException("Configuration or patient indexes changed. Start a new experiment.");
        }
        else
        {
            File.WriteAllText(Path.Combine(run, "config.json"), configJson);
            File.WriteAllText(Path.Combine(run, "clients.json"), manifestJson);
            File.Copy(partition, Path.Combine(run, "dataset_config.json"));
            Json(Path.Combine(run, "environment.json"), new { StartedUtc = DateTime.UtcNow,
                Dotnet = Environment.Version.ToString(), OS = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
                TorchSharp = typeof(Tensor).Assembly.GetName().Version?.ToString(), Device = "cpu", c.Threads,
                Reproducibility = "Deterministic seed schedule; bitwise portability across hardware is not guaranteed." });
        }
        set_num_threads(c.Threads);
        string cacheRoot = Path.Combine(root, "implementation", "cache");
        var train = infos.Where(i => i.Split == "train").ToDictionary(i => i.Id, i => i.Count);
        long[] supports = Enumerable.Range(0, 4).Select(i => infos.Where(x => x.Split == "train").Sum(x => x.Supports[i])).ToArray();
        float[]? weights = c.ClassWeighting == "inverse-frequency"
            ? supports.Select(n => n == 0 ? 0f : (float)supports.Sum() / (4 * n)).ToArray() : null;
        Json(Path.Combine(run, "class_weights.json"), new { c.ClassWeighting, Supports = supports, Weights = weights, Scope = "training clients only; fixed for all clients/rounds/seeds" });
        var completed = new List<SeedResult>();
        try
        {
            foreach (int seed in c.Seeds)
            {
                token.ThrowIfCancellationRequested();
                string seedDir = Path.Combine(run, $"seed-{seed}"); Directory.CreateDirectory(seedDir);
                var history = new List<RoundResult>();
                // Only atomically published complete directories count as committed rounds.
                for (int r = 1; r <= c.GlobalRounds; r++)
                {
                    string committed = RoundPath(seedDir, r);
                    if (!Directory.Exists(committed)) break;
                    var result = Read<RoundResult>(Path.Combine(committed, "metrics.json"));
                    if (result.Seed != seed || result.Round != r || !File.Exists(Path.Combine(committed, "global.bin")))
                        throw new InvalidDataException("Invalid committed round.");
                    if (result.Clients.Length != 16 || result.Clients.Any(x => x.Checkpoint is null || !File.Exists(Path.Combine(committed, x.Checkpoint))))
                        throw new InvalidDataException("Committed round is missing local model checkpoints.");
                    history.Add(result);
                }
                if (Directory.GetDirectories(seedDir, "round-*").Any(p => !p.EndsWith(".partial", StringComparison.Ordinal)
                    && !Enumerable.Range(1, history.Count).Select(r => RoundPath(seedDir, r)).Contains(p)))
                    throw new InvalidDataException("Non-contiguous round checkpoints.");
                manual_seed(seed);
                using var global = Networks.Create(c.Model);
                if (history.Count > 0) global.load(Path.Combine(RoundPath(seedDir, history.Count), "global.bin"));
                for (int round = history.Count + 1; round <= c.GlobalRounds; round++)
                {
                    token.ThrowIfCancellationRequested();
                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    string staging = RoundPath(seedDir, round) + ".partial";
                    if (Directory.Exists(staging)) Directory.Delete(staging, true);
                    Directory.CreateDirectory(staging);
                    // Every client starts from precisely the same global state.
                    string initial = Path.Combine(staging, "input.bin"); global.save(initial);
                    var results = new List<ClientResult>();
                    using var aggregator = new FedAvg(global, train, round);
                    int clientIndex = 0;
                    foreach (var client in data.Clients.Where(x => x.Split == "train"))
                    {
                        token.ThrowIfCancellationRequested();
                        using var localData = new LocalData(data, client, c, cacheRoot);
                        if (localData.Rows.LongLength != train[client.Id]) throw new InvalidDataException("Client example count changed.");
                        using var local = Networks.Create(c.Model); local.load(initial);
                        var result = LocalTraining.Train(local, localData, c, seed, round, clientIndex++, weights,
                            p => Json(Path.Combine(run, "status.json"), p), token);
                        aggregator.Add(client.Id, round, result.Examples, local);
                        string localRelative = Path.Combine("locals", client.Id + ".bin");
                        Directory.CreateDirectory(Path.Combine(staging, "locals"));
                        local.save(Path.Combine(staging, localRelative));
                        results.Add(result with { Checkpoint = localRelative });
                        Json(Path.Combine(staging, "clients.json"), results);
                    }
                    aggregator.Apply(global);
                    token.ThrowIfCancellationRequested();
                    Json(Path.Combine(run, "status.json"), new { Status = "evaluating-development", Seed = seed, Round = round });
                    var (dev, byClient) = Evaluate(global, data, "dev", c, cacheRoot, token);
                    _ = Trainer.Selection(dev, "sv-pr-auc");
                    var metrics = new RoundResult(seed, round, results.ToArray(), dev, byClient, watch.Elapsed.TotalSeconds);
                    global.save(Path.Combine(staging, "global.bin"));
                    Json(Path.Combine(staging, "metrics.json"), metrics);
                    File.Delete(initial);
                    Directory.Move(staging, RoundPath(seedDir, round));
                    history.Add(metrics);
                    Console.WriteLine($"Seed {seed}, round {round}: dev S/V PR-AUC {dev.RareSv.PrAuc:F6}, ROC-AUC {dev.RareSv.RocAuc:F6}");
                    SaveHistory(seedDir, history);
                }
                SaveHistory(seedDir, history);
                var best = history.OrderByDescending(h => h.Dev.RareSv.PrAuc).ThenBy(h => h.Round).First();
                var selected = new SeedResult(seed, best.Round, best.Dev,
                    Path.GetRelativePath(run, Path.Combine(RoundPath(seedDir, best.Round), "global.bin")));
                Json(Path.Combine(seedDir, "selected.json"), selected); completed.Add(selected);
                Json(Path.Combine(run, "seeds.json"), completed);
            }
            WriteSummary(run, completed);
            Json(Path.Combine(run, "status.json"), new { Status = "completed", Seeds = completed.Count, TestRead = false, CompletedUtc = DateTime.UtcNow });
            ModelReports.Write(root, run, c, "dev", token);
        }
        catch (Exception ex)
        {
            Json(Path.Combine(run, "status.json"), new { Status = ex is OperationCanceledException ? "interrupted" : "failed", Error = ex.Message,
                Recovery = "Resume from the last complete global round; incomplete local work is repeated.", UpdatedUtc = DateTime.UtcNow });
            throw;
        }
    }

    public static (Evaluation Pooled, Dictionary<string, Evaluation> ByClient) Evaluate(nn.Module<Tensor, Tensor> model,
        Dataset data, string split, Configuration c, string cacheRoot, CancellationToken token)
    {
        model.eval(); using var noGrad = no_grad();
        var pooled = new Confusion(c.RareThreshold);
        var byClient = new Dictionary<string, Evaluation>();
        foreach (var client in data.Clients.Where(x => x.Split == split))
        {
            using var local = new LocalData(data, client, c, cacheRoot);
            var metrics = new Confusion(c.RareThreshold);
            for (int start = 0; start < local.Rows.Length; start += c.BatchSize)
            {
                token.ThrowIfCancellationRequested(); using var scope = NewDisposeScope();
                var rows = local.Rows.Skip(start).Take(c.BatchSize).ToArray();
                var (x, y) = Trainer.Batch(rows, local.Read);
                var logits = model.forward(x);
                double loss = nn.functional.cross_entropy(logits, y).item<float>();
                var probabilities = logits.softmax(1).data<float>().ToArray();
                var truth = rows.Select(r => (long)r.Label).ToArray();
                pooled.Add(truth, probabilities, loss); metrics.Add(truth, probabilities, loss);
            }
            byClient.Add(client.Id, metrics.Result());
        }
        return (pooled.Result(), byClient);
    }

    static void SaveHistory(string seedDir, List<RoundResult> history)
    {
        using var writer = new StreamWriter(Path.Combine(seedDir, "rounds.csv"));
        writer.WriteLine("round,accuracy,sv_pr_auc,sv_roc_auc,sv_miss_rate,sv_far,seconds");
        foreach (var r in history) writer.WriteLine(Csv.Line([r.Round, r.Dev.Accuracy, r.Dev.RareSv.PrAuc,
            r.Dev.RareSv.RocAuc, r.Dev.RareSv.MissRate, r.Dev.RareSv.Far, r.Seconds]));
    }

    static void WriteSummary(string run, List<SeedResult> results)
    {
        object Stats(Func<SeedResult, double?> metric)
        {
            var a = results.Select(metric).Where(v => v.HasValue).Select(v => v!.Value).ToArray();
            double? mean = a.Length == 0 ? null : a.Average();
            return new { Mean = mean, PopulationStdDev = mean.HasValue ? (double?)Math.Sqrt(a.Select(v => Math.Pow(v - mean.Value, 2)).Average()) : null, DefinedSeeds = a.Length };
        }
        Json(Path.Combine(run, "summary.json"), new { Scope = "Full development clients; each seed selected independently by S/V PR-AUC",
            Seeds = results, SvPrAuc = Stats(r => r.Dev.RareSv.PrAuc), SvRocAuc = Stats(r => r.Dev.RareSv.RocAuc),
            SvMissRate = Stats(r => r.Dev.RareSv.MissRate), SvFar = Stats(r => r.Dev.RareSv.Far), Accuracy = Stats(r => r.Dev.Accuracy) });
    }

    public static void EvaluateTest(string root, string output, CancellationToken token)
    {
        string run = Path.GetFullPath(output);
        using var runLock = new FileStream(Path.Combine(run, ".run.lock"), FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        var c = ModelReports.ValidateRun(root, run);
        FineTuning.RequireCompleted(run, c);
        set_num_threads(c.Threads);
        ModelReports.Write(root, run, c, "test", token);
        foreach (int seed in c.Seeds)
        {
            var selected = Read<SeedResult>(Path.Combine(run, $"seed-{seed}", "selected.json"));
            var report = Read<ModelEvaluation>(Path.Combine(run, "model-reports", $"test-seed-{seed}-global-selected-server.json"));
            string path = Path.Combine(run, $"seed-{seed}", "test.json");
            if (!File.Exists(path)) Json(path, new { Seed = seed, selected.BestRound, report.Pooled, report.ByClient, EvaluatedUtc = DateTime.UtcNow });
        }
        Json(Path.Combine(run, "test_status.json"), new { Status = "completed", ModelsPerSeed = 34, CompletedUtc = DateTime.UtcNow });
    }
}
