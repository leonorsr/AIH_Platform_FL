using System.Text.Json;
using Aih.Ecg;
using Aih.ModelTests;
using TorchSharp;
using static TorchSharp.torch;

namespace Aih.Implementation;

public sealed record RoundResult(int Seed, int Round, ClientResult[] Clients, Evaluation Dev,
    Dictionary<string, Evaluation> DevByClient, double Seconds, RoundSelection? Selection = null,
    Dictionary<string, Evaluation>? LocalDev = null);
public sealed record SeedResult(int Seed, int BestRound, Evaluation Dev, string Checkpoint);

public static class Experiment
{
    public static void Json(string path, object value) => Trainer.WriteJson(path, value);
    static T Read<T>(string path) => JsonSerializer.Deserialize<T>(File.ReadAllText(path), ExperimentConfig.Json)
        ?? throw new InvalidDataException("Empty file: " + path);
    static string RoundPath(string seedDir, int round) => Path.Combine(seedDir, $"round-{round:D3}");

    public static ClientInfo[] Inspect(Dataset data, Configuration configuration)
    {
        // Test indexes and signals are deliberately excluded from preparation.
        var cohort = RoundSchedule.FixedCohort(data.Clients, configuration).ToHashSet();
        var clients = data.Clients.Where(c => c.Split == "dev" || (c.Split == "train"
            && (configuration.ClientSelectionMode != "fixed" || cohort.Contains(c.Id)))).Select(c => {
            var info = LocalData.Inspect(data, c, configuration);
            Console.WriteLine($"{info.Id}: {info.Split}, {info.Count:N0} selected / {info.SourceCount:N0} source examples");
            return info;
        }).ToArray();
        if (clients.Where(c => c.Split == "dev").Any(c => c.Supports[1] + c.Supports[2] == 0))
            throw new InvalidDataException("Every dev patient must contain S/V examples for rotating dev-client selection.");
        return clients;
    }

    public static void Plan(string root, Configuration c)
    {
        c.Validate();
        var dataset = new Dataset(Path.Combine(root, "data"));
        var clients = Inspect(dataset, c);
        var schedule = c.Seeds.SelectMany(seed => Enumerable.Range(1, c.GlobalRounds).Select(r => RoundSchedule.Choose(dataset.Clients, c, seed, r))).ToArray();
        Console.WriteLine(JsonSerializer.Serialize(new {
            c.Model, c.Seeds, c.GlobalRounds, c.LocalEpochs, c.BatchSize, c.LearningRate,
            c.TrainSamplesPerClient, c.DevSamplesPerClient, c.SamplingSeed, c.OptimizedCpuConvolutions,
            Clients = clients,
            TrainExamples = clients.Where(x => x.Split == "train").Sum(x => x.Count),
            DevExamples = clients.Where(x => x.Split == "dev").Sum(x => x.Count),
            c.ClientsPerRound,
            c.ClientSelectionMode, c.ClientSelectionSeed,
            Schedule = schedule,
            ClientTrainingJobs = (long)c.ClientsPerRound * c.GlobalRounds * c.Seeds.Length,
            LocalEpochsAcrossAllClients = (long)c.ClientsPerRound * c.GlobalRounds * c.LocalEpochs * c.Seeds.Length,
            TrainingExampleVisits = schedule.Sum(r => r.TrainClients.Sum(id => clients.Single(x => x.Id == id).Count)) * c.LocalEpochs,
            Aggregation = "distinct-example-weighted FedAvg",
            Optimizer = "Adam; reset each client/round",
            FineTuning = c.FineTuning,
            FineTuningTrainingJobs = schedule.GroupBy(x => x.Seed).Sum(g => (long)g.SelectMany(r => r.TrainClients).Distinct().Count()) * c.FineTuning.LearningRates.Length * c.FineTuning.TrainableLayers.Length,
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
        var infos = Inspect(data, c);
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
            Json(Path.Combine(run, "cohort.json"), new { c.ClientSelectionMode, c.ClientSelectionSeed,
                FixedAcrossRoundsAndTrainingSeeds = c.ClientSelectionMode == "fixed",
                TrainingClients = infos.Where(x => x.Split == "train").Select(x => new { x.Id, x.Patient, x.Count }).ToArray() });
            foreach (string split in new[] { "train", "dev" })
                Samples.SaveManifest(Path.Combine(run, split + "_samples.csv"), data.Clients.Where(x => x.Split == split && infos.Any(i => i.Id == x.Id))
                    .SelectMany(client => LocalData.Select(data, client, c, out _)));
            Json(Path.Combine(run, "environment.json"), new { StartedUtc = DateTime.UtcNow,
                Dotnet = Environment.Version.ToString(), OS = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
                TorchSharp = typeof(Tensor).Assembly.GetName().Version?.ToString(), Device = "cpu", c.Threads,
                Reproducibility = "Deterministic seed schedule; bitwise portability across hardware is not guaranteed." });
        }
        set_num_threads(c.Threads);
        string cacheRoot = Path.Combine(root, "implementation", "cache");
        using var developmentPool = new EvaluationPool(data, "dev", c, cacheRoot);
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
                Json(Path.Combine(seedDir, "schedule.json"), Enumerable.Range(1, c.GlobalRounds).Select(r => RoundSchedule.Choose(data.Clients, c, seed, r)).ToArray());
                // Only atomically published complete directories count as committed rounds.
                for (int r = 1; r <= c.GlobalRounds; r++)
                {
                    string committed = RoundPath(seedDir, r);
                    if (!Directory.Exists(committed)) break;
                    var result = Read<RoundResult>(Path.Combine(committed, "metrics.json"));
                    if (result.Seed != seed || result.Round != r || !File.Exists(Path.Combine(committed, "global.bin")))
                        throw new InvalidDataException("Invalid committed round.");
                    if (result.Clients.Length != c.ClientsPerRound || result.Clients.Any(x => x.Checkpoint is null || !File.Exists(Path.Combine(committed, x.Checkpoint))))
                        throw new InvalidDataException("Committed round is missing local model checkpoints.");
                    var expectedSelection = RoundSchedule.Choose(data.Clients, c, seed, r);
                    if (result.Selection is null || !result.Clients.Select(x => x.Client).Order().SequenceEqual(expectedSelection.TrainClients)
                        || result.Selection.DevClient != expectedSelection.DevClient || result.Selection.TestClient != expectedSelection.TestClient)
                        throw new InvalidDataException("Committed selection does not match the deterministic schedule.");
                    history.Add(result);
                }
                if (Directory.GetDirectories(seedDir, "round-*").Any(p => System.Text.RegularExpressions.Regex.IsMatch(Path.GetFileName(p), @"^round-\d+$")
                    && !Enumerable.Range(1, history.Count).Select(r => RoundPath(seedDir, r)).Contains(p)))
                    throw new InvalidDataException("Non-contiguous round checkpoints.");
                manual_seed(seed);
                using var global = ModelFactory.Create(c);
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
                    var selection = RoundSchedule.Choose(data.Clients, c, seed, round);
                    var participants = selection.TrainClients.ToDictionary(id => id, id => train[id]);
                    using var aggregator = new FedAvg(global, participants, round);
                    int clientIndex = 0;
                    foreach (var client in data.Clients.Where(x => participants.ContainsKey(x.Id)))
                    {
                        token.ThrowIfCancellationRequested();
                        using var localData = new LocalData(data, client, c, cacheRoot);
                        if (localData.Rows.LongLength != train[client.Id]) throw new InvalidDataException("Client example count changed.");
                        using var local = ModelFactory.Create(c); local.load(initial);
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
                    var (dev, byClient) = Evaluate(global, data, "dev", c, cacheRoot, token, developmentPool.For(selection.DevClient));
                    _ = Trainer.Selection(dev, "sv-pr-auc");
                    var localDev = new Dictionary<string, Evaluation>();
                    foreach (var client in results)
                    {
                        using var local = ModelFactory.Create(c); local.load(Path.Combine(staging, client.Checkpoint!));
                        var (evaluation, _) = Evaluate(local, data, "dev", c, cacheRoot, token, developmentPool.For(selection.DevClient));
                        localDev.Add(client.Client, evaluation);
                    }
                    var metrics = new RoundResult(seed, round, results.ToArray(), dev, byClient, watch.Elapsed.TotalSeconds, selection, localDev);
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
                Json(Path.Combine(seedDir, "participation.json"), data.Clients.Where(x => x.Split == "train").Select(client => new {
                    Client = client.Id, Participations = history.Count(r => r.Clients.Any(x => x.Client == client.Id)),
                    LastRound = history.Where(r => r.Clients.Any(x => x.Client == client.Id)).Select(r => (int?)r.Round).LastOrDefault()
                }).ToArray());
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
        Dataset data, string split, Configuration c, string cacheRoot, CancellationToken token, EvaluationData? prepared = null)
    {
        model.eval(); using var noGrad = no_grad();
        var pooled = new Confusion(c.RareThreshold);
        var byClient = new Dictionary<string, Evaluation>();
        using var owned = prepared is null ? new EvaluationData(data, split, c, cacheRoot) : null;
        var evaluationData = prepared ?? owned!;
        if (evaluationData.Split != split) throw new InvalidDataException("Evaluation split mismatch.");
        foreach (var entry in evaluationData.Clients)
        {
            var client = entry.Client; var local = entry.Data;
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
        Json(Path.Combine(run, "summary.json"), new { Scope = "Configured development sample; each seed selected independently by S/V PR-AUC; see config.json for sample caps",
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
        ModelReports.WriteRoundTests(root, run, c, token);
        foreach (int seed in c.Seeds)
        {
            var selected = Read<SeedResult>(Path.Combine(run, $"seed-{seed}", "selected.json"));
            var report = Read<ModelEvaluation>(Path.Combine(run, "model-reports", $"test-seed-{seed}-global-selected-server.json"));
            string path = Path.Combine(run, $"seed-{seed}", "test.json");
            if (!File.Exists(path)) Json(path, new { Seed = seed, selected.BestRound, report.Pooled, report.ByClient, EvaluatedUtc = DateTime.UtcNow });
        }
        Json(Path.Combine(run, "test_status.json"), new { Status = "completed", Policy = "One preselected test patient per round/source; all inference deferred until tuning is locked", CompletedUtc = DateTime.UtcNow });
    }
}
