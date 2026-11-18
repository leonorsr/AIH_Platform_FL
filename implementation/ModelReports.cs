using System.Text.Json;
using Aih.Ecg;
using Aih.ModelTests;

namespace Aih.Implementation;

public sealed record ModelEvaluation(int Seed, string Kind, string? Client, int Round, string Split,
    string Checkpoint, Evaluation Pooled, Dictionary<string, Evaluation> ByClient);

public static class ModelReports
{
    public static T Read<T>(string path) => JsonSerializer.Deserialize<T>(File.ReadAllText(path), ExperimentConfig.Json)
        ?? throw new InvalidDataException("Empty file: " + path);

    public static Configuration ValidateRun(string root, string run)
    {
        var c = Configuration.Load(Path.Combine(run, "config.json"));
        using var status = JsonDocument.Parse(File.ReadAllText(Path.Combine(run, "status.json")));
        if (status.RootElement.GetProperty("Status").GetString() != "completed") throw new InvalidOperationException("Complete all federated seeds first.");
        var data = new Dataset(Path.Combine(root, "data"));
        if (LocalData.Hash(Path.Combine(data.Root, "preparation_config.json")) != LocalData.Hash(Path.Combine(run, "dataset_config.json")))
            throw new InvalidDataException("Dataset configuration changed.");
        foreach (var info in Read<ClientInfo[]>(Path.Combine(run, "clients.json")))
        {
            var client = data.Clients.Single(x => x.Id == info.Id);
            if (LocalData.Hash(data.Index(client)) != info.IndexHash) throw new InvalidDataException("Client index changed: " + client.Id);
        }
        return c;
    }

    public static void Write(string root, string run, Configuration c, string split, CancellationToken token)
    {
        if (split is not ("dev" or "test")) throw new ArgumentException("Report split must be dev or test.");
        TorchSharp.torch.set_num_threads(c.Threads);
        var data = new Dataset(Path.Combine(root, "data"));
        string directory = Path.Combine(run, "model-reports"); Directory.CreateDirectory(directory);
        using var pool = new EvaluationPool(data, split, c, Path.Combine(root, "implementation/cache"));
        foreach (int seed in c.Seeds)
        {
            var history = RoundSchedule.History(run, c, seed);
            var selected = Read<SeedResult>(Path.Combine(run, $"seed-{seed}", "selected.json"));
            string Patient(int round) => split == "dev" ? history[round-1].Selection!.DevClient : history[round-1].Selection!.TestClient;
            var specifications = new List<(string Kind, string? Client, int Round, string Checkpoint, string Patient)> {
                ("global-final", null, c.GlobalRounds, Path.Combine($"seed-{seed}", $"round-{c.GlobalRounds:D3}", "global.bin"), Patient(c.GlobalRounds)),
                ("global-selected", null, selected.BestRound, selected.Checkpoint, Patient(selected.BestRound))
            };
            foreach (var source in RoundSchedule.LatestLocals(run, c, seed))
            {
                specifications.Add(("local-final", source.Client, source.Round, source.Checkpoint, Patient(source.Round)));
                if (split == "test")
                {
                    var tuning = Read<TuningResult>(Path.Combine(run, $"seed-{seed}", "fine-tuning", source.Client, "result.json"));
                    specifications.Add(("local-fine-tuned", source.Client, source.Round,
                        Path.Combine($"seed-{seed}", "fine-tuning", source.Client, tuning.Checkpoint), source.TestClient));
                }
            }
            foreach (var spec in specifications)
            {
                token.ThrowIfCancellationRequested();
                string path = Path.Combine(directory, $"{split}-seed-{seed}-{spec.Kind}-{spec.Client ?? "server"}.json");
                if (File.Exists(path)) continue;
                if (split == "dev")
                {
                    var round = history[spec.Round-1];
                    var evaluation = spec.Client is null ? round.Dev : round.LocalDev![spec.Client];
                    Experiment.Json(path, new ModelEvaluation(seed, spec.Kind, spec.Client, spec.Round, split, spec.Checkpoint,
                        evaluation, new Dictionary<string, Evaluation> { [spec.Patient] = evaluation }));
                    continue;
                }
                using var model = ModelFactory.Create(c); model.load(Path.Combine(run, spec.Checkpoint));
                var (pooled, byClient) = Experiment.Evaluate(model, data, split, c, Path.Combine(root, "implementation/cache"), token, pool.For(spec.Patient));
                Experiment.Json(path, new ModelEvaluation(seed, spec.Kind, spec.Client, spec.Round, split, spec.Checkpoint, pooled, byClient));
            }
        }
    }

    public static void WriteRoundTests(string root, string run, Configuration c, CancellationToken token)
    {
        // This method is called only after every fine-tuning configuration is locked.
        var data = new Dataset(Path.Combine(root, "data"));
        using var pool = new EvaluationPool(data, "test", c, Path.Combine(root, "implementation/cache"));
        foreach (int seed in c.Seeds) foreach (var round in RoundSchedule.History(run, c, seed))
        {
            string dir = Path.Combine(run, $"seed-{seed}", "round-tests"); Directory.CreateDirectory(dir);
            var specs = new List<(string Kind,string? Client,string Checkpoint)> {
                ("global-round", null, Path.Combine($"seed-{seed}", $"round-{round.Round:D3}", "global.bin"))
            };
            specs.AddRange(round.Clients.Select(client => ("local-round", (string?)client.Client,
                Path.Combine($"seed-{seed}", $"round-{round.Round:D3}", client.Checkpoint!))));
            foreach (var spec in specs)
            {
                token.ThrowIfCancellationRequested();
                string path = Path.Combine(dir, $"round-{round.Round:D3}-{spec.Client ?? "server"}.json");
                if (File.Exists(path)) continue;
                using var model = ModelFactory.Create(c); model.load(Path.Combine(run, spec.Checkpoint));
                var (pooled, byClient) = Experiment.Evaluate(model, data, "test", c, Path.Combine(root, "implementation/cache"), token,
                    pool.For(round.Selection!.TestClient));
                Experiment.Json(path, new ModelEvaluation(seed, spec.Kind, spec.Client, round.Round, "test", spec.Checkpoint, pooled, byClient));
            }
        }
    }
}
