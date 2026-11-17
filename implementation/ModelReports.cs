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
        foreach (int seed in c.Seeds)
        {
            string seedDir = Path.Combine(run, $"seed-{seed}");
            var selected = Read<SeedResult>(Path.Combine(seedDir, "selected.json"));
            var specifications = new List<(string Kind, string? Client, int Round, string Checkpoint)> {
                ("global-final", null, c.GlobalRounds, Path.Combine($"seed-{seed}", $"round-{c.GlobalRounds:D3}", "global.bin")),
                ("global-selected", null, selected.BestRound, selected.Checkpoint)
            };
            foreach (var client in data.Clients.Where(x => x.Split == "train"))
            {
                specifications.Add(("local-final", client.Id, c.GlobalRounds,
                    Path.Combine($"seed-{seed}", $"round-{c.GlobalRounds:D3}", "locals", client.Id + ".bin")));
                if (split == "test")
                {
                    var tuning = Read<TuningResult>(Path.Combine(seedDir, "fine-tuning", client.Id, "result.json"));
                    specifications.Add(("local-fine-tuned", client.Id, c.GlobalRounds,
                        Path.Combine($"seed-{seed}", "fine-tuning", client.Id, tuning.Checkpoint)));
                }
            }
            foreach (var spec in specifications)
            {
                token.ThrowIfCancellationRequested();
                string path = Path.Combine(directory, $"{split}-seed-{seed}-{spec.Kind}-{spec.Client ?? "server"}.json");
                if (File.Exists(path)) continue;
                if (split == "dev" && spec.Kind.StartsWith("global-", StringComparison.Ordinal))
                {
                    var round = Read<RoundResult>(Path.Combine(seedDir, $"round-{spec.Round:D3}", "metrics.json"));
                    Experiment.Json(path, new ModelEvaluation(seed, spec.Kind, spec.Client, spec.Round, split, spec.Checkpoint, round.Dev, round.DevByClient));
                    continue;
                }
                using var model = Networks.Create(c.Model); model.load(Path.Combine(run, spec.Checkpoint));
                var (pooled, byClient) = Experiment.Evaluate(model, data, split, c, Path.Combine(root, "implementation/cache"), token);
                Experiment.Json(path, new ModelEvaluation(seed, spec.Kind, spec.Client, spec.Round, split, spec.Checkpoint, pooled, byClient));
                Console.WriteLine($"Report {split}, seed {seed}, {spec.Kind}, {spec.Client}: S/V PR-AUC {pooled.RareSv.PrAuc:F6}");
            }
        }
    }
}
