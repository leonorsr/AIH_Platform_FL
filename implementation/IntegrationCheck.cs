using System.Diagnostics;
using System.Text.Json;
using Aih.Ecg;
using Aih.ModelTests;

namespace Aih.Implementation;

// Runs the production training/evaluation code against an isolated synthetic
// 16/2/2 fixture. Never opens the real dataset or its held-out test patients.
public static class IntegrationCheck
{
    static void Require(bool condition, string message)
    { if (!condition) throw new InvalidDataException("Integration check failed: " + message); }

    public static void Run(string output, CancellationToken token)
    {
        string directory = Path.GetFullPath(output);
        if (Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any())
            throw new IOException("Integration-test output must be empty. Use a new directory.");
        Directory.CreateDirectory(directory);
        var watch = Stopwatch.StartNew();
        string fixture = Path.Combine(directory, "synthetic-fixture");
        string data = Path.Combine(fixture, "data");
        Directory.CreateDirectory(data);
        var splitPatients = new Dictionary<string, string[]> {
            ["train"] = Enumerable.Range(0, 16).Select(i => $"synthetic_{i:D2}").ToArray(),
            ["dev"] = ["synthetic_16", "synthetic_17"], ["test"] = ["synthetic_18", "synthetic_19"]
        };
        Experiment.Json(Path.Combine(data, "preparation_config.json"), new {
            purpose = "synthetic integration fixture; not Icentia11k performance evaluation",
            patient_ids = splitPatients,
            patient_to_client = Enumerable.Range(0, 20).ToDictionary(i => $"synthetic_{i:D2}", i => $"client_{i:D2}")
        });
        for (int client = 0; client < 20; client++)
        {
            string split = client < 16 ? "train" : client < 18 ? "dev" : "test";
            string id = $"client_{client:D2}", patient = $"synthetic_{client:D2}";
            string indexDir = Path.Combine(data, split == "train" ? "clients" : "evaluation_clients");
            Directory.CreateDirectory(indexDir);
            using var writer = new StreamWriter(Path.Combine(indexDir, id + ".csv"));
            writer.WriteLine("client_id,patient_id,split,n_samples,sampling_rate_hz,label_id,label,segment_path");
            for (int label = 0; label < 4; label++)
            {
                var values = Enumerable.Range(0, 500).Select(i => (float)(.5 + .3 * Math.Sin(i * (.03 + label * .01) + client * .07)
                    + .1 * Math.Cos(i * .08))).ToArray();
                string relative = $"{split}/{patient}/beat-{label}.npz";
                Npz.Write(Path.Combine(data, relative), values, Dataset.Labels[label], label, "Synthetic",
                    new Quality(0, 0, 0, 0, 0, "Synthetic"));
                writer.WriteLine(Csv.Line([id, patient, split, 500, 200, label, Dataset.Labels[label], relative]));
            }
        }
        var c = new Configuration {
            Name = "integration-smoke-synthetic", GlobalRounds = 1, LocalEpochs = 1,
            BatchSize = 4, Threads = 2, LearningRate = .0001, UseDiskCache = true,
            FineTuning = new FineTuningConfig { LearningRates = [.0001], TrainableLayers = ["head", "all"], MaxEpochs = 1 }
        };
        string config = Path.Combine(directory, "integration-config.json"), run = Path.Combine(directory, "results");
        Experiment.Json(config, c);
        Experiment.Json(Path.Combine(directory, "verification.json"), new { Status = "running", SyntheticOnly = true, StartedUtc = DateTime.UtcNow });
        Console.WriteLine("INTEGRATION TEST: synthetic ECG only; 16 clients; 3 seeds; 1 round; 1 local epoch; 2 one-epoch fine-tuning candidates per client.");
        try
        {
            Experiment.Run(fixture, config, run, false, token);
            Program.Tune(fixture, run, token);
            Experiment.EvaluateTest(fixture, run, token);
            var manifests = ModelReports.Read<ClientInfo[]>(Path.Combine(run, "clients.json"));
            Require(manifests.Where(x => x.Split == "train").Sum(x => x.Count) == 64, "training sample count");
            int localModels = 0, searches = 0;
            foreach (int seed in c.Seeds)
            {
                var round = ModelReports.Read<RoundResult>(Path.Combine(run, $"seed-{seed}/round-001/metrics.json"));
                Require(round.Clients.Length == 16 && round.Clients.All(x => x.Examples == 4 && x.Epochs == 1), "client updates");
                for (int client = 0; client < 16; client++)
                {
                    Require(File.Exists(Path.Combine(run, $"seed-{seed}/round-001/locals/client_{client:D2}.bin")), "local checkpoint"); localModels++;
                    var result = ModelReports.Read<TuningResult>(Path.Combine(run, $"seed-{seed}/fine-tuning/client_{client:D2}/result.json"));
                    Require(result.Candidates.Length == 3 && result.Candidates.Count(x => x.History.Length == 1) == 2, "candidate history");
                    Require(result.Dev.RareSv.PrAuc >= result.BaselineDev.RareSv.PrAuc, "baseline selection safeguard");
                    searches++;
                }
            }
            var reports = Directory.GetFiles(Path.Combine(run, "model-reports"), "*.json").Select(ModelReports.Read<ModelEvaluation>).ToArray();
            Require(reports.Length == 204 && reports.Count(r => r.Split == "test") == 102, "global/local/fine-tuned dev/test report counts");
            Require(reports.All(r => r.Pooled.Count == 8 && r.ByClient.Count == 2 && double.IsFinite(r.Pooled.Loss)
                && r.Pooled.RareSv.PrAuc.HasValue && r.Pooled.RareSv.RocAuc.HasValue), "finite evaluation metrics");
            // Repeating the entire pipeline must not train or modify checkpoints.
            var checkpointHashes = Directory.GetFiles(run, "*.bin", SearchOption.AllDirectories).ToDictionary(p => p, LocalData.Hash);
            Experiment.Run(fixture, Path.Combine(run, "config.json"), run, true, token);
            Program.Tune(fixture, run, token);
            Experiment.EvaluateTest(fixture, run, token);
            Require(checkpointHashes.All(p => LocalData.Hash(p.Key) == p.Value), "resume changed a completed checkpoint");
            Experiment.Json(Path.Combine(directory, "verification.json"), new {
                Status = "passed", SyntheticOnly = true, RealPatientDataRead = false,
                Seeds = c.Seeds.Length, ClientsPerRound = 16, GlobalRounds = 1, LocalEpochs = 1,
                LocalCheckpoints = localModels, FineTuningSearches = searches, ModelReports = reports.Length,
                ResumePreservesCheckpoints = true, Seconds = watch.Elapsed.TotalSeconds,
                ResultsDirectory = run, CompletedUtc = DateTime.UtcNow,
                Scope = "Functional integration only; synthetic metrics do not estimate ECG model performance."
            });
            Console.WriteLine($"INTEGRATION TEST PASSED in {watch.Elapsed.TotalSeconds:F1}s. Verification: {Path.Combine(directory, "verification.json")}");
        }
        catch (Exception ex)
        {
            Experiment.Json(Path.Combine(directory, "verification.json"), new { Status = "failed", SyntheticOnly = true, Error = ex.ToString(), Seconds = watch.Elapsed.TotalSeconds });
            throw;
        }
    }
}
