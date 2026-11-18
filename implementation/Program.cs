using Aih.Ecg;
using Aih.ModelTests;

namespace Aih.Implementation;

public static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            if (args.Length == 0 || args[0] is "help" or "--help")
            {
                Console.WriteLine("C# ECG FedAvg simulation\n"
                    + "  plan [--config path]                  Inspect indexes; no signals, cache building, or training\n"
                    + "  self-test                             Synthetic aggregation checks; no training\n"
                    + "  benchmark [--config path] [--out dir]  Short synthetic CPU throughput comparison\n"
                    + "  integration-test [--out empty-dir]     Very short complete pipeline on synthetic ECG only\n"
                    + "  run [--config path] --out empty-dir    Explicitly start training\n"
                    + "  resume --out run-dir                  Resume after the last complete round\n"
                    + "  fine-tune --out run-dir                Search local configurations using shared dev patients\n"
                    + "  evaluate-test --out run-dir            Evaluate locked global/local/fine-tuned models\n"
                    + "  pipeline [--config path] --out dir     Explicitly run/resume FedAvg, dev reports, fine-tuning, then test\n"
                    + "Defaults: MobileNetV1 1D alpha 1.00; 3 fixed clients drawn from 16; 25 local epochs per round; 10 rounds; 3 seeds.");
                return 0;
            }
            if (args[0] == "self-test")
            {
                if (args.Length != 1) throw new ArgumentException("self-test accepts no options.");
                NativeBackend.Initialize(); Checks.Run(); return 0;
            }
            if (args[0] is not ("plan" or "benchmark" or "run" or "resume" or "fine-tune" or "evaluate-test" or "pipeline" or "integration-test")) throw new ArgumentException("Unknown command.");
            var options = new Dictionary<string, string>();
            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] is not ("--config" or "--out") || i + 1 >= args.Length) throw new ArgumentException("Invalid option.");
                options.Add(args[i], args[++i]);
            }
            if ((args[0] is "resume" or "fine-tune" or "evaluate-test") && options.ContainsKey("--config")) throw new ArgumentException("Existing experiments use their saved configuration.");
            string root = Dataset.FindRoot();
            string config = options.GetValueOrDefault("--config", Path.Combine(root, "implementation/configs/fedavg.json"));
            if (args[0] == "plan") { Experiment.Plan(root, Configuration.Load(config)); return 0; }
            if (args[0] == "benchmark")
            {
                NativeBackend.Initialize();
                PerformanceBenchmark.Run(Configuration.Load(config), options.GetValueOrDefault("--out",
                    Path.Combine(root, "artifacts/fl/benchmarks", DateTime.UtcNow.ToString("yyyyMMdd-HHmmss"))));
                return 0;
            }
            string output = options.GetValueOrDefault("--out") ?? (args[0] == "integration-test"
                ? Path.Combine(root, "artifacts/fl/integration-smoke", DateTime.UtcNow.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N")[..6])
                : throw new ArgumentException("Provide --out explicitly."));
            NativeBackend.Initialize();
            using var cancellation = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
            if (args[0] == "integration-test")
            {
                if (options.ContainsKey("--config")) throw new ArgumentException("integration-test uses its fixed bounded synthetic configuration.");
                IntegrationCheck.Run(output, cancellation.Token);
            }
            else if (args[0] == "evaluate-test") Experiment.EvaluateTest(root, output, cancellation.Token);
            else if (args[0] == "fine-tune") Tune(root, output, cancellation.Token);
            else
            {
                bool resume = args[0] == "resume" || (args[0] == "pipeline" && File.Exists(Path.Combine(output, "config.json")));
                if (resume && args[0] == "pipeline" && options.ContainsKey("--config")) throw new ArgumentException("A resumed pipeline uses its saved configuration; omit --config.");
                Experiment.Run(root, resume ? Path.Combine(output, "config.json") : config, output, resume, cancellation.Token);
                if (args[0] == "pipeline") { Tune(root, output, cancellation.Token); Experiment.EvaluateTest(root, output, cancellation.Token); }
            }
            return 0;
        }
        catch (OperationCanceledException) { Console.Error.WriteLine("Stopped. Resume will restart the interrupted round from its last committed global checkpoint."); return 130; }
        catch (Exception ex) { Console.Error.WriteLine(ex); return 1; }
    }

    internal static void Tune(string root, string output, CancellationToken token)
    {
        string run = Path.GetFullPath(output);
        using var runLock = new FileStream(Path.Combine(run, ".run.lock"), FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        var c = ModelReports.ValidateRun(root, run);
        ModelReports.Write(root, run, c, "dev", token);
        try { FineTuning.Run(root, run, c, token); }
        catch (Exception ex)
        {
            Experiment.Json(Path.Combine(run, "fine_tuning_status.json"), new { Status = ex is OperationCanceledException ? "interrupted" : "failed",
                Error = ex.Message, Recovery = "Repeat fine-tune or pipeline; completed client searches are reused." });
            throw;
        }
    }
}
