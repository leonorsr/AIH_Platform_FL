using Aih.ModelTests;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Aih.Implementation;

// These checks never run backpropagation, an optimizer, or the ECG training loop.
public static class Checks
{
    static void Require(bool ok, string message) { if (!ok) throw new Exception(message); }
    static void Reject(Action action)
    {
        try { action(); }
        catch (Exception e) when (e is ArgumentException or InvalidDataException or InvalidOperationException) { return; }
        throw new Exception("Expected invalid input to be rejected.");
    }
    static Module<Tensor, Tensor> Tiny() => Sequential(("linear", Linear(1, 1)), ("bn", BatchNorm1d(1)));
    static void Fill(Module<Tensor, Tensor> model, double value, long counter)
    {
        using var ng = no_grad();
        foreach (var t in model.state_dict().Values) t.fill_(t.dtype == ScalarType.Int64 ? counter : value);
    }

    public static void Run()
    {
        PerformanceChecks.Run();
        var c = new Configuration(); c.Validate();
        Require(c.LocalEpochs == 25 && c.GlobalRounds == 10 && c.Seeds.Length == 3, "Incorrect experiment defaults.");
        Reject(() => new Configuration { Seeds = [1, 1, 2] }.Validate());
        Reject(() => new Configuration { LearningRate = double.NaN }.Validate());
        Reject(() => new Configuration { Model = "mlp" }.Validate());
        Reject(() => new FineTuningConfig { LearningRates = [double.NaN] }.Validate());
        Reject(() => new FineTuningConfig { TrainableLayers = ["head", "head"] }.Validate());
        Reject(() => FineTuning.RequireCompleted(Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N")), c));
        var pool = Enumerable.Range(0,20).Select(i => new Aih.Ecg.Client($"client_{i:D2}", $"patient_{i:D2}", i < 16 ? "train" : i < 18 ? "dev" : "test")).ToArray();
        var first = RoundSchedule.Choose(pool,c,2024,1); var again = RoundSchedule.Choose(pool,c,2024,1);
        Require(first.TrainClients.SequenceEqual(again.TrainClients) && first.DevClient == again.DevClient && first.TestClient == again.TestClient, "Schedule must be reproducible.");
        foreach (int seed in c.Seeds) foreach (int round in Enumerable.Range(1,c.GlobalRounds))
        {
            var selection = RoundSchedule.Choose(pool,c,seed,round);
            Require(selection.TrainClients.SequenceEqual(first.TrainClients), "Fixed cohort must be identical across rounds and training seeds.");
            Require(selection.TrainClients.Length == 3 && selection.TrainClients.Distinct().Count() == 3, "Exactly three distinct participants required.");
            Require(selection.TrainClients.All(id => pool.Any(x => x.Id == id && x.Split == "train"))
                && pool.Any(x => x.Id == selection.DevClient && x.Split == "dev")
                && pool.Any(x => x.Id == selection.TestClient && x.Split == "test"), "Selection crosses patient partitions.");
        }
        using var global = Tiny(); using var a = Tiny(); using var b = Tiny();
        Fill(global, 0, 0); Fill(a, 2, 2); Fill(b, 8, 7);
        var counts = new Dictionary<string, long> { ["a"] = 1, ["b"] = 3 };
        using (var avg = new FedAvg(global, counts, 1))
        {
            Reject(() => avg.Apply(global));
            Reject(() => avg.Add("a", 0, 1, a));
            Reject(() => avg.Add("x", 1, 1, a));
            Reject(() => avg.Add("a", 1, 2, a));
            avg.Add("a", 1, 1, a);
            Reject(() => avg.Add("a", 1, 1, a));
            Reject(() => avg.Apply(global));
            avg.Add("b", 1, 3, b); avg.Apply(global);
            foreach (var (name, t) in global.state_dict())
            {
                using var scope = NewDisposeScope();
                if (t.dtype == ScalarType.Int64) Require(t.item<long>() == 7, "BatchNorm counters must take the maximum.");
                else Require(t.flatten().data<float>().ToArray().All(v => Math.Abs(v - 6.5) < 1e-6), "Sample-weighted FedAvg failed: " + name);
            }
        }
        using (var avg = new FedAvg(global, new Dictionary<string, long> { ["a"] = 1 }, 2))
        {
            using var wrong = Linear(2, 1);
            Reject(() => avg.Add("a", 2, 1, wrong));
            Fill(a, double.NaN, 0); Reject(() => avg.Add("a", 2, 1, a));
            Fill(a, 4, 3); avg.Add("a", 2, 1, a); avg.Apply(global);
            Require(Math.Abs(global.state_dict()["linear.weight"].item<float>() - 4) < 1e-6, "Single-client aggregation failed.");
        }
        // Verify the full winning architecture and its BatchNorm state are supported.
        using var winner = Networks.Create(c.Model);
        Require(winner.parameters().Sum(p => p.numel()) == 3180548, "Winner architecture changed.");
        FineTuning.SetLayers(winner, "head");
        Require(winner.named_parameters().All(p => p.parameter.requires_grad == p.name.StartsWith("output.", StringComparison.Ordinal)), "Head-only freezing failed.");
        FineTuning.SetLayers(winner, "all");
        Require(winner.parameters().All(p => p.requires_grad), "Full fine-tuning did not unfreeze all parameters.");
        var confusion = new Confusion();
        confusion.Add([0, 1], [.8f, .1f, .05f, .05f, .05f, .8f, .1f, .05f], .1);
        var original = confusion.Result();
        Require(!FineTuning.Improves(original, original), "A tie must retain the original checkpoint.");
        var lower = original with { RareSv = original.RareSv with { PrAuc = .1 } };
        Require(FineTuning.Improves(original, lower) && !FineTuning.Improves(lower, original), "Fine-tuning must rank by local-model dev S/V PR-AUC.");
        using (var avg = new FedAvg(winner, new Dictionary<string, long> { ["a"] = 1 }, 1))
        { avg.Add("a", 1, 1, winner); avg.Apply(winner); }
        string temp = Path.Combine(Path.GetTempPath(), "aih-fl-check-" + Guid.NewGuid().ToString("N") + ".bin");
        try
        {
            winner.eval();
            using var ng = no_grad(); using var scope = NewDisposeScope();
            var input = zeros(2, 1, 500);
            var before = winner.forward(input);
            winner.save(temp);
            using var restored = Networks.Create(c.Model); restored.load(temp); restored.eval();
            var after = restored.forward(input);
            Require(before.shape.SequenceEqual(new long[] { 2, 4 }) && allclose(before, after), "Checkpoint round-trip failed.");
        }
        finally { if (File.Exists(temp)) File.Delete(temp); }
        Console.WriteLine("Passed: configuration, weighted FedAvg, update rejection, BatchNorm policy, checkpoints, fine-tuning layer freezing, PR-AUC selection/ties, and test gate. No training executed.");
    }
}
