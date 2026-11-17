using Aih.ModelTests;
using TorchSharp;
using static TorchSharp.torch;

namespace Aih.Implementation;

public sealed record ClientResult(string Client, long Examples, int Epochs, double FinalLoss, double Seconds,
    string? Checkpoint = null);

public static class LocalTraining
{
    public static int Seed(int seed, int round, int client, int epoch = 0)
        => unchecked((seed * 1000003 + round * 10007 + client * 1009 + epoch * 97) & int.MaxValue);

    // Each example is visited exactly once per epoch. No early stopping,
    // replacement sampling, or test/development data enter local training.
    public static ClientResult Train(nn.Module<Tensor, Tensor> model, LocalData data, Configuration c,
        int seed, int round, int clientIndex, float[]? weights, Action<object> progress, CancellationToken token)
    {
        manual_seed(Seed(seed, round, clientIndex));
        using var optimizer = optim.Adam(model.parameters(), lr: c.LearningRate);
        using var classWeights = weights is null ? null : tensor(weights, dtype: ScalarType.Float32);
        var watch = System.Diagnostics.Stopwatch.StartNew();
        double finalLoss = 0;
        for (int epoch = 1; epoch <= c.LocalEpochs; epoch++)
        {
            token.ThrowIfCancellationRequested();
            model.train();
            int[] order = Enumerable.Range(0, data.Rows.Length).ToArray();
            new Random(Seed(seed, round, clientIndex, epoch)).Shuffle(order);
            double sumLoss = 0, sumWeight = 0;
            for (int start = 0; start < order.Length; start += c.BatchSize)
            {
                token.ThrowIfCancellationRequested();
                using var scope = NewDisposeScope();
                int count = Math.Min(c.BatchSize, order.Length - start);
                var rows = new Example[count];
                for (int i = 0; i < count; i++) rows[i] = data.Rows[order[start + i]];
                var (x, y) = Trainer.Batch(rows, data.Read);
                optimizer.zero_grad();
                var logits = model.forward(x);
                var loss = nn.functional.cross_entropy(logits, y, weight: classWeights);
                double value = loss.item<float>();
                if (!double.IsFinite(value)) throw new InvalidDataException("Non-finite local training loss.");
                loss.backward(); optimizer.step();
                double weight = weights is null ? count : rows.Sum(r => (double)weights[r.Label]);
                sumLoss += value * weight; sumWeight += weight;
                if ((start / c.BatchSize) % 250 == 0 || start + count == order.Length)
                    progress(new { Status = "training", Seed = seed, Round = round, Client = data.Rows[0].Client,
                        Epoch = epoch, c.LocalEpochs, Processed = start + count, Total = order.Length, UpdatedUtc = DateTime.UtcNow });
            }
            finalLoss = sumLoss / sumWeight;
            Console.WriteLine($"Seed {seed}, round {round}/{c.GlobalRounds}, {data.Rows[0].Client}, epoch {epoch}/{c.LocalEpochs}, loss {finalLoss:F6}");
        }
        return new(data.Rows[0].Client, data.Rows.LongLength, c.LocalEpochs, finalLoss, watch.Elapsed.TotalSeconds);
    }
}
