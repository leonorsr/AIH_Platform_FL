using TorchSharp;
using static TorchSharp.torch;

namespace Aih.Implementation;

// Full participation, sample-count weighting. BatchNorm floating buffers are
// averaged too; integer num_batches_tracked buffers take the maximum.
public sealed class FedAvg : IDisposable
{
    readonly Dictionary<string, Tensor> sums = new();
    readonly Dictionary<string, (long[] Shape, ScalarType Type)> schema;
    readonly IReadOnlyDictionary<string, long> expected;
    readonly HashSet<string> received = new(StringComparer.Ordinal);
    readonly int round;
    long total;

    public FedAvg(nn.Module<Tensor, Tensor> reference, IReadOnlyDictionary<string, long> expected, int round)
    {
        if (expected.Count == 0 || expected.Values.Any(n => n <= 0)) throw new ArgumentException("Invalid client counts.");
        this.expected = expected;
        this.round = round;
        schema = reference.state_dict().ToDictionary(p => p.Key, p => (p.Value.shape, p.Value.dtype));
        foreach (var (name, entry) in schema)
            if (entry.Type != ScalarType.Float32 && !(entry.Type == ScalarType.Int64 && name.EndsWith("num_batches_tracked", StringComparison.Ordinal)))
                throw new InvalidDataException("Unsupported state tensor: " + name);
    }

    public void Add(string client, int receivedRound, long count, nn.Module<Tensor, Tensor> model)
    {
        if (receivedRound != round || received.Contains(client) || !expected.TryGetValue(client, out long n) || n != count)
            throw new InvalidDataException("Stale, duplicate, unknown, or incorrectly weighted client update.");
        var state = model.state_dict();
        if (!state.Keys.ToHashSet().SetEquals(schema.Keys)) throw new InvalidDataException("State names differ.");
        // Validate the whole update before adding any of its tensors.
        foreach (var (name, t) in state)
        {
            using var scope = NewDisposeScope();
            if (t.dtype != schema[name].Type || !t.shape.SequenceEqual(schema[name].Shape) || !t.isfinite().all().item<bool>())
                throw new InvalidDataException("Invalid state tensor: " + name);
        }
        using var noGrad = no_grad();
        foreach (var (name, t) in state)
        {
            using var scope = NewDisposeScope();
            if (t.dtype == ScalarType.Float32)
            {
                var weighted = t.to(ScalarType.Float64) * count;
                if (sums.TryGetValue(name, out var sum)) sum.add_(weighted);
                else sums.Add(name, weighted.MoveToOuterDisposeScope());
            }
            else
            {
                var next = sums.TryGetValue(name, out var old) ? maximum(old, t) : t.detach().clone();
                if (old is not null) old.Dispose();
                sums[name] = next.MoveToOuterDisposeScope();
            }
        }
        total = checked(total + count);
        received.Add(client);
    }

    public void Apply(nn.Module<Tensor, Tensor> destination)
    {
        if (!received.SetEquals(expected.Keys)) throw new InvalidDataException("All expected clients must complete before aggregation.");
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        var state = destination.state_dict();
        if (!state.Keys.ToHashSet().SetEquals(schema.Keys)
            || state.Any(p => p.Value.dtype != schema[p.Key].Type || !p.Value.shape.SequenceEqual(schema[p.Key].Shape)))
            throw new InvalidDataException("Destination state does not match the aggregation schema.");
        foreach (var (name, t) in state)
        {
            var result = t.dtype == ScalarType.Float32 ? (sums[name] / total).to(t.dtype) : sums[name];
            if (!result.isfinite().all().item<bool>()) throw new InvalidDataException("Non-finite aggregate.");
        }
        foreach (var (name, t) in state)
            t.copy_(t.dtype == ScalarType.Float32 ? (sums[name] / total).to(t.dtype) : sums[name]);
    }

    public void Dispose() { foreach (var t in sums.Values) t.Dispose(); }
}
