using Aih.Ecg;

namespace Aih.Implementation;

// Reuse fixed dev/test metadata and signal caches across models/epochs instead
// of reparsing hundreds of thousands of source CSV rows for each evaluation.
public sealed class EvaluationData : IDisposable
{
    public string Split { get; }
    public List<(Client Client, LocalData Data)> Clients { get; } = new();
    public EvaluationData(Dataset data, string split, Configuration c, string cacheRoot, string? clientId = null)
    {
        Split = split;
        try
        {
            foreach (var client in data.Clients.Where(x => x.Split == split && (clientId == null || x.Id == clientId)))
                Clients.Add((client, new LocalData(data, client, c, cacheRoot)));
            if (Clients.Count == 0) throw new InvalidDataException("No matching evaluation client.");
        }
        catch { Dispose(); throw; }
    }
    public void Dispose() { foreach (var entry in Clients) entry.Data.Dispose(); }
}

public sealed class EvaluationPool : IDisposable
{
    readonly Dataset data; readonly string split, cacheRoot; readonly Configuration configuration;
    readonly Dictionary<string, EvaluationData> cached = new();
    public EvaluationPool(Dataset data, string split, Configuration c, string cacheRoot)
    { this.data = data; this.split = split; configuration = c; this.cacheRoot = cacheRoot; }
    public EvaluationData For(string client)
    {
        if (!cached.TryGetValue(client, out var value))
        { value = new EvaluationData(data, split, configuration, cacheRoot, client); cached.Add(client, value); }
        return value;
    }
    public void Dispose() { foreach (var value in cached.Values) value.Dispose(); }
}
