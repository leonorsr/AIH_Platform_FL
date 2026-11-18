using System.Security.Cryptography;
using Aih.Ecg;
using Aih.ModelTests;

namespace Aih.Implementation;

public sealed record ClientInfo(string Id, string Patient, string Split, long Count, long[] Supports, string IndexHash,
    long SourceCount = 0);

public sealed class LocalData : IDisposable
{
    public Example[] Rows { get; }
    readonly Dataset data;
    readonly PackedSignals? cache;

    public LocalData(Dataset data, Client client, Configuration c, string cacheRoot)
    {
        this.data = data;
        Rows = Select(data, client, c, out _);
        if (Rows.Length == 0) throw new InvalidDataException("Empty client: " + client.Id);
        if (c.UseDiskCache) cache = new PackedSignals(data, Rows, [], Path.Combine(cacheRoot, client.Id));
    }

    public float[] Read(Example row) => cache?.Read(row) ?? Npz.ReadSignal(data.SignalPath(row.Path));
    public void Dispose() => cache?.Dispose();

    public static IEnumerable<Example> ReadRows(Dataset data, Client client)
    {
        foreach (var row in Csv.Read(data.Index(client)))
        {
            if (row["client_id"] != client.Id || row["patient_id"] != client.Patient || row["split"] != client.Split
                || row["n_samples"] != "500" || row["sampling_rate_hz"] != "200") throw new InvalidDataException("Incompatible index: " + client.Id);
            int label = int.Parse(row["label_id"]);
            if (label < 0 || label > 3 || row["label"] != Dataset.Labels[label]) throw new InvalidDataException("Invalid label.");
            data.SignalPath(row["segment_path"]); // Reject paths escaping the dataset.
            yield return new(client.Id, client.Patient, client.Split, row["segment_path"], label);
        }
    }

    public static string Hash(string path)
    {
        using var input = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(input));
    }

    public static Example[] Select(Dataset data, Client client, Configuration c, out long sourceCount)
    {
        int? limit = client.Split == "train" ? c.TrainSamplesPerClient : client.Split == "dev" ? c.DevSamplesPerClient : null;
        long count = 0;
        IEnumerable<Example> Counted()
        {
            foreach (var row in ReadRows(data, client)) { count++; yield return row; }
        }
        int seed = BitConverter.ToInt32(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(c.SamplingSeed + ":" + client.Id)), 0) & int.MaxValue;
        var selected = Samples.SelectRows(Counted(), limit, seed,
            limit.HasValue ? "stratified-per-client-class" : "uniform-per-client");
        sourceCount = count;
        return selected.OrderBy(r => r.Path, StringComparer.Ordinal).ToArray();
    }

    public static ClientInfo Inspect(Dataset data, Client client, Configuration c)
    {
        var supports = new long[4];
        var rows = Select(data, client, c, out long sourceCount);
        foreach (var row in rows) supports[row.Label]++;
        if (supports.Sum() == 0) throw new InvalidDataException("Empty client: " + client.Id);
        return new(client.Id, client.Patient, client.Split, supports.Sum(), supports, Hash(data.Index(client)), sourceCount);
    }
}
