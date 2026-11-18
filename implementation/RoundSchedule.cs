using Aih.Ecg;
using Aih.ModelTests;

namespace Aih.Implementation;

public sealed record RoundSelection(int Seed, int Round, string[] TrainClients, string DevClient, string TestClient);
public sealed record LocalSource(string Client, int Round, string DevClient, string TestClient, string Checkpoint);

public static class RoundSchedule
{
    public static string[] FixedCohort(IReadOnlyList<Client> clients, Configuration c)
    {
        var ids = clients.Where(x => x.Split == "train").Select(x => x.Id).Order(StringComparer.Ordinal).ToArray();
        new Random(c.ClientSelectionSeed).Shuffle(ids);
        return ids.Take(c.ClientsPerRound).Order(StringComparer.Ordinal).ToArray();
    }
    public static RoundSelection Choose(IReadOnlyList<Client> clients, Configuration c, int seed, int round)
    {
        var train = clients.Where(x => x.Split == "train").Select(x => x.Id).Order(StringComparer.Ordinal).ToArray();
        var dev = clients.Where(x => x.Split == "dev").Select(x => x.Id).Order(StringComparer.Ordinal).ToArray();
        var test = clients.Where(x => x.Split == "test").Select(x => x.Id).Order(StringComparer.Ordinal).ToArray();
        var random = new Random(LocalTraining.Seed(seed, round, 7919));
        if (c.ClientSelectionMode == "fixed") train = FixedCohort(clients, c);
        else { random.Shuffle(train); train = train.Take(c.ClientsPerRound).Order(StringComparer.Ordinal).ToArray(); }
        return new(seed, round, train,
            dev[random.Next(dev.Length)], test[random.Next(test.Length)]);
    }

    public static RoundResult[] History(string run, Configuration c, int seed)
        => Enumerable.Range(1, c.GlobalRounds).Select(round => ModelReports.Read<RoundResult>(
            Path.Combine(run, $"seed-{seed}", $"round-{round:D3}", "metrics.json"))).ToArray();

    public static LocalSource[] LatestLocals(string run, Configuration c, int seed)
        => History(run, c, seed).SelectMany(r => r.Clients.Select(client => new LocalSource(client.Client, r.Round,
                r.Selection!.DevClient, r.Selection.TestClient,
                Path.Combine($"seed-{seed}", $"round-{r.Round:D3}", client.Checkpoint!))))
            .GroupBy(x => x.Client).Select(g => g.OrderByDescending(x => x.Round).First())
            .OrderBy(x => x.Client, StringComparer.Ordinal).ToArray();
}
