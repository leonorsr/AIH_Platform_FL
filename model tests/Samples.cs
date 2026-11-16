using Aih.Ecg;
using System.Security.Cryptography;
using System.Text;

namespace Aih.ModelTests;
public sealed record Example(string Client,string Patient,string Split,string Path,int Label)
{
    public long CacheOffset {get;set;}=-1;
}
public sealed class Samples : IDisposable
{
    public Dataset Data {get;}
    public Example[] Train {get;}
    public Example[] Dev {get;}
    readonly PackedSignals? packed;
    readonly Dictionary<string,float[]> cache=new(StringComparer.Ordinal);
    public Samples(string dataRoot,ExperimentConfig config,string output)
    {
        Data=new Dataset(dataRoot);
        Train=Select(Data,"train",config.TrainSamplesPerClient,config.SamplingSeed,config.SamplingStrategy);
        Dev=Select(Data,"dev",config.DevSamplesPerClient,config.SamplingSeed+1,config.SamplingStrategy);
        foreach(var split in new[]{Train,Dev})if(split.Length==0)throw new InvalidDataException("Empty split.");
        SaveManifest(Path.Combine(output,"train_samples.csv"),Train);SaveManifest(Path.Combine(output,"dev_samples.csv"),Dev);
        Trainer.WriteJson(Path.Combine(output,"sampling.json"),new {Strategy=config.SamplingStrategy,TrainCounts=Enumerable.Range(0,4).ToDictionary(i=>Dataset.Labels[i],i=>Train.Count(r=>r.Label==i)),DevCounts=Enumerable.Range(0,4).ToDictionary(i=>Dataset.Labels[i],i=>Dev.Count(r=>r.Label==i)),PrAucScope="Selected-sample prevalence; not an estimate for the full population."});
        Console.WriteLine($"Fixed sample: {Train.Length:N0} training / {Dev.Length:N0} development examples. The test set will not be read.");
        if(config.UseDiskCache)packed=new PackedSignals(Data,Train,Dev,Path.Combine(Path.GetDirectoryName(Data.Root)!,"model tests","cache"));
        else if(Train.Length+Dev.Length<=config.CacheSignalsUpTo)
        {
            foreach(var row in Train.Concat(Dev))cache.Add(row.Path,Npz.ReadSignal(Data.SignalPath(row.Path)));
            Console.WriteLine($"In-memory cache: {cache.Count:N0} signals.");
        }
    }
    public void Dispose()=>packed?.Dispose();
    public float[] Read(Example row)=>packed!=null?packed.Read(row):cache.TryGetValue(row.Path,out var signal)?signal:Npz.ReadSignal(Data.SignalPath(row.Path));
    public static Example[] Select(Dataset data,string split,int? limit,int seed,string strategy="uniform-per-client")
    {
        var result=new List<Example>();
        foreach(var client in data.Clients.Where(c=>c.Split==split))
        {
            IEnumerable<Example> ValidatedRows()
            {
                foreach(var row in Csv.Read(data.Index(client)))
                {
                    if(row["client_id"]!=client.Id||row["patient_id"]!=client.Patient||row["split"]!=split||row["n_samples"]!="500"||row["sampling_rate_hz"]!="200")throw new InvalidDataException("Incompatible index: "+client.Id);
                    int label=int.Parse(row["label_id"]);
                    if(label<0||label>3||row["label"]!=Dataset.Labels[label])throw new InvalidDataException("Invalid label.");
                    yield return new Example(client.Id,client.Patient,split,row["segment_path"],label);
                }
            }
            var selected=SelectRows(ValidatedRows(),limit,StableSeed(seed,client.Id),strategy);
            if(selected.Length==0)throw new InvalidDataException("Empty client: "+client.Id);
            result.AddRange(selected.OrderBy(r=>r.Path,StringComparer.Ordinal));
        }
        return result.ToArray();
    }
    public static Example[] SelectRows(IEnumerable<Example> source,int? limit,int seed,string strategy)
    {
        if(strategy is not ("uniform-per-client" or "stratified-per-client-class"))throw new ArgumentException("Unknown sampling strategy.");
        bool stratified=strategy=="stratified-per-client-class";
        if(stratified&&limit is null or <4)throw new ArgumentException("Sample limit is too small for stratification.");
        var buckets=Enumerable.Range(0,stratified?4:1).Select(_=>new List<Example>()).ToArray();var counts=new int[buckets.Length];
        var random=Enumerable.Range(0,buckets.Length).Select(i=>new Random(unchecked(seed+i*104729))).ToArray();
        int? quota=stratified?limit/4:limit;
        foreach(var row in source)
        {
            int k=stratified?row.Label:0;if(k<0||k>=buckets.Length)throw new InvalidDataException("Invalid class.");
            var bucket=buckets[k];counts[k]++;
            if(!quota.HasValue||bucket.Count<quota.Value)bucket.Add(row);
            else {int j=random[k].Next(counts[k]);if(j<quota.Value)bucket[j]=row;}
        }
        return buckets.SelectMany(b=>b).ToArray();
    }
    static int StableSeed(int seed,string client)=>BitConverter.ToInt32(SHA256.HashData(Encoding.UTF8.GetBytes(seed+":"+client)),0)&int.MaxValue;
    public static void SaveManifest(string path,IEnumerable<Example> rows)
    {
        using var writer=new StreamWriter(path);writer.WriteLine("client_id,patient_id,split,segment_path,label_id");
        foreach(var row in rows)writer.WriteLine(Csv.Line([row.Client,row.Patient,row.Split,row.Path,row.Label]));
    }
}
