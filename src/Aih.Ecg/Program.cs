using System.Text.Json;

namespace Aih.Ecg;
public static class Program
{
    public static int Main(string[] args)
    {
        try {
            if(args.Length==0||args[0] is "--help" or "help")
            {
                Console.WriteLine("AIH ECG — C#\n  eda [--data data] [--out artifacts/eda]\n  prepare --raw data/raw [--out data/reprepared] [--limit 1] [--record /path/pXXXXX_s00]\n  validate [--data data] [--all]\n  self-test\nRelative paths are resolved from the current directory.");return 0;
            }
            if(args[0]=="self-test"){SelfTest.Run();return 0;}
            var options=new Dictionary<string,string>();bool all=false;
            for(int i=1;i<args.Length;i++)
            {
                if(args[i]=="--all"){all=true;continue;}
                if(!new[]{"--data","--out","--raw","--limit","--record"}.Contains(args[i])||i+1>=args.Length)throw new ArgumentException($"Invalid option: {args[i]}");
                options.Add(args[i],args[++i]);
            }
            string data=options.GetValueOrDefault("--data",Path.Combine(Dataset.FindRoot(),"data"));object result;
            switch(args[0])
            {
                case "eda":result=Exploration.Run(data,options.GetValueOrDefault("--out",Path.Combine(Dataset.FindRoot(),"artifacts/eda")));break;
                case "prepare":
                    string raw=options.GetValueOrDefault("--raw")??Environment.GetEnvironmentVariable("ICENTIA_RAW_DIR")??Path.Combine(data,"raw");
                    result=Preparation.Run(data,raw,options.GetValueOrDefault("--out",Path.Combine(data,"reprepared")),options.TryGetValue("--limit",out var limit)?int.Parse(limit):null,options.GetValueOrDefault("--record"));break;
                case "validate":result=Validation.Run(data,all);break;
                default:throw new ArgumentException("Unknown command; use --help.");
            }
            Console.WriteLine(JsonSerializer.Serialize(result,new JsonSerializerOptions{WriteIndented=true}));return 0;
        }catch(Exception ex){Console.Error.WriteLine(ex.Message);return 1;}
    }
}
public static class Validation
{
    public static object Run(string root,bool allArrays=false)
    {
        var data=new Dataset(root);long total=0,arrays=0;var missing=new List<string>();
        foreach(var c in data.Clients)
        {
            long n=0;var seen=new HashSet<string>(StringComparer.Ordinal);
            foreach(var row in Csv.Read(data.Index(c)))
            {
                if(row["client_id"]!=c.Id||row["patient_id"]!=c.Patient||row["split"]!=c.Split)throw new InvalidDataException($"Incorrect partition: {c.Id}");
                if(!Dataset.Labels.Contains(row["label"])||int.Parse(row["label_id"])!=System.Array.IndexOf(Dataset.Labels,row["label"])||row["n_samples"]!="500"||row["sampling_rate_hz"]!="200")throw new InvalidDataException("Invalid metadata.");
                string path=data.SignalPath(row["segment_path"]);
                if(!row["segment_path"].StartsWith(c.Split+"/"+c.Patient+"/",StringComparison.Ordinal)||!seen.Add(path))throw new InvalidDataException("Duplicate path or patient mismatch.");
                if(allArrays||n==0)
                {
                    if(!File.Exists(path)){if(missing.Count<20)missing.Add(path);}else{Npz.ReadSignal(path);arrays++;}
                }
                n++;
            }
            string dir=Path.Combine(data.Root,c.Split,c.Patient);
            long local=Directory.Exists(dir)?Directory.EnumerateFiles(dir,"*.npz").LongCount():0;
            if(local!=n)throw new InvalidDataException($"{c.Id}: {n} index entries and {local} files.");
            total+=n;Console.WriteLine($"{c.Id}: {n:N0} signals");
        }
        if(missing.Count>0)throw new FileNotFoundException(string.Join("\n",missing));
        return new {Clients=data.Clients.Count,Signals=total,ArraysChecked=arrays,AllArrays=allArrays};
    }
}
