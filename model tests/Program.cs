using Aih.Ecg;
namespace Aih.ModelTests;
public static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            if(args.Length==0||args[0]=="--help")
            {
                Console.WriteLine("ECG neural-network smoke tests in C#\n  run [--config path.json] [--out empty-directory]\n  report --run results-directory\n  self-test\nSelection uses a development sample; the held-out test set is not used.");return 0;
            }
            NativeBackend.Initialize();
            if(args[0]=="benchmark"){Benchmark.Run();return 0;}
            if(args[0]=="self-test"){Checks.Run();return 0;}
            string root=Dataset.FindRoot();var options=new Dictionary<string,string>();
            for(int i=1;i<args.Length;i++){if(!new[]{"--config","--out","--run"}.Contains(args[i])||i+1>=args.Length)throw new ArgumentException("Invalid option.");options.Add(args[i],args[++i]);}
            if(args[0]=="report"){Console.WriteLine(ResultNotebook.Write(root,options.GetValueOrDefault("--run")??throw new ArgumentException("Missing --run.")));}
            else if(args[0]=="resume"){string previous=Path.GetFullPath(options.GetValueOrDefault("--run")??throw new ArgumentException("Missing --run."));Trainer.Run(root,Path.Combine(previous,"config.json"),previous,resume:true);}
            else if(args[0]=="run")Trainer.Run(root,options.GetValueOrDefault("--config",Path.Combine(root,"model tests/configs/smoke.json")),options.GetValueOrDefault("--out"));
            else if(args[0]=="evaluate")Trainer.EvaluateTest(root,options.GetValueOrDefault("--run")??throw new ArgumentException("Missing --run."));
            else throw new ArgumentException("Unknown command.");
            return 0;
        }catch(Exception ex){Console.Error.WriteLine(ex.ToString());return 1;}
    }
}
