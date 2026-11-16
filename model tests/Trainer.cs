using Aih.Ecg;
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Aih.ModelTests;
public sealed record EpochResult(int Epoch,double TrainLoss,Evaluation Dev,double Seconds);
public sealed record CandidateResult(string Model,int Seed,int BestEpoch,long Parameters,Evaluation Dev,string Checkpoint,double Seconds);
public sealed record RankedModel(string Model,double MeanSelectionScore,double SelectionStdDev,int Seeds,double MeanAccuracy,double? MeanSvPrAuc,double? MeanSvRocAuc,double? MeanMissRate,double? MeanFar);
public static class Trainer
{
    public static double Selection(Evaluation e,string metric)=>metric switch {
        "sv-pr-auc"=>e.RareSv.PrAuc??throw new InvalidDataException("The development sample contains no S/V examples; PR-AUC is undefined. Increase the development sample."),
        "macro-pr-auc"=>e.MacroPrAuc??throw new InvalidDataException("Undefined PR-AUC."),"macro-f1"=>e.MacroF1,"accuracy"=>e.Accuracy,_=>throw new ArgumentException("Invalid metric.")};
    public static string Run(string root,string configPath,string? output=null,bool resume=false)
    {
        var c=ExperimentConfig.Load(configPath);set_num_threads(c.Threads);
        if(!c.TrainSamplesPerClient.HasValue||!c.DevSamplesPerClient.HasValue)throw new InvalidOperationException("This comparison is limited to smoke tests. Set sample limits for training and development.");
        string run=Path.GetFullPath(output??Path.Combine(root,"model tests/results",$"{DateTime.UtcNow:yyyyMMdd-HHmmss}-{c.Name}-{Guid.NewGuid().ToString("N")[..6]}"));
        if(!resume&&Directory.Exists(run)&&Directory.EnumerateFileSystemEntries(run).Any())throw new IOException("The results directory must be empty.");
        Directory.CreateDirectory(run);
        if(resume&&Hash(Path.Combine(run,"dataset_config.json"))!=Hash(Path.Combine(root,"data","preparation_config.json")))throw new InvalidDataException("The dataset configuration has changed since this run started.");
        WriteJson(Path.Combine(run,"config.json"),c);
        string dataRoot=Path.Combine(root,"data");
        File.Copy(Path.Combine(dataRoot,"preparation_config.json"),Path.Combine(run,"dataset_config.json"),true);
        WriteJson(Path.Combine(run,"environment.json"),new {Dotnet=Environment.Version.ToString(),OS=System.Runtime.InteropServices.RuntimeInformation.OSDescription,Architecture=System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),TorchSharp=typeof(Tensor).Assembly.GetName().Version?.ToString(),c.Threads,Device=c.Device,ModelDevices=c.ModelDevices,StartedUtc=DateTime.UtcNow,SeedReproducibility="Seeds are recorded; bitwise equality between CPU and MPS backends is not guaranteed."});
        WriteJson(Path.Combine(run,"status.json"),new {Status="preparing-dataset",IsPilot=c.IsPilot,StartedUtc=DateTime.UtcNow,Device=c.Device});
        ResultNotebook.Write(root,run);
        try
        {
            using var samples=new Samples(dataRoot,c,run);
            WriteJson(Path.Combine(run,"sample_hashes.json"),new {Train=Hash(Path.Combine(run,"train_samples.csv")),Dev=Hash(Path.Combine(run,"dev_samples.csv"))});
            // Validate the selection metric before starting training.
            if(c.SelectionMetric=="sv-pr-auc"&&!samples.Dev.Any(r=>r.Label is 1 or 2))throw new InvalidDataException("The development sample contains no S/V examples. Increase DevSamplesPerClient.");
            var results=resume&&File.Exists(Path.Combine(run,"candidates.json"))?JsonSerializer.Deserialize<List<CandidateResult>>(File.ReadAllText(Path.Combine(run,"candidates.json")),ExperimentConfig.Json)!:new List<CandidateResult>();
            foreach(string name in c.Models)foreach(int seed in c.Seeds)
            {
                if(results.Any(r=>r.Model==name&&r.Seed==seed)){Console.WriteLine($"Resume: {name}, seed {seed}, is already complete.");continue;}
                WriteJson(Path.Combine(run,"status.json"),new {Status="training",Model=name,Seed=seed,Completed=results.Count,Expected=c.Models.Length*c.Seeds.Length,Device=c.ModelDevices.GetValueOrDefault(name,c.Device),UpdatedUtc=DateTime.UtcNow});
                ResultNotebook.Write(root,run);
                var result=Train(name,seed,samples,c,run);results.Add(result);WriteJson(Path.Combine(run,"candidates.json"),results);ResultNotebook.Write(root,run);
            }
            var ranked=results.GroupBy(r=>r.Model).Select(g=>{
                double[] score=g.Select(r=>Selection(r.Dev,c.SelectionMetric)).ToArray();double mean=score.Average();
                double? Mean(IEnumerable<double?> values){var a=values.Where(v=>v.HasValue).Select(v=>v!.Value).ToArray();return a.Length==0?null:a.Average();}
                return new RankedModel(g.Key,mean,Math.Sqrt(score.Select(v=>(v-mean)*(v-mean)).Average()),g.Count(),g.Average(r=>r.Dev.Accuracy),Mean(g.Select(r=>r.Dev.RareSv.PrAuc)),Mean(g.Select(r=>r.Dev.RareSv.RocAuc)),Mean(g.Select(r=>r.Dev.RareSv.MissRate)),Mean(g.Select(r=>r.Dev.RareSv.Far)));
            }).OrderByDescending(r=>r.MeanSelectionScore).ThenBy(r=>r.Model,StringComparer.Ordinal).ToArray();
            WriteJson(Path.Combine(run,"ranking.json"),ranked);
            var winner=results.Single(r=>r.Model==ranked[0].Model&&r.Seed==c.Seeds[0]);
            WriteJson(Path.Combine(run,"winner.json"),winner);Report(run,c,ranked,results);
            WriteJson(Path.Combine(run,"status.json"),new {Status="completed",IsPilot=c.IsPilot,SelectionMetric=c.SelectionMetric,Winner=ranked[0].Model,TestRead=false,CompletedUtc=DateTime.UtcNow});
            ResultNotebook.Write(root,run);
            Console.WriteLine($"Comparison complete. Highest-ranked model in this experiment: {ranked[0].Model}; {c.SelectionMetric}={ranked[0].MeanSelectionScore:F6}. Report: {run}");
            return run;
        }catch(Exception ex){WriteJson(Path.Combine(run,"status.json"),new {Status="failed",Error=ex.Message,TimeUtc=DateTime.UtcNow});throw;}
    }
    static CandidateResult Train(string name,int seed,Samples data,ExperimentConfig c,string run)
    {
        string device=c.ModelDevices.GetValueOrDefault(name,c.Device);
        manual_seed(seed);using var model=Networks.Create(name);model.to(new Device(device));using var optimizer=optim.Adam(model.parameters(),lr:c.LearningRate);
        long parameters=model.parameters().Sum(p=>p.numel());
        long[] supports=Enumerable.Range(0,4).Select(i=>data.Train.LongCount(r=>r.Label==i)).ToArray();
        float[]? weightValues=c.ClassWeighting=="inverse-frequency"?supports.Select(n=>n==0?0f:(float)data.Train.Length/(4*n)).ToArray():null;
        using var classWeights=weightValues==null?null:tensor(weightValues,dtype:ScalarType.Float32,device:new Device(device));
        string dir=Path.Combine(run,name,$"seed-{seed}");Directory.CreateDirectory(dir);string checkpoint=Path.Combine(dir,"best.bin");
        Console.WriteLine($"\n{name}, seed {seed}, {parameters:N0} parameters");var watch=Stopwatch.StartNew();double best=double.NegativeInfinity;int stale=0,bestEpoch=0;Evaluation? bestDev=null;
        WriteJson(Path.Combine(dir,"class_weights.json"),new {Mode=c.ClassWeighting,Labels=Dataset.Labels,Supports=supports,Weights=weightValues});
        var history=new List<EpochResult>();
        for(int epoch=1;epoch<=c.Epochs;epoch++)
        {
            var epochWatch=Stopwatch.StartNew();model.train();int[] order=Enumerable.Range(0,data.Train.Length).ToArray();var rng=new Random(unchecked(seed+epoch*100003));rng.Shuffle(order);
            double trainLoss=0,lossWeight=0;
            for(int start=0;start<order.Length;start+=c.BatchSize)
            {
                using var scope=NewDisposeScope();int count=Math.Min(c.BatchSize,order.Length-start);
                var rows=order.Skip(start).Take(count).Select(i=>data.Train[i]).ToArray();var (input,target)=Batch(rows,data.Read,device);
                optimizer.zero_grad();var logits=model.forward(input);var loss=nn.functional.cross_entropy(logits,target,weight:classWeights);double value=loss.cpu().item<float>();
                if(!double.IsFinite(value))throw new InvalidDataException("Non-finite loss.");loss.backward();optimizer.step();double batchWeight=weightValues==null?count:rows.Sum(r=>(double)weightValues[r.Label]);trainLoss+=value*batchWeight;lossWeight+=batchWeight;
                if((start/c.BatchSize)%1000==0||start+count==order.Length)
                {
                    double speed=(start+count)/Math.Max(.01,epochWatch.Elapsed.TotalSeconds);
                    WriteJson(Path.Combine(run,"progress.json"),new {Model=name,Seed=seed,Epoch=epoch,MaxEpochs=c.Epochs,Processed=start+count,Total=order.Length,Device=device,SamplesPerSecond=speed,RemainingSeconds=(order.Length-start-count)/Math.Max(1,speed),UpdatedUtc=DateTime.UtcNow});
                    Console.WriteLine($"    {name} seed {seed} epoch {epoch}: {start+count:N0}/{order.Length:N0}, {speed:F0} examples/s");
                }
            }
            Evaluation dev=Evaluate(model,data.Dev,data.Read,c.BatchSize,c.RareThreshold,device);double selected=Selection(dev,c.SelectionMetric);
            history.Add(new(epoch,trainLoss/lossWeight,dev,epochWatch.Elapsed.TotalSeconds));WriteJson(Path.Combine(dir,"history.json"),history);
            WriteJson(Path.Combine(run,"progress.json"),new {Model=name,Seed=seed,Epoch=epoch,MaxEpochs=c.Epochs,Dev=dev.RareSv,EpochSeconds=epochWatch.Elapsed.TotalSeconds,UpdatedUtc=DateTime.UtcNow});
            ResultNotebook.Write(Path.GetDirectoryName(data.Data.Root)!,run);
            Console.WriteLine($"  epoch {epoch}: loss={trainLoss/lossWeight:F5}, dev accuracy={dev.Accuracy:P2}, PR-AUC S/V={dev.RareSv.PrAuc:F6}, ROC-AUC S/V={dev.RareSv.RocAuc:F6}, miss={dev.RareSv.MissRate:P2}, FAR={dev.RareSv.Far:P2} ({epochWatch.Elapsed.TotalSeconds:F1}s)");
            if(selected>best)
            {
                best=selected;bestEpoch=epoch;bestDev=dev;stale=0;model.save(checkpoint+".tmp");File.Move(checkpoint+".tmp",checkpoint,true);
            }else if(++stale>=c.Patience)break;
        }
        model.load(checkpoint);Evaluation restored=Evaluate(model,data.Dev,data.Read,c.BatchSize,c.RareThreshold,device);
        if(Math.Abs(Selection(restored,c.SelectionMetric)-best)>1e-8)throw new InvalidDataException("The checkpoint does not reproduce its development evaluation.");
        WriteJson(Path.Combine(dir,"dev_by_client.json"),data.Dev.GroupBy(r=>r.Client).ToDictionary(g=>g.Key,g=>Evaluate(model,g.ToArray(),data.Read,c.BatchSize,c.RareThreshold,device)));
        var result=new CandidateResult(name,seed,bestEpoch,parameters,bestDev!,Path.GetRelativePath(run,checkpoint),watch.Elapsed.TotalSeconds);WriteJson(Path.Combine(dir,"result.json"),result);return result;
    }
    public static (Tensor Input,Tensor Target) Batch(Example[] examples,Func<Example,float[]> read,string device="cpu")
    {
        var x=new float[examples.Length*500];var y=new long[examples.Length];
        for(int i=0;i<examples.Length;i++){var a=read(examples[i]);if(a.Length!=500)throw new InvalidDataException("Expected 500 points.");System.Array.Copy(a,0,x,i*500,500);y[i]=examples[i].Label;}
        return (tensor(x,dtype:ScalarType.Float32).to(new Device(device)).reshape(examples.Length,1,500),tensor(y,dtype:ScalarType.Int64).to(new Device(device)));
    }
    public static Evaluation Evaluate(nn.Module<Tensor,Tensor> model,Example[] rows,Func<Example,float[]> read,int batchSize,double threshold,string device="cpu")
    {
        model.eval();using var noGrad=no_grad();var metrics=new Confusion(threshold);
        for(int start=0;start<rows.Length;start+=batchSize)
        {
            using var scope=NewDisposeScope();var batch=rows.Skip(start).Take(Math.Min(batchSize,rows.Length-start)).ToArray();
            var (input,target)=Batch(batch,read,device);var logits=model.forward(input);var loss=nn.functional.cross_entropy(logits,target);var probabilities=logits.softmax(1);
            metrics.Add(batch.Select(r=>(long)r.Label).ToArray(),probabilities.cpu().data<float>().ToArray(),loss.cpu().item<float>());
        }
        return metrics.Result();
    }
    public static void EvaluateTest(string root,string run)
    {
        run=Path.GetFullPath(run);var c=ExperimentConfig.Load(Path.Combine(run,"config.json"));
        if(c.IsPilot)throw new InvalidOperationException("Do not evaluate the test set during a pilot comparison. A completed full-data experiment is required.");
        if(File.Exists(Path.Combine(run,"test_result.json")))throw new IOException("The test set has already been evaluated in this experiment.");
        var status=JsonDocument.Parse(File.ReadAllText(Path.Combine(run,"status.json")));if(status.RootElement.GetProperty("Status").GetString()!="completed")throw new InvalidDataException("Incomplete comparison.");
        var winner=JsonSerializer.Deserialize<CandidateResult>(File.ReadAllText(Path.Combine(run,"winner.json")),ExperimentConfig.Json)!;
        var data=new Dataset(Path.Combine(root,"data"));var rows=Samples.Select(data,"test",null,c.SamplingSeed);set_num_threads(c.Threads);
        string device=c.ModelDevices.GetValueOrDefault(winner.Model,c.Device);
        using var model=Networks.Create(winner.Model);model.to(new Device(device));model.load(Path.Combine(run,winner.Checkpoint));
        var result=Evaluate(model,rows,r=>Npz.ReadSignal(data.SignalPath(r.Path)),c.BatchSize,c.RareThreshold,device);WriteJson(Path.Combine(run,"test_result.json"),new {winner.Model,winner.Seed,Result=result});
        ResultNotebook.Write(root,run);
        Console.WriteLine(JsonSerializer.Serialize(result,ExperimentConfig.Json));
    }
    static string Hash(string p){using var input=File.OpenRead(p);return Convert.ToHexString(SHA256.HashData(input));}
    public static void WriteJson(string p,object value){string temp=p+"."+Guid.NewGuid().ToString("N")+".tmp";File.WriteAllText(temp,JsonSerializer.Serialize(value,ExperimentConfig.Json));File.Move(temp,p,true);}
    static void Report(string run,ExperimentConfig c,RankedModel[] ranking,List<CandidateResult> results)
    {
        string F(double? x)=>x?.ToString("F6",CultureInfo.InvariantCulture)??"undefined";
        using(var writer=new StreamWriter(Path.Combine(run,"ranking.csv")))
        {
            writer.WriteLine("model,seeds,selection_mean,selection_std,accuracy,sv_pr_auc,sv_roc_auc,sv_miss_rate,sv_far");
            foreach(var r in ranking)writer.WriteLine(Csv.Line([r.Model,r.Seeds,r.MeanSelectionScore,r.SelectionStdDev,r.MeanAccuracy,r.MeanSvPrAuc,r.MeanSvRocAuc,r.MeanMissRate,r.MeanFar]));
        }
        using var w=new StreamWriter(Path.Combine(run,"README.md"));w.WriteLine($"# Neural-network comparison — {c.Name}\n\n{(c.IsPilot?"**Limited-sample pilot: this does not establish the best network on the full dataset.**":"Comparison using all training and development examples.")}\n\nSelection by `{c.SelectionMetric}` on development data, averaged over {c.Seeds.Length} seed(s). The test set was not used.\n\n| Network | Selection score | Accuracy | PR-AUC S/V | ROC-AUC S/V | Miss rate S/V | FAR S/V |\n|---|---:|---:|---:|---:|---:|---:|");
        foreach(var r in ranking)w.WriteLine($"| {r.Model} | {F(r.MeanSelectionScore)} | {F(r.MeanAccuracy)} | {F(r.MeanSvPrAuc)} | {F(r.MeanSvRocAuc)} | {F(r.MeanMissRate)} | {F(r.MeanFar)} |");
        w.WriteLine("\nPR-AUC uses trapezoidal integration; Average Precision is reported separately in JSON. S/V uses p(S)+p(V); N/Q are the negative group for this metric without treating Q as normal. Miss/FAR use a fixed threshold of "+c.RareThreshold.ToString(CultureInfo.InvariantCulture)+" for S/V. Per-class metrics use argmax decisions and one-vs-rest AUC. Undefined denominators produce null, not zero.\n\nThe reference checkpoint uses the first seed of the top-ranked architecture; seeds are not selected by their best score. Detailed per-class metrics are stored in each result.json; plotting curves is not required to calculate their areas.");
    }
}
