using System.Globalization;
using System.Text.Json;
using Aih.Ecg;
using ScottPlot;

namespace Aih.ModelTests;

public static class ResultNotebook
{
    public static string Write(string root,string run)
    {
        run=Path.GetFullPath(run);
        var config=ExperimentConfig.Load(Path.Combine(run,"config.json"));
        var results=File.Exists(Path.Combine(run,"candidates.json"))?Read<CandidateResult[]>(Path.Combine(run,"candidates.json")):[];
        var groups=results.GroupBy(r=>r.Model).Select(g=>new {
            Model=g.Key, Seeds=g.Count(),
            Pr=Mean(g.Select(r=>r.Dev.RareSv.PrAuc)),
            Std=Std(g.Select(r=>r.Dev.RareSv.PrAuc)),
            Roc=Mean(g.Select(r=>r.Dev.RareSv.RocAuc)),
            Miss=Mean(g.Select(r=>r.Dev.RareSv.MissRate)),
            Far=Mean(g.Select(r=>r.Dev.RareSv.Far)),
            Accuracy=g.Average(r=>r.Dev.Accuracy)
        }).OrderByDescending(g=>g.Pr??double.NegativeInfinity).ToArray();
        var cells=new List<object>();
        void Md(string text)=>cells.Add(new {cell_type="markdown",id="summary-"+cells.Count,metadata=new{},source=Lines(text)});
        string partition="The 20 patients retain their fixed partition: **16 for training, 2 for development and 2 for testing**, with no patients shared between splits. The held-out test set was not used.";
        string samplingPath=Path.Combine(run,"sampling.json");
        if(File.Exists(samplingPath))
        {
            using var sampling=JsonDocument.Parse(File.ReadAllText(samplingPath));
            int train=sampling.RootElement.GetProperty("TrainCounts").EnumerateObject().Sum(p=>p.Value.GetInt32());
            int dev=sampling.RootElement.GetProperty("DevCounts").EnumerateObject().Sum(p=>p.Value.GetInt32());
            partition+=$"\n\nThese smoke tests used **{Number(train)} training beats** and **{Number(dev)} development beats**, with the same sample for every network.";
            if(config.SamplingStrategy=="stratified-per-client-class")
                partition+=$" Sampling was stratified by class and patient: up to {config.TrainSamplesPerClient/4} examples per class/patient for training and {config.DevSamplesPerClient/4} for development, without replacement.";
        }
        partition+=$" Each network was trained with **{config.Seeds.Length} seeds and up to {config.Epochs} epochs**. For each seed, the checkpoint with the highest development S/V PR-AUC is used. Stratification changes prevalence; these results are for model screening.";
        Md("# Smoke-test results\n\n## Data partition\n\n"+partition);
        string table="## Results\n\n| Model | S/V PR-AUC (mean ± SD) | ROC-AUC S/V | Miss rate | FAR | Accuracy | Seeds |\n|---|---:|---:|---:|---:|---:|---:|";
        foreach(var g in groups)
            table+=$"\n| {Label(g.Model).Replace("\n"," ")} | {F(g.Pr)} ± {F(g.Std)} | {F(g.Roc)} | {Percent(g.Miss)} | {Percent(g.Far)} | {Percent(g.Accuracy)} | {g.Seeds}/{config.Seeds.Length} |";
        table+="\n\nPR-AUC/ROC-AUC use the score `p(S)+p(V)`. Miss rate and FAR use a threshold of 0.5. SD: standard deviation across seeds.";
        Md(table);
        var defined=groups.Where(g=>g.Pr.HasValue).ToArray();
        if(defined.Length>0)
        {
            string plots=Path.Combine(run,"plots");Directory.CreateDirectory(plots);
            double[] means=defined.Select(g=>g.Pr!.Value).ToArray(),errors=defined.Select(g=>g.Std??0).ToArray();
            double[] xs=Enumerable.Range(0,defined.Length).Select(i=>(double)i).ToArray();
            var plot=new Plot();
            plot.Title("Smoke tests — S/V PR-AUC comparison");
            double maximum=Math.Min(1,Math.Max(.05,means.Zip(errors,(m,s)=>m+s).Max()*1.25));
            if(defined.Length>7)
            {
                double[] positions=xs.Select(x=>defined.Length-1-x).ToArray();
                var bars=means.Select((value,i)=>new Bar{Position=positions[i],Value=value,Error=errors[i]}).ToArray();
                plot.Add.Bars(bars).Horizontal=true;
                plot.Axes.Left.SetTicks(positions,defined.Select(g=>Label(g.Model)).ToArray());
                plot.XLabel("Development PR-AUC (mean ± standard deviation)");plot.Axes.SetLimitsX(0,maximum);plot.Axes.SetLimitsY(-.7,defined.Length-.3);
            }
            else
            {
                plot.Add.Bars(means);plot.Add.ErrorBar(xs,means,errors).Color=Colors.Black;
                plot.Axes.Bottom.SetTicks(xs,defined.Select(g=>Label(g.Model)).ToArray());
                plot.YLabel("Development PR-AUC (mean ± standard deviation)");plot.Axes.SetLimitsY(0,maximum);
            }
            string path=Path.Combine(plots,"pr-auc-comparison.svg");plot.SaveSvg(path,1100,Math.Max(600,defined.Length*54+100));
            Md("## PR-AUC comparison\n\nModels are ordered by decreasing mean; error bars represent standard deviation, not confidence intervals.");
            cells.Add(new {cell_type="code",id="pr-auc-chart",metadata=new {polyglot_notebook=new {kernelName="csharp"}},execution_count=1,
                source=Lines("using System.IO;\ndisplay(File.ReadAllText("+JsonSerializer.Serialize(path)+"), \"image/svg+xml\");"),
                outputs=new[]{new {output_type="display_data",metadata=new{},data=new Dictionary<string,object>{{"image/svg+xml",File.ReadAllText(path)},{"text/plain","Model PR-AUC comparison in smoke tests"}}}}});
        }
        var notebook=new {cells,metadata=new {kernelspec=new {display_name=".NET (C#)",language="C#",name=".net-csharp"},language_info=new {name="C#"},polyglot_notebook=new {kernelInfo=new {defaultKernelName="csharp",items=new[]{new {name="csharp",languageName="C#",aliases=Array.Empty<string>()}}}}},nbformat=4,nbformat_minor=5};
        string output=Path.Combine(root,"model tests/ModelTests_results.ipynb"),temp=output+"."+Guid.NewGuid().ToString("N")+".tmp";
        File.WriteAllText(temp,JsonSerializer.Serialize(notebook,ExperimentConfig.Json)+"\n");File.Move(temp,output,true);return output;
    }
    static T Read<T>(string path)=>JsonSerializer.Deserialize<T>(File.ReadAllText(path),ExperimentConfig.Json)!;
    static string[] Lines(string text)=>text.Split('\n').Select(s=>s+"\n").ToArray();
    static double? Mean(IEnumerable<double?> values){var a=values.Where(v=>v.HasValue).Select(v=>v!.Value).ToArray();return a.Length==0?null:a.Average();}
    static double? Std(IEnumerable<double?> values){var a=values.Where(v=>v.HasValue).Select(v=>v!.Value).ToArray();if(a.Length==0)return null;double mean=a.Average();return Math.Sqrt(a.Select(v=>(v-mean)*(v-mean)).Average());}
    static string F(double? value)=>value?.ToString("F4",CultureInfo.InvariantCulture)??"—";
    static string Percent(double? value)=>value.HasValue?(value.Value*100).ToString("F2",CultureInfo.InvariantCulture)+"%":"—";
    static string Number(int value)=>value.ToString("N0",CultureInfo.InvariantCulture).Replace(',',' ');
    static string Label(string name)=>name switch {
        "mlp-simple"=>"Simple MLP\n64",
        "mlp"=>"Medium MLP\n128–64",
        "mlp-complex"=>"Complex MLP\n256–128–64",
        "mlp-very-complex"=>"Extra-complex MLP\n512–256–128–64",
        "cnn-small"=>"Small CNN","cnn-deep"=>"Deep CNN","resnet1d"=>"ResNet 1D",
        "mobilenetv1-1d-025"=>"MobileNetV1 1D\nα=0.25",
        "mobilenetv1-1d-050"=>"MobileNetV1 1D\nα=0.50",
        "mobilenetv1-1d-075"=>"MobileNetV1 1D\nα=0.75",
        "mobilenetv1-1d-100"=>"MobileNetV1 1D\nα=1.00",_=>name};
}
