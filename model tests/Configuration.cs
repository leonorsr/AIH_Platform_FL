using System.Text.Json;

namespace Aih.ModelTests;
public sealed class ExperimentConfig
{
    public string Name {get;set;}="pilot";
    public string Purpose {get;set;}="pilot";
    public string SamplingStrategy {get;set;}="uniform-per-client";
    public string[] Models {get;set;}=["mlp","cnn-small","cnn-deep","resnet1d"];
    public int[] Seeds {get;set;}=[2024];
    public int SamplingSeed {get;set;}=2024;
    public int Epochs {get;set;}=5;
    public int BatchSize {get;set;}=64;
    public double LearningRate {get;set;}=.001;
    public int Patience {get;set;}=3;
    public int? TrainSamplesPerClient {get;set;}=512;
    public int? DevSamplesPerClient {get;set;}=1024;
    public int CacheSignalsUpTo {get;set;}=25000;
    public bool UseDiskCache {get;set;}=false;
    public string Device {get;set;}="cpu";
    public Dictionary<string,string> ModelDevices {get;set;}=new();
    public int Threads {get;set;}=4;
    public string SelectionMetric {get;set;}="sv-pr-auc";
    public string ClassWeighting {get;set;}="none";
    public double RareThreshold {get;set;}=.5;
    public bool IsPilot => TrainSamplesPerClient.HasValue || DevSamplesPerClient.HasValue;
    public static readonly JsonSerializerOptions Json=new(){PropertyNameCaseInsensitive=true,WriteIndented=true};
    public static ExperimentConfig Load(string path)
    {
        var c=JsonSerializer.Deserialize<ExperimentConfig>(File.ReadAllText(path),Json)??throw new InvalidDataException("Empty configuration.");c.Validate();return c;
    }
    public void Validate()
    {
        if(SamplingStrategy is not ("uniform-per-client" or "stratified-per-client-class"))throw new ArgumentException("Invalid SamplingStrategy.");
        if(SamplingStrategy=="stratified-per-client-class"&&(TrainSamplesPerClient is null or <4||DevSamplesPerClient is null or <4))throw new ArgumentException("Stratified sampling requires per-client limits of at least 4.");
        if(Purpose=="smoke-test"&&(Epochs>3||TrainSamplesPerClient is null or >2048||DevSamplesPerClient is null or >2048))throw new ArgumentException("Smoke tests allow at most 3 epochs and 2048 examples per client.");
        if(Device is not ("cpu" or "mps"))throw new ArgumentException("Device must be cpu or mps.");
        if(ModelDevices.Any(p=>!Networks.Names.Contains(p.Key)||p.Value is not ("cpu" or "mps")))throw new ArgumentException("Invalid ModelDevices.");
        if(Models.Length==0||Models.Distinct().Count()!=Models.Length||Models.Any(m=>!Networks.Names.Contains(m)))throw new ArgumentException("Invalid or duplicate models.");
        if(Seeds.Length==0||Seeds.Distinct().Count()!=Seeds.Length||Seeds.Any(s=>s<0))throw new ArgumentException("Invalid seeds.");
        if(Epochs<=0||BatchSize<=0||Patience<=0||Threads<=0||TrainSamplesPerClient is <=0||DevSamplesPerClient is <=0||CacheSignalsUpTo<0||!double.IsFinite(RareThreshold)||RareThreshold<0||RareThreshold>1||!double.IsFinite(LearningRate)||LearningRate<=0)throw new ArgumentException("Invalid hyperparameters.");
        if(ClassWeighting is not ("none" or "inverse-frequency"))throw new ArgumentException("Invalid ClassWeighting.");
        if(SelectionMetric is not ("accuracy" or "macro-f1" or "sv-pr-auc" or "macro-pr-auc"))throw new ArgumentException("Invalid SelectionMetric.");
    }
}
