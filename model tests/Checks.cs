using TorchSharp;
using static TorchSharp.torch;
namespace Aih.ModelTests;
public static class Checks
{
    static void Check(bool condition,string name){if(!condition)throw new Exception("Test failed: "+name);}
    static void Near(double? a,double b,string name)=>Check(a.HasValue&&Math.Abs(a.Value-b)<1e-9,name);
    public static void Run()
    {
        var perfect=Curves.Compute([true,false,true,false],[.9f,.2f,.8f,.1f]);Near(perfect.PrAuc,1,"perfect PR");Near(perfect.RocAuc,1,"perfect ROC");
        var mixed=Curves.Compute([true,false,true,false],[.9f,.8f,.7f,.1f]);Near(mixed.PrAuc,19.0/24,"PR trapezoidal");Near(mixed.AveragePrecision,5.0/6,"Average Precision");Near(mixed.RocAuc,.75,"mixed ROC");
        var ties=Curves.Compute([true,false],[.5f,.5f]);Near(ties.RocAuc,.5,"ROC ties");Near(ties.AveragePrecision,.5,"AP ties");
        Check(Curves.Compute([false,false],[.9f,.1f]).PrAuc is null,"absent class");
        Check(Curves.Compute([true,true],[.9f,.1f]).RocAuc is null,"no negative examples");
        var cm=new Confusion(.5);cm.Add([0,1,2,3],[.8f,.1f,.05f,.05f,.1f,.7f,.1f,.1f,.6f,.1f,.2f,.1f,.1f,.1f,.1f,.7f],.25);
        var metrics=cm.Result();Near(metrics.Accuracy,.75,"accuracy");Near(metrics.RareSv.MissRate,.5,"miss S/V");Near(metrics.RareSv.Far,0,"FAR S/V");
        var population=Enumerable.Range(0,113).Select(i=>new Example("client_00","p00000","train","sample-"+i,i<100?0:i<102?1:i<103?2:3)).ToArray();
        var selected=Samples.SelectRows(population,16,2024,"stratified-per-client-class");
        Check(selected.Length==11&&selected.Select(r=>r.Path).Distinct().Count()==11,"bounded sampling without replacement");
        Check(selected.Count(r=>r.Label==1)==2&&selected.Count(r=>r.Label==2)==1&&selected.Count(r=>r.Label==0)==4&&selected.Count(r=>r.Label==3)==4,"retain rare classes within quotas");
        Check(selected.SequenceEqual(Samples.SelectRows(population,16,2024,"stratified-per-client-class")),"reproducible sampling");
        try{new ExperimentConfig{Purpose="smoke-test",Epochs=20}.Validate();throw new Exception("Smoke-test limit was ignored.");}catch(ArgumentException){}
        set_num_threads(2);manual_seed(2024);
        long Count(string name){using var model=Networks.Create(name);return model.parameters().Sum(p=>p.numel());}
        long[] mlpSizes=new[]{"mlp-simple","mlp","mlp-complex","mlp-very-complex"}.Select(Count).ToArray();
        Check(mlpSizes.Zip(mlpSizes.Skip(1),(a,b)=>a<b).All(v=>v),"increasing complexity of the four MLP variants");
        long[] mobileSizes=new[]{"mobilenetv1-1d-025","mobilenetv1-1d-050","mobilenetv1-1d-075","mobilenetv1-1d-100"}.Select(Count).ToArray();
        Check(mobileSizes[0]==206468&&mobileSizes.Zip(mobileSizes.Skip(1),(a,b)=>a<b).All(v=>v),"alpha 0.25 compatibility and increasing MobileNetV1 width");
        foreach(int length in new[]{125,250,500})
        {
            using var scope=NewDisposeScope();using var expected=torch.nn.AdaptiveAvgPool1d(8);using var actual=new ExactAdaptivePool1d(8);
            var input=rand(2,3,length);Check(allclose(expected.forward(input),actual.forward(input),rtol:1e-5,atol:1e-6),"equivalent pooling "+length);
        }
        string temp=Path.Combine(Path.GetTempPath(),"aih-model-checks-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(temp);
        try
        {
            foreach(string name in Networks.Names)
            {
                using var scope=NewDisposeScope();using var model=Networks.Create(name);model.train();
                using var optimizer=optim.Adam(model.parameters(),lr:.001);var x=rand(8,1,500);var y=tensor(new long[]{0,1,2,3,0,1,2,3});
                var logits=model.forward(x);Check(logits.shape.SequenceEqual(new long[]{8,4}),name+" shape");
                var loss=nn.functional.cross_entropy(logits,y);Check(double.IsFinite(loss.item<float>()),name+" loss");
                optimizer.zero_grad();loss.backward();optimizer.step();model.eval();
                using var noGrad=no_grad();var before=model.forward(x);string file=Path.Combine(temp,name+".bin");model.save(file);
                using var loaded=Networks.Create(name);loaded.load(file);loaded.eval();var after=loaded.forward(x);Check(allclose(before,after),name+" checkpoint");
                Console.WriteLine($"{name}: forward, backward, parameter update and checkpoint OK");
            }
        }finally{Directory.Delete(temp,true);}
        Console.WriteLine("Metric tests and tests for all "+Networks.Names.Length+" networks passed.");
    }
}
