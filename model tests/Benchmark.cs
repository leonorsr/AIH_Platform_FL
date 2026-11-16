using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch;
namespace Aih.ModelTests;
public static class Benchmark
{
    public static void Run()
    {
        NativeBackend.Initialize();set_num_threads(4);
        foreach(string deviceName in new[]{"cpu","mps"})
        {
            try
            {
                var device=new Device(deviceName);
                foreach(string name in Networks.Names)
                {
                    using var scope=NewDisposeScope();using var model=Networks.Create(name);model.to(device);model.train();
                    using var optimizer=optim.Adam(model.parameters(),lr:.001);var x=rand(256,1,500,device:device);var y=zeros(256,dtype:ScalarType.Int64,device:device);
                    var watch=Stopwatch.StartNew();
                    for(int i=0;i<8;i++){using var s=NewDisposeScope();optimizer.zero_grad();var logits=model.forward(x);var loss=nn.functional.cross_entropy(logits,y);loss.backward();optimizer.step();double v=loss.cpu().item<float>();}
                    Console.WriteLine($"{deviceName} {name}: {8*256/watch.Elapsed.TotalSeconds:F0} exemplos/s");
                }
            }catch(Exception ex){Console.WriteLine($"{deviceName}: unavailable: {ex.Message}");}
        }
    }
}
