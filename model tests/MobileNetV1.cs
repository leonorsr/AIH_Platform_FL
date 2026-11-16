using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Aih.ModelTests;

// MobileNetV1 adapted from 2D to ECG 1D with an alpha width multiplier.
// Retains 13 depthwise/pointwise blocks, BatchNorm and ReLU; does not use V2 inverted residual blocks.
public static class MobileNetV1
{
    public static Module<Tensor,Tensor> Create(double alpha=.25)
    {
        if(alpha is not (.25 or .5 or .75 or 1))throw new ArgumentOutOfRangeException(nameof(alpha));
        int Channels(int value)=>(int)(value*alpha);
        int stem=Channels(32);
        var layers=new List<(string,Module<Tensor,Tensor>)>{
            ("stem",Conv1d(1,stem,3,stride:2,padding:1,bias:false)),
            ("stem_bn",BatchNorm1d(stem)),("stem_relu",ReLU())};
        int input=stem,index=0;
        foreach(var (baseOutput,stride) in new (int,int)[]{(64,1),(128,2),(128,1),(256,2),(256,1),(512,2),(512,1),(512,1),(512,1),(512,1),(512,1),(1024,2),(1024,1)})
        {
            int output=Channels(baseOutput);
            string p="block"+(++index);
            layers.Add((p+"_dw",Conv1d(input,input,3,stride:stride,padding:1,groups:input,bias:false)));
            layers.Add((p+"_dw_bn",BatchNorm1d(input)));layers.Add((p+"_dw_relu",ReLU()));
            layers.Add((p+"_pw",Conv1d(input,output,1,bias:false)));
            layers.Add((p+"_pw_bn",BatchNorm1d(output)));layers.Add((p+"_pw_relu",ReLU()));input=output;
        }
        layers.Add(("global_pool",new ExactAdaptivePool1d(1)));layers.Add(("flatten",Flatten()));layers.Add(("output",Linear(input,4)));
        return Sequential(layers.ToArray());
    }
}
