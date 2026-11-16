using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Aih.ModelTests;
public static class Networks
{
    public static readonly string[] Names=["mlp-simple","mlp","mlp-complex","mlp-very-complex","cnn-small","cnn-deep","resnet1d","mobilenetv1-1d-025","mobilenetv1-1d-050","mobilenetv1-1d-075","mobilenetv1-1d-100"];
    public static Module<Tensor,Tensor> Create(string name) => name switch
    {
        "mlp" => Sequential(("flatten",Flatten()),("fc1",Linear(500,128)),("relu1",ReLU()),("drop",Dropout(.2)),("fc2",Linear(128,64)),("relu2",ReLU()),("output",Linear(64,4))),
        "mlp-simple" => Mlp(64),
        "mlp-complex" => Mlp(256,128,64),
        "mlp-very-complex" => Mlp(512,256,128,64),
        "cnn-small" => Sequential(("conv1",Conv1d(1,16,7,padding:3)),("relu1",ReLU()),("pool1",MaxPool1d(2)),("conv2",Conv1d(16,32,5,padding:2)),("relu2",ReLU()),("pool2",new ExactAdaptivePool1d(8)),("flatten",Flatten()),("fc",Linear(256,64)),("relu3",ReLU()),("output",Linear(64,4))),
        "cnn-deep" => Sequential(("conv1",Conv1d(1,16,7,padding:3)),("relu1",ReLU()),("pool1",MaxPool1d(2)),("conv2",Conv1d(16,32,5,padding:2)),("relu2",ReLU()),("pool2",MaxPool1d(2)),("conv3",Conv1d(32,64,3,padding:1)),("relu3",ReLU()),("pool3",new ExactAdaptivePool1d(8)),("flatten",Flatten()),("drop",Dropout(.2)),("fc",Linear(512,64)),("relu4",ReLU()),("output",Linear(64,4))),
        "resnet1d" => Sequential(("stem",Conv1d(1,16,7,padding:3)),("relu1",ReLU()),("block1",new ResidualBlock(16)),("pool1",MaxPool1d(2)),("conv2",Conv1d(16,32,3,padding:1)),("relu2",ReLU()),("block2",new ResidualBlock(32)),("pool2",new ExactAdaptivePool1d(8)),("flatten",Flatten()),("output",Linear(256,4))),
        "mobilenetv1-1d-025" => MobileNetV1.Create(),
        "mobilenetv1-1d-050" => MobileNetV1.Create(.5),
        "mobilenetv1-1d-075" => MobileNetV1.Create(.75),
        "mobilenetv1-1d-100" => MobileNetV1.Create(1),
        _=>throw new ArgumentException("Unknown network: "+name)
    };
    static Module<Tensor,Tensor> Mlp(params int[] widths)
    {
        var layers=new List<(string,Module<Tensor,Tensor>)>{("flatten",Flatten())};int input=500;
        for(int i=0;i<widths.Length;i++)
        {
            layers.Add(("fc"+(i+1),Linear(input,widths[i])));layers.Add(("relu"+(i+1),ReLU()));
            if(i==0)layers.Add(("drop",Dropout(.2)));input=widths[i];
        }
        layers.Add(("output",Linear(input,4)));return Sequential(layers.ToArray());
    }
}
public sealed class ResidualBlock : Module<Tensor,Tensor>
{
    readonly Module<Tensor,Tensor> conv1,conv2,relu;
    public ResidualBlock(int channels):base(nameof(ResidualBlock))
    {
        conv1=Conv1d(channels,channels,3,padding:1);conv2=Conv1d(channels,channels,3,padding:1);relu=ReLU();RegisterComponents();
    }
    public override Tensor forward(Tensor input)
    {
        using var scope=NewDisposeScope();
        return relu.forward(conv2.forward(relu.forward(conv1.forward(input)))+input).MoveToOuterDisposeScope();
    }
}

// Match adaptive average pooling boundaries, including non-divisible input lengths.
// Computing slice means also works on the MPS backend.
public sealed class ExactAdaptivePool1d : Module<Tensor,Tensor>
{
    readonly int bins;
    public ExactAdaptivePool1d(int bins):base(nameof(ExactAdaptivePool1d)){this.bins=bins;}
    public override Tensor forward(Tensor input)
    {
        using var scope=NewDisposeScope();long length=input.shape[2];var pooled=new Tensor[bins];
        for(int i=0;i<bins;i++){long start=i*length/bins,end=((i+1)*length+bins-1)/bins;pooled[i]=input.narrow(2,start,end-start).mean(new long[]{2},keepdim:true);}
        return cat(pooled,dim:2).MoveToOuterDisposeScope();
    }
}
