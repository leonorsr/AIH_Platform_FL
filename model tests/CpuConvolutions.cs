using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Aih.ModelTests;

// Identical parameters and state-dictionary names to bias-free Conv1d. These
// formulations avoid the slow grouped Conv1d path on the Apple CPU backend.
public sealed class CpuDepthwiseConv1d : Module<Tensor, Tensor>
{
    readonly Parameter weight;
    readonly int stride;
    public CpuDepthwiseConv1d(int channels, int stride) : base(nameof(CpuDepthwiseConv1d))
    {
        if (stride is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(stride));
        this.stride = stride;
        weight = Parameter(empty(channels, 1, 3));
        init.kaiming_uniform_(weight, a: Math.Sqrt(5));
        RegisterComponents();
    }
    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();
        long count = (input.shape[2] - 1) / stride + 1;
        var padded = functional.pad(input, new long[] { 1, 1 });
        Tensor Part(int k) => padded.slice(2, k, k + (count - 1) * stride + 1, stride)
            * weight.select(2, k).reshape(1, -1, 1);
        return (Part(0) + Part(1) + Part(2)).MoveToOuterDisposeScope();
    }
}

public sealed class CpuPointwiseConv1d : Module<Tensor, Tensor>
{
    readonly Parameter weight;
    readonly int inputChannels, outputChannels;
    public CpuPointwiseConv1d(int inputChannels, int outputChannels) : base(nameof(CpuPointwiseConv1d))
    {
        this.inputChannels = inputChannels; this.outputChannels = outputChannels;
        weight = Parameter(empty(outputChannels, inputChannels, 1));
        init.kaiming_uniform_(weight, a: Math.Sqrt(5));
        RegisterComponents();
    }
    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();
        long batch = input.shape[0], length = input.shape[2];
        var rows = input.transpose(1, 2).contiguous().reshape(batch * length, inputChannels);
        var result = rows.mm(weight.squeeze(2).transpose(0, 1));
        return result.reshape(batch, length, outputChannels).transpose(1, 2).contiguous().MoveToOuterDisposeScope();
    }
}
