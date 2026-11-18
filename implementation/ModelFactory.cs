using Aih.ModelTests;
using TorchSharp;

namespace Aih.Implementation;

public static class ModelFactory
{
    public static torch.nn.Module<torch.Tensor, torch.Tensor> Create(Configuration c)
        => MobileNetV1.Create(1, optimizedCpu: c.OptimizedCpuConvolutions);
}
