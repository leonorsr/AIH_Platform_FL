using Aih.ModelTests;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Aih.Implementation;

public static class PerformanceChecks
{
    static void Require(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    static void Copy(Module<Tensor, Tensor> source, Module<Tensor, Tensor> target)
    {
        using var ng = no_grad();
        var state = source.state_dict();
        Require(state.Keys.ToHashSet().SetEquals(target.state_dict().Keys), "Optimized state schema differs.");
        foreach (var (name, t) in target.state_dict()) t.copy_(state[name]);
    }
    static void Pair(Module<Tensor, Tensor> reference, Module<Tensor, Tensor> optimized, int channels, int length)
    {
        using (reference) using (optimized) using (var scope = NewDisposeScope())
        {
            Copy(reference, optimized);
            var x = rand(4, channels, length).requires_grad_(true);
            var z = x.detach().clone().requires_grad_(true);
            var a = reference.forward(x); var b = optimized.forward(z);
            Require(allclose(a, b, rtol: 1e-4, atol: 2e-5), "Optimized convolution output differs.");
            var probe = rand_like(a);
            (a * probe).sum().backward(); (b * probe).sum().backward();
            Require(allclose(x.grad!, z.grad!, rtol: 1e-4, atol: 2e-5), "Optimized input gradients differ.");
            var parameters = reference.named_parameters().ToDictionary(p => p.name, p => p.parameter);
            foreach (var (name, p) in optimized.named_parameters())
                Require(allclose(parameters[name].grad!, p.grad!, rtol: 1e-4, atol: 2e-4), "Optimized parameter gradients differ.");
        }
    }
    public static void Run()
    {
        set_num_threads(2); manual_seed(2024);
        foreach (int length in new[] { 31, 32, 500 }) foreach (int stride in new[] { 1, 2 })
            Pair(Conv1d(3, 3, 3, stride: stride, padding: 1, groups: 3, bias: false), new CpuDepthwiseConv1d(3, stride), 3, length);
        foreach (int length in new[] { 31, 32, 500 })
            Pair(Conv1d(3, 7, 1, bias: false), new CpuPointwiseConv1d(3, 7), 3, length);
        using var reference = MobileNetV1.Create(1);
        using var optimized = MobileNetV1.Create(1, optimizedCpu: true);
        Copy(reference, optimized); reference.eval(); optimized.eval();
        using var ng = no_grad(); using var scope = NewDisposeScope();
        var input = rand(4, 1, 500);
        Require(allclose(reference.forward(input), optimized.forward(input), rtol: 1e-4, atol: 2e-5), "Full-model optimized outputs differ.");
        Require(reference.parameters().Sum(p => p.numel()) == optimized.parameters().Sum(p => p.numel()), "Parameter count changed.");
        Console.WriteLine("Passed CPU convolution output, input/weight gradient, state-schema, odd/even length/stride, and full-model equivalence checks.");
    }
}
