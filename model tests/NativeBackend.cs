using System.Runtime.InteropServices;
namespace Aih.ModelTests;
public static class NativeBackend
{
    public static void Initialize()
    {
        // Explicit loading avoids relying on @rpath resolution on macOS.
        if(!OperatingSystem.IsMacOS())return;
        if(RuntimeInformation.ProcessArchitecture!=Architecture.Arm64)throw new PlatformNotSupportedException("This LibTorch package requires Apple Silicon (arm64).");
        string native=Path.Combine(AppContext.BaseDirectory,"runtimes/osx-arm64/native");
        foreach(string name in new[]{"libc10.dylib","libomp.dylib","libtorch_cpu.dylib","libtorch.dylib","libLibTorchSharp.dylib"})
        {
            string path=Path.Combine(native,name);if(File.Exists(path))NativeLibrary.Load(path);
        }
    }
}
