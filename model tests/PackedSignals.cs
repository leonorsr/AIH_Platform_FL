using Aih.Ecg;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace Aih.ModelTests;
// Cache derived from NPZ files without changing the source data. Each vector occupies exactly 2,000 bytes.
public sealed class PackedSignals : IDisposable
{
    readonly FileStream input;
    public string PathOnDisk {get;}
    public PackedSignals(Dataset data,Example[] train,Example[] dev,string directory)
    {
        Directory.CreateDirectory(directory);var rows=train.Concat(dev).ToArray();
        string fingerprint=data.Root+"|"+File.ReadAllText(Path.Combine(data.Root,"preparation_config.json"))+"|"+string.Join("|",data.Clients.Where(c=>c.Split!="test").Select(c=>{var f=new FileInfo(data.Index(c));return f.FullName+":"+f.Length+":"+f.LastWriteTimeUtc.Ticks;}));
        using(var hash=IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
        {
            hash.AppendData(Encoding.UTF8.GetBytes(fingerprint));
            for(int i=0;i<rows.Length;i++){rows[i].CacheOffset=i*2000L;hash.AppendData(Encoding.UTF8.GetBytes(rows[i].Path+"\n"));}
            PathOnDisk=Path.Combine(directory,Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant()+".f32");
        }
        if(!File.Exists(PathOnDisk))Build(data,rows);
        if(new FileInfo(PathOnDisk).Length!=rows.Length*2000L)throw new InvalidDataException("Incorrect cache size.");
        input=new FileStream(PathOnDisk,FileMode.Open,FileAccess.Read,FileShare.Read,1,FileOptions.RandomAccess);
        foreach(int i in new[]{0,rows.Length/2,rows.Length-1}.Distinct())
            if(!Read(rows[i]).SequenceEqual(Npz.ReadSignal(data.SignalPath(rows[i].Path))))throw new InvalidDataException("Cache does not match the checked NPZ files.");
        Console.WriteLine($"Binary cache ready: {rows.Length:N0} signals, {input.Length/1e9:F2} GB.");
    }
    void Build(Dataset data,Example[] rows)
    {
        string partial=PathOnDisk+".partial",progress=PathOnDisk+".progress";int start=File.Exists(progress)?int.Parse(File.ReadAllText(progress)):0;
        if(!File.Exists(partial))start=0;if(start<0||start>rows.Length)throw new InvalidDataException("Invalid cache progress.");
        using var output=new FileStream(partial,FileMode.OpenOrCreate,FileAccess.Write,FileShare.None,1024*1024,FileOptions.SequentialScan);
        if(output.Length<start*2000L)start=0;output.SetLength(start*2000L);output.Position=start*2000L;
        var watch=Stopwatch.StartNew();int initial=start,lastLog=start;
        for(int pos=start;pos<rows.Length;pos+=512)
        {
            int n=Math.Min(512,rows.Length-pos);var block=new float[n*500];
            Parallel.For(0,n,new ParallelOptions{MaxDegreeOfParallelism=4},j=>{float[] x=Npz.ReadSignal(data.SignalPath(rows[pos+j].Path));System.Array.Copy(x,0,block,j*500,500);});
            output.Write(MemoryMarshal.AsBytes(block.AsSpan()));int done=pos+n;
            if(done-lastLog>=20000||done==rows.Length)
            {
                output.Flush(true);File.WriteAllText(progress+".tmp",done.ToString());File.Move(progress+".tmp",progress,true);
                double speed=(done-initial)/Math.Max(.01,watch.Elapsed.TotalSeconds);
                Console.WriteLine($"Cache: {done:N0}/{rows.Length:N0} ({(double)done/rows.Length:P1}), {speed:F0} signals/s, remaining ~{(rows.Length-done)/Math.Max(1,speed)/60:F1} min");lastLog=done;
            }
        }
        output.Flush(true);output.Dispose();File.Move(partial,PathOnDisk);if(File.Exists(progress))File.Delete(progress);
    }
    public float[] Read(Example row)
    {
        if(row.CacheOffset<0)throw new InvalidDataException("Example is not included in the cache.");var values=new float[500];var bytes=MemoryMarshal.AsBytes(values.AsSpan());int done=0;
        while(done<bytes.Length){int n=RandomAccess.Read(input.SafeFileHandle,bytes[done..],row.CacheOffset+done);if(n==0)throw new EndOfStreamException();done+=n;}
        return values;
    }
    public void Dispose()=>input.Dispose();
}
