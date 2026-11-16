using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Aih.Ecg;

public static class Csv
{
    public static string[] Parse(string line)
    {
        var result = new List<string>(); var field = new StringBuilder(); bool quoted = false;
        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (c == '"') { if (quoted && i + 1 < line.Length && line[i + 1] == '"') { field.Append('"'); i++; } else quoted = !quoted; }
            else if (c == ',' && !quoted) { result.Add(field.ToString()); field.Clear(); }
            else field.Append(c);
        }
        if (quoted) throw new InvalidDataException("Unterminated quoted CSV field.");
        result.Add(field.ToString()); return result.ToArray();
    }
    public static IEnumerable<Dictionary<string,string>> Read(string path)
    {
        using var reader = new StreamReader(path); var header = Parse(reader.ReadLine() ?? throw new InvalidDataException(path));
        while (reader.ReadLine() is { } line)
        {
            var fields = Parse(line); if (fields.Length != header.Length) throw new InvalidDataException($"Invalid CSV: {path}");
            yield return header.Select((h,i) => (h, fields[i])).ToDictionary(x => x.h, x => x.Item2);
        }
    }
    public static string Line(IEnumerable<object?> values) => string.Join(",", values.Select(v => {
        string s = Convert.ToString(v, CultureInfo.InvariantCulture) ?? "";
        return s.IndexOfAny([',','"','\n','\r']) >= 0 ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;
    }));
}

public sealed record Client(string Id, string Patient, string Split);
public sealed class Dataset
{
    public string Root { get; }
    public IReadOnlyList<Client> Clients { get; }
    public static readonly string[] Labels = ["N", "S", "V", "Q"];
    public Dataset(string root)
    {
        Root = Path.GetFullPath(root);
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Root,"preparation_config.json")));
        var config = doc.RootElement; var clients = new List<Client>(); var mapping = config.GetProperty("patient_to_client");
        foreach (string split in new[] {"train","dev","test"})
            foreach (var p in config.GetProperty("patient_ids").GetProperty(split).EnumerateArray())
            { string patient = p.GetString()!; clients.Add(new(mapping.GetProperty(patient).GetString()!, patient, split)); }
        if (clients.Count != 20 || clients.Select(c=>c.Patient).Distinct().Count()!=20 || clients.Select(c=>c.Id).Distinct().Count()!=20
            || clients.Count(c=>c.Split=="train")!=16 || clients.Count(c=>c.Split=="dev")!=2 || clients.Count(c=>c.Split=="test")!=2)
            throw new InvalidDataException("Expected 20 unique clients with a 16/2/2 split.");
        Clients = clients.OrderBy(c=>c.Id, StringComparer.Ordinal).ToArray();
    }
    public static string FindRoot()
    {
        for (var p = new DirectoryInfo(Environment.CurrentDirectory); p != null; p=p.Parent)
            if (File.Exists(Path.Combine(p.FullName,"data","preparation_config.json"))) return p.FullName;
        throw new DirectoryNotFoundException("Run from the repository directory.");
    }
    public string Index(Client c) => Path.Combine(Root, c.Split=="train"?"clients":"evaluation_clients",c.Id+".csv");
    public string SignalPath(string relative)
    {
        string p=Path.GetFullPath(Path.Combine(Root,relative));
        if (!p.StartsWith(Root+Path.DirectorySeparatorChar,StringComparison.Ordinal)) throw new InvalidDataException("Path is outside the dataset directory.");
        return p;
    }
}

// NPY v1/v2 little-endian float32 arrays and Unicode scalars inside NPZ archives.
public static class Npz
{
    public static float[] ReadSignal(string path)
    {
        using var zip=ZipFile.OpenRead(path); var entry=zip.GetEntry("signal.npy") ?? throw new InvalidDataException("Missing signal.npy");
        using var s=entry.Open(); using var r=new BinaryReader(s);
        if (!r.ReadBytes(6).SequenceEqual(new byte[]{147,78,85,77,80,89})) throw new InvalidDataException("Invalid NPY magic bytes.");
        byte major=r.ReadByte(); r.ReadByte(); int len=major==1?r.ReadUInt16():major is 2 or 3?checked((int)r.ReadUInt32()):throw new InvalidDataException("Unsupported NPY version.");
        string h=Encoding.UTF8.GetString(r.ReadBytes(len));
        if (!Regex.IsMatch(h,@"['""]descr['""]\s*:\s*['""]<f4['""]") || !Regex.IsMatch(h,@"['""]fortran_order['""]\s*:\s*False"))
            throw new InvalidDataException("Expected little-endian float32 data in C order.");
        var shape=Regex.Match(h,@"['""]shape['""]\s*:\s*\(\s*(\d+)\s*,\s*\)");
        if (!shape.Success) throw new InvalidDataException("Expected a one-dimensional NPY array.");
        int n=int.Parse(shape.Groups[1].Value); if(n!=500) throw new InvalidDataException($"Expected 500 points; received {n}.");
        var x=new float[n]; for(int i=0;i<n;i++) x[i]=r.ReadSingle();
        if(s.ReadByte()!=-1) throw new InvalidDataException("Unexpected trailing NPY data.");
        if(x.Any(v=>!float.IsFinite(v)||v<0||v>1)) throw new InvalidDataException("Signal contains non-finite values or values outside [0,1].");
        return x;
    }
    static void Array(ZipArchive z,string name,string dtype,string shape,Action<BinaryWriter> body)
    {
        using var s=z.CreateEntry(name+".npy",CompressionLevel.Optimal).Open(); using var w=new BinaryWriter(s);
        string h=$"{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}";
        h+=new string(' ',(64-(10+h.Length+1)%64)%64)+"\n";
        w.Write(new byte[]{147,78,85,77,80,89,1,0}); w.Write((ushort)h.Length); w.Write(Encoding.ASCII.GetBytes(h)); body(w);
    }
    public static void Write(string path,float[] signal,string label,int labelId,string quality,Quality q)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!); string temp=path+".tmp";
        try {
            using(var z=ZipFile.Open(temp,ZipArchiveMode.Create)) {
                Array(z,"signal","<f4",$"({signal.Length},)",w=>{foreach(float v in signal)w.Write(v);});
                Text(z,"label",label); Array(z,"label_id","<i8","()",w=>w.Write((long)labelId)); Text(z,"quality",quality);
                foreach(var (name,value) in new[]{("q_sqi_percent",q.Q), ("p_sqi",q.P), ("k_sqi",q.K), ("bas_sqi_percent",q.Bas), ("fuzzy_quality_score",q.Score)})
                    Array(z,name,"<f8","()",w=>w.Write(value));
            }
            File.Move(temp,path,true);
        } finally { if(File.Exists(temp))File.Delete(temp); }
    }
    static void Text(ZipArchive z,string name,string value)=>Array(z,name,$"<U{value.Length}","()",w=>{foreach(char c in value)w.Write((uint)c);});
}
