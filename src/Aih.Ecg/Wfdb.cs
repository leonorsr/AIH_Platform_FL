using System.Globalization;
using System.Text.RegularExpressions;

namespace Aih.Ecg;
public sealed record Annotation(int Sample,string Label);
public sealed record Recording(double[] Signal,int SamplingRate,IReadOnlyList<Annotation> Annotations);
public static class Wfdb
{
    // Icentia11k reader: one channel, WFDB format 16, 250 Hz.
    public static Recording Read(string prefix)
    {
        var lines=File.ReadAllLines(prefix+".hea").Where(l=>!string.IsNullOrWhiteSpace(l)&&!l.StartsWith('#')).ToArray();
        string[] first=Regex.Split(lines[0].Trim(),@"\s+"); string[] ch=Regex.Split(lines[1].Trim(),@"\s+");
        if(first[1]!="1"||first[2]!="250"||ch[1]!="16") throw new NotSupportedException("Expected a single-channel ECG at 250 Hz in WFDB format 16.");
        int count=int.Parse(first[3],CultureInfo.InvariantCulture);
        var gainMatch=Regex.Match(ch[2],@"^(?<gain>[\d.eE+\-]+)(?:\((?<base>[+\-]?\d+)\))?/");
        if(!gainMatch.Success) throw new InvalidDataException("Invalid WFDB gain.");
        double gain=double.Parse(gainMatch.Groups["gain"].Value,CultureInfo.InvariantCulture);
        int baseline=gainMatch.Groups["base"].Success?int.Parse(gainMatch.Groups["base"].Value):int.Parse(ch[4]);
        if(!double.IsFinite(gain)||gain<=0)throw new InvalidDataException("Invalid WFDB gain.");
        string dat=Path.GetFullPath(Path.Combine(Path.GetDirectoryName(prefix)!,ch[0]));
        if(Path.GetDirectoryName(dat)!=Path.GetDirectoryName(Path.GetFullPath(prefix)))throw new InvalidDataException("DAT file is outside the recording directory.");
        using var r=new BinaryReader(File.OpenRead(dat));
        if(r.BaseStream.Length!=2L*count)throw new InvalidDataException("DAT length does not match the header.");
        var x=new double[count];for(int i=0;i<count;i++){short value=r.ReadInt16();if(value==short.MinValue)throw new InvalidDataException("Missing WFDB sample.");x[i]=(value-baseline)/gain;}
        return new(x,250,ReadAnnotations(prefix+".atr"));
    }
    public static List<Annotation> ReadAnnotations(string path)
    {
        using var r=new BinaryReader(File.OpenRead(path));var result=new List<Annotation>();int sample=0;
        while(r.BaseStream.Position<r.BaseStream.Length)
        {
            int word=r.ReadUInt16();if(word==0)break;int type=word>>10,delta=word&1023;
            if(type==59) { int high=r.ReadUInt16(),low=r.ReadUInt16(); sample=checked(sample+((high<<16)|low)); continue; }
            if(type==63){int size=delta+(delta&1);if(r.ReadBytes(size).Length!=size)throw new EndOfStreamException();continue;}
            if(type is 60 or 61 or 62)continue;
            sample=checked(sample+delta);
            string? label=type switch {1=>"N",9=>"S",5=>"V",13=>"Q",_=>null};
            if(label!=null)result.Add(new(sample,label));
        }
        return result;
    }
}
