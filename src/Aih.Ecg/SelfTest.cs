namespace Aih.Ecg;
public static class SelfTest
{
    static void Check(bool ok,string name){if(!ok)throw new InvalidOperationException("Teste falhou: "+name);}
    public static void Run()
    {
        Check(Csv.Parse("a,\"b,c\",\"d\"\"e\"").SequenceEqual(new[]{"a","b,c","d\"e"}),"CSV quoting");
        var constant=SignalProcessing.Resample(Enumerable.Repeat(1.0,2500).ToArray());
        Check(constant.Length==2000&&constant.Skip(30).Take(1900).All(v=>Math.Abs(v-1)<.002),"reamostragem e ganho DC");
        var t=Enumerable.Range(0,4000).Select(i=>Math.Sin(2*Math.PI*10*i/200)+.3*Math.Sin(2*Math.PI*.1*i/200)+.4*Math.Sin(2*Math.PI*60*i/200)).ToArray();
        var f=SignalProcessing.Filter(t);Check(f.All(double.IsFinite),"filtros finitos");
        double Amplitude(double[] x,double hz){double re=0,im=0;for(int i=400;i<3600;i++){re+=x[i]*Math.Cos(2*Math.PI*hz*i/200);im+=x[i]*Math.Sin(2*Math.PI*hz*i/200);}return 2*Math.Sqrt(re*re+im*im)/3200;}
        Check(Amplitude(f,60)<.02&&Amplitude(f,10)>.9,"notch e banda passante");
        var synthetic=new double[2000];for(int peak=200;peak<1800;peak+=200)for(int j=-10;j<=10;j++)synthetic[peak+j]=Math.Exp(-j*j/8.0);
        int[] peaks=SignalProcessing.Peaks(synthetic);Check(peaks.SequenceEqual(new[]{200,400,600,800,1000,1200,1400,1600}),"deteção R sintética");
        var matched=SignalProcessing.Match([100,110,300],[new(125,"N"),new(375,"V")]);Check(matched.Count==2&&matched[0].Annotation.Label=="N"&&matched[2].Annotation.Label=="V","matching 1:1");
        float[] beat=SignalProcessing.Beat(synthetic,200);Check(beat.Length==500&&beat.Min()==0&&beat.Max()==1,"interpolação e normalização");
        Check(SignalProcessing.Beat(synthetic,10).Length==0,"rejeição na fronteira");
        var q=SignalProcessing.Fuzzy(100,.5,8,100);Check(q.Grade=="Excellent","SQI excelente");
        string dir=Path.Combine(Path.GetTempPath(),"aih-selftest-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(dir);
        try
        {
            string npz=Path.Combine(dir,"beat.npz");Npz.Write(npz,beat,"N",0,q.Grade,q);Check(Npz.ReadSignal(npz).SequenceEqual(beat),"NPZ roundtrip");
            // WFDB SKIP usa palavras de 16 bits em ordem alta/baixa, seguidas de anotação.
            string atr=Path.Combine(dir,"sample.atr");using(var w=new BinaryWriter(File.Create(atr))){w.Write((ushort)((59<<10)));w.Write((ushort)0);w.Write((ushort)2000);w.Write((ushort)((1<<10)|5));w.Write((ushort)0);}
            var ann=Wfdb.ReadAnnotations(atr);Check(ann.Count==1&&ann[0].Sample==2005&&ann[0].Label=="N","WFDB SKIP");
        }finally{Directory.Delete(dir,true);}
        Console.WriteLine("Todos os testes passaram: CSV, reamostragem, filtros, picos R, matching, normalização, SQI, NPZ e WFDB.");
    }
}
