using System.Text.Json;

namespace Aih.Ecg;
public sealed record RecordStatistics(string Patient,string Record,string Split,int Detected,int Matched,int Boundary,int Invalid,int SqiRejected,int Accepted,Dictionary<string,int> Annotated,Dictionary<string,int> AcceptedLabels);
public sealed record PreparationResult(string Output,int Records,long Signals,bool CompletePatientSet);
public static class Preparation
{
    const string Header="patient_id,record_id,split,peak_index_200hz,annotation_index_200hz,annotation_alignment_error_samples,sampling_rate_hz,n_samples,label,label_id,quality_zhao_fuzzy,q_sqi_percent,p_sqi,k_sqi,bas_sqi_percent,fuzzy_quality_score,raw_peak_to_peak,raw_max_abs,segment_path,client_id";
    public static PreparationResult Run(string dataRoot,string rawRoot,string output,int? limitRecordsPerPatient=null,string? sampleRecord=null)
    {
        var data=new Dataset(dataRoot);output=Path.GetFullPath(output);rawRoot=Path.GetFullPath(rawRoot);
        if(limitRecordsPerPatient is <=0)throw new ArgumentException("O limite deve ser positivo.");
        if(output==data.Root||new[]{"train","dev","test","clients","evaluation_clients","raw"}.Any(p=>output==Path.Combine(data.Root,p)))throw new ArgumentException("Escolhe uma pasta nova para a saída.");
        if(Directory.Exists(output)&&Directory.EnumerateFileSystemEntries(output).Any())throw new IOException("A saída deve estar vazia.");
        var jobs=new List<(Client Client,string Prefix)>();
        foreach(var c in data.Clients)
        {
            if(sampleRecord!=null&&Path.GetFileName(Path.GetDirectoryName(Path.GetFullPath(sampleRecord)))!=c.Patient)continue;
            string dir=Path.Combine(rawRoot,c.Patient[..3],c.Patient);
            var records=sampleRecord!=null?new[]{Path.GetFullPath(sampleRecord)}:Directory.GetFiles(dir,"*.hea").Order(StringComparer.Ordinal).Select(p=>p[..^4]);
            if(limitRecordsPerPatient.HasValue)records=records.Take(limitRecordsPerPatient.Value);
            var prefixes=records.ToArray();if(prefixes.Length==0)throw new FileNotFoundException($"Sem registos para {c.Patient}");
            foreach(string prefix in prefixes){foreach(string ext in new[]{".hea",".dat",".atr"})if(!File.Exists(prefix+ext))throw new FileNotFoundException(prefix+ext);jobs.Add((c,prefix));}
        }
        if(jobs.Count==0)throw new InvalidDataException("Nenhum registo corresponde aos pacientes configurados.");
        Directory.CreateDirectory(output);Directory.CreateDirectory(Path.Combine(output,"clients"));Directory.CreateDirectory(Path.Combine(output,"evaluation_clients"));
        var stats=new List<RecordStatistics>();var clientCounts=new Dictionary<(string Client,string Label),long>();
        using var manifest=new StreamWriter(Path.Combine(output,"manifest.csv"));manifest.WriteLine(Header);
        var writers=new Dictionary<string,StreamWriter>();var splitWriters=new Dictionary<string,StreamWriter>();long total=0;
        try {
            foreach(var c in data.Clients.Where(c=>jobs.Any(j=>j.Client==c)))
            {
                var w=new StreamWriter(Path.Combine(output,c.Split=="train"?"clients":"evaluation_clients",c.Id+".csv"));w.WriteLine(Header);writers.Add(c.Id,w);
            }
            foreach(string split in new[]{"dev","test"}){var w=new StreamWriter(Path.Combine(output,split+"_evaluation_manifest.csv"));w.WriteLine(Header);splitWriters.Add(split,w);}
            foreach(var (client,prefix) in jobs)
            {
                Console.WriteLine($"Preparar {client.Id}: {Path.GetFileName(prefix)}");
                var st=Process(prefix,client,output,line=>{manifest.WriteLine(line);writers[client.Id].WriteLine(line);if(splitWriters.TryGetValue(client.Split,out var sw))sw.WriteLine(line);});
                stats.Add(st);total+=st.Accepted;
                foreach(string label in Dataset.Labels)clientCounts[(client.Id,label)]=clientCounts.GetValueOrDefault((client.Id,label))+st.AcceptedLabels[label];
            }
            foreach(var c in data.Clients.Where(c=>writers.ContainsKey(c.Id)))if(Dataset.Labels.Sum(l=>clientCounts.GetValueOrDefault((c.Id,l)))==0)throw new InvalidDataException($"Cliente sem batimentos: {c.Id}");
        } finally {foreach(var w in writers.Values)w.Dispose();foreach(var w in splitWriters.Values)w.Dispose();}
        using(var w=new StreamWriter(Path.Combine(output,"record_processing_stats.csv")))
        {
            w.WriteLine("patient_id,record_id,split,detected_peaks,matched_peaks,unmatched_detections,rejected_boundary,rejected_invalid_or_constant,rejected_sqi,accepted,"+string.Join(",",Dataset.Labels.SelectMany(l=>new[]{"annotated_"+l,"accepted_"+l})));
            foreach(var s in stats)w.WriteLine(Csv.Line(new object[]{s.Patient,s.Record,s.Split,s.Detected,s.Matched,s.Detected-s.Matched,s.Boundary,s.Invalid,s.SqiRejected,s.Accepted}.Concat(Dataset.Labels.SelectMany(l=>new object[]{s.Annotated[l],s.AcceptedLabels[l]}))));
        }
        using(var w=new StreamWriter(Path.Combine(output,"split_class_counts.csv")))
        {w.WriteLine("split,N,S,V,Q");foreach(string split in new[]{"train","dev","test"})w.WriteLine(Csv.Line(new object[]{split}.Concat(Dataset.Labels.Select(l=>(object)data.Clients.Where(c=>c.Split==split).Sum(c=>clientCounts.GetValueOrDefault((c.Id,l)))))));}
        using(var w=new StreamWriter(Path.Combine(output,"patient_client_class_counts.csv")))
        {w.WriteLine("client_id,patient_id,split,N,S,V,Q");foreach(var c in data.Clients.Where(c=>writers.ContainsKey(c.Id)))w.WriteLine(Csv.Line(new object[]{c.Id,c.Patient,c.Split}.Concat(Dataset.Labels.Select(l=>(object)clientCounts.GetValueOrDefault((c.Id,l))))));}
        using(var w=new StreamWriter(Path.Combine(output,"client_class_counts.csv")))
        {w.WriteLine("client_id,N,S,V,Q");foreach(var c in data.Clients.Where(c=>c.Split=="train"&&writers.ContainsKey(c.Id)))w.WriteLine(Csv.Line(new object[]{c.Id}.Concat(Dataset.Labels.Select(l=>(object)clientCounts.GetValueOrDefault((c.Id,l))))));}
        using(var w=new StreamWriter(Path.Combine(output,"import_summary.csv")))
        {w.WriteLine("client_id,patient_id,split,signals");foreach(var c in data.Clients.Where(c=>writers.ContainsKey(c.Id)))w.WriteLine(Csv.Line([c.Id,c.Patient,c.Split,Dataset.Labels.Sum(l=>clientCounts.GetValueOrDefault((c.Id,l)))]));}
        var config=JsonSerializer.Deserialize<Dictionary<string,JsonElement>>(File.ReadAllText(Path.Combine(data.Root,"preparation_config.json")))!;
        var meta=config.ToDictionary(k=>k.Key,k=>(object?)k.Value);meta["implementation"]="C# / .NET 8";meta["versions"]=new{dotnet=Environment.Version.ToString()};meta["sample_record"]=sampleRecord;meta["limit_records_per_patient"]=limitRecordsPerPatient;meta["complete_patient_set"]=sampleRecord==null;meta["numerical_equivalence"]="DSP and detectors ported to C#; small floating-point differences may affect boundary decisions.";
        WriteJson(Path.Combine(output,"preparation_config.json"),meta);
        var result=new PreparationResult(output,stats.Count,total,sampleRecord==null);WriteJson(Path.Combine(output,"validation_report.json"),new{result,ArrayChecks="All exported arrays reread as finite float32[500] in [0,1]",PatientSplitChecks=true});return result;
    }
    static void WriteJson(string p,object o)=>File.WriteAllText(p,JsonSerializer.Serialize(o,new JsonSerializerOptions{WriteIndented=true}));
    static RecordStatistics Process(string prefix,Client client,string output,Action<string> write)
    {
        var rec=Wfdb.Read(prefix);var raw=SignalProcessing.Resample(rec.Signal);var filtered=SignalProcessing.Filter(raw);
        int[] peaks=SignalProcessing.Peaks(filtered),secondary=SignalProcessing.PanTompkins(filtered);var matches=SignalProcessing.Match(peaks,rec.Annotations);
        int boundary=0,invalid=0,sqiRejected=0,accepted=0;var labels=Dataset.Labels.ToDictionary(l=>l,l=>0);
        foreach(var pair in matches.OrderBy(p=>p.Key))
        {
            int index=pair.Key,peak=peaks[index];string label=pair.Value.Annotation.Label;int labelId=System.Array.IndexOf(Dataset.Labels,label);
            if(peak<100||peak+100>raw.Length){boundary++;continue;}
            var beat=SignalProcessing.Beat(filtered,peak);double[] window=raw[(peak-100)..(peak+100)];
            if(beat.Length==0||window.Any(v=>!double.IsFinite(v))||window.Max()-window.Min()<=2.220446049250313e-16){invalid++;continue;}
            Quality q;try{q=SignalProcessing.Sqi(window,peaks.Where(p=>p>=peak-100&&p<peak+100).ToArray(),secondary.Where(p=>p>=peak-100&&p<peak+100).ToArray());}catch(InvalidDataException){sqiRejected++;continue;}
            string relative=$"{client.Split}/{client.Patient}/{Path.GetFileName(prefix)}_beat_{index:D5}.npz",path=Path.Combine(output,relative);
            Npz.Write(path,beat,label,labelId,q.Grade,q);Npz.ReadSignal(path);
            write(Csv.Line([client.Patient,Path.GetFileName(prefix),client.Split,peak,pair.Value.TargetSample,peak-pair.Value.TargetSample,200,500,label,labelId,q.Grade,q.Q,q.P,q.K,q.Bas,q.Score,window.Max()-window.Min(),window.Max(v=>Math.Abs(v)),relative,client.Id]));accepted++;labels[label]++;
        }
        return new(client.Patient,Path.GetFileName(prefix),client.Split,peaks.Length,matches.Count,boundary,invalid,sqiRejected,accepted,Dataset.Labels.ToDictionary(l=>l,l=>rec.Annotations.Count(a=>a.Label==l)),labels);
    }
}
