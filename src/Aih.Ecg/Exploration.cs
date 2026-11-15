using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;

namespace Aih.Ecg;
public sealed record ExplorationResult(string Report,long Signals,int Clients,int MissingExamples);
public static class Exploration
{
    public static ExplorationResult Run(string dataRoot,string output)
    {
        var data=new Dataset(dataRoot);Directory.CreateDirectory(output);
        var counts=new Dictionary<(string Client,string Label),long>();var quality=new Dictionary<(string Split,string Grade),long>();
        var records=new Dictionary<string,HashSet<string>>();var examples=new Dictionary<string,Dictionary<string,string>>();
        foreach(var client in data.Clients)
        {
            records[client.Id]=new(StringComparer.Ordinal);
            foreach(var row in Csv.Read(data.Index(client)))
            {
                if(row["patient_id"]!=client.Patient||row["client_id"]!=client.Id||row["split"]!=client.Split||!Dataset.Labels.Contains(row["label"]))throw new InvalidDataException($"Índice incompatível: {client.Id}");
                if(int.Parse(row["n_samples"])!=500||int.Parse(row["sampling_rate_hz"])!=200||int.Parse(row["label_id"])!=System.Array.IndexOf(Dataset.Labels,row["label"]))throw new InvalidDataException("Metadados de sinal inválidos.");
                var key=(client.Id,row["label"]);counts[key]=counts.GetValueOrDefault(key)+1;
                var qkey=(client.Split,row["quality_zhao_fuzzy"]);quality[qkey]=quality.GetValueOrDefault(qkey)+1;
                records[client.Id].Add(row["record_id"]);
                if(client.Split=="train"&&!examples.ContainsKey(row["label"]))examples.Add(row["label"],row);
            }
        }
        using(var writer=new StreamWriter(Path.Combine(output,"client_class_counts.csv")))
        {
            writer.WriteLine("client_id,patient_id,split,N,S,V,Q,total,records");
            foreach(var c in data.Clients){var n=Dataset.Labels.Select(l=>counts.GetValueOrDefault((c.Id,l))).ToArray();writer.WriteLine(Csv.Line(new object[]{c.Id,c.Patient,c.Split}.Concat(n.Cast<object>()).Append(n.Sum()).Append(records[c.Id].Count)));}
        }
        using(var writer=new StreamWriter(Path.Combine(output,"quality_counts.csv")))
        {writer.WriteLine("split,quality,count");foreach(var q in quality.OrderBy(x=>x.Key.Split).ThenBy(x=>x.Key.Grade))writer.WriteLine(Csv.Line([q.Key.Split,q.Key.Grade,q.Value]));}
        using(var writer=new StreamWriter(Path.Combine(output,"split_class_counts.csv")))
        {
            writer.WriteLine("split,N,S,V,Q,total");foreach(var split in new[]{"train","dev","test"}){var n=Dataset.Labels.Select(l=>data.Clients.Where(c=>c.Split==split).Sum(c=>counts.GetValueOrDefault((c.Id,l)))).ToArray();writer.WriteLine(Csv.Line(new object[]{split}.Concat(n.Cast<object>()).Append(n.Sum())));}
        }
        var html=new StringBuilder("<!doctype html><html lang='pt'><meta charset='utf-8'><title>EDA Icentia11k</title><style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#182332}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}svg{width:100%;height:auto}small{color:#526174}</style><h1>EDA — Icentia11k</h1>");
        long total=counts.Values.Sum();html.Append($"<p>{data.Clients.Count} clientes; {total:N0} batimentos; divisão 16 treino / 2 dev / 2 teste.</p><p>Q significa não classificável. Amplitudes normalizadas por batimento. A análise dos índices não valida integralmente os arrays.</p><h2>Classes por cliente</h2><table><tr><th>Cliente / paciente</th><th>Split</th>");
        foreach(string label in Dataset.Labels)html.Append($"<th>{label}</th>");html.Append("<th>Total</th></tr>");
        foreach(var c in data.Clients){html.Append($"<tr><td>{c.Id} / {c.Patient}</td><td>{c.Split}</td>");foreach(string l in Dataset.Labels)html.Append($"<td>{counts.GetValueOrDefault((c.Id,l)):N0}</td>");html.Append($"<td>{Dataset.Labels.Sum(l=>counts.GetValueOrDefault((c.Id,l))):N0}</td></tr>");}html.Append("</table><h2>Proporções de classes por cliente</h2>");
        string[] colors=["#4c78a8","#54a24b","#e45756","#f58518"];
        html.Append("<p>"+string.Join(" · ",Dataset.Labels.Select((l,i)=>$"<span style='color:{colors[i]}'>{l}</span>"))+"</p><svg viewBox='0 0 1100 650' role='img' aria-label='Proporções de classes por cliente'>");
        for(int i=0;i<data.Clients.Count;i++)
        {
            var c=data.Clients[i];double all=Dataset.Labels.Sum(l=>counts.GetValueOrDefault((c.Id,l)));double left=180;
            html.Append($"<text x='0' y='{i*31+19}' font-size='13'>{c.Id} ({c.Split})</text>");
            for(int j=0;j<4;j++){double fraction=all>0?counts.GetValueOrDefault((c.Id,Dataset.Labels[j]))/all:0;double width=900*fraction;html.Append(FormattableString.Invariant($"<rect x='{left:F4}' y='{i*31+2}' width='{width:F4}' height='22' fill='{colors[j]}'><title>{Dataset.Labels[j]}: {fraction:P3}</title></rect>"));left+=width;}
        }
        html.Append("</svg><h2>Qualidade por split</h2><table><tr><th>Split</th><th>Qualidade</th><th>Batimentos</th></tr>");
        foreach(var q in quality.OrderBy(x=>x.Key.Split).ThenBy(x=>x.Key.Grade))html.Append($"<tr><td>{WebUtility.HtmlEncode(q.Key.Split)}</td><td>{WebUtility.HtmlEncode(q.Key.Grade)}</td><td>{q.Value:N0}</td></tr>");
        html.Append("</table><h2>Exemplos de treino</h2><p>500 pontos interpolados de uma janela de 1 segundo; não são amostras nativas a 200 Hz.</p>");int missing=0;
        foreach(string label in Dataset.Labels)
        {
            if(!examples.TryGetValue(label,out var row)){html.Append($"<p>{label}: sem exemplos.</p>");continue;}
            string path=data.SignalPath(row["segment_path"]);if(!File.Exists(path)){missing++;html.Append($"<p>{label}: ficheiro não encontrado.</p>");continue;}
            float[] signal=Npz.ReadSignal(path);string svg=SignalSvg(signal,$"{label} — {row["client_id"]} — {row["quality_zhao_fuzzy"]}");
            File.WriteAllText(Path.Combine(output,$"example_{label}.svg"),svg);html.Append(svg);
        }
        html.Append("<p><small>Dev e teste são conjuntos reservados. Não usar a exploração do teste para ajustar parâmetros de treino.</small></p></html>");
        string report=Path.GetFullPath(Path.Combine(output,"eda.html"));File.WriteAllText(report,html.ToString());
        var result=new ExplorationResult(report,total,data.Clients.Count,missing);File.WriteAllText(Path.Combine(output,"summary.json"),JsonSerializer.Serialize(result,new JsonSerializerOptions{WriteIndented=true}));return result;
    }
    public static string SignalSvg(float[] x,string title)
    {
        string points=string.Join(" ",x.Select((v,i)=>FormattableString.Invariant($"{60+i*1000.0/(x.Length-1):F2},{190-150*v:F2}")));
        return $"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1100 250' role='img' aria-label='{WebUtility.HtmlEncode(title)}'><title>{WebUtility.HtmlEncode(title)}</title><text x='60' y='22' font-family='sans-serif' font-size='16'>{WebUtility.HtmlEncode(title)}</text><path d='M60 40V190H1060' stroke='#aaa' fill='none'/><polyline points='{points}' fill='none' stroke='#2563eb' stroke-width='1.5'/><text x='10' y='45'>1</text><text x='10' y='190'>0</text><text x='60' y='215'>−0,5 s</text><text x='540' y='215'>0</text><text x='1020' y='215'>+0,5 s</text><text x='400' y='240'>Tempo relativo aproximado ao pico R</text></svg>";
    }
}
