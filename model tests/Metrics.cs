namespace Aih.ModelTests;
public sealed record ClassMetrics(string Label,long Support,long Negatives,double? Precision,double? Recall,double? F1,double? MissRate,double? Far,double? PrAuc,double? AveragePrecision,double? RocAuc);
public sealed record BinaryMetrics(long PositiveSupport,long NegativeSupport,double Threshold,long Tp,long Fp,long Tn,long Fn,double? MissRate,double? Far,double? PrAuc,double? AveragePrecision,double? RocAuc);
public sealed record Evaluation(long Count,double Loss,double Accuracy,double MacroF1,double? MacroPrAuc,double? MacroRocAuc,int DefinedPrClasses,int DefinedRocClasses,long[][] Confusion,ClassMetrics[] Classes,BinaryMetrics RareSv,CurvePoint[]? RareCurve=null);
public sealed record CurvePoint(double Recall,double Precision,double Fpr,double Tpr);
public static class Curves
{
    public static CurvePoint[] Sample(bool[] positive,float[] score,int maximum=400)
    {
        int positives=positive.Count(v=>v),negatives=positive.Length-positives;
        if(positives==0||negatives==0)return [];
        int[] order=Enumerable.Range(0,score.Length).ToArray();System.Array.Sort(order,(a,b)=>score[b].CompareTo(score[a]));
        var points=new List<CurvePoint>{new(0,1,0,0)};double tp=0,fp=0;
        for(int i=0;i<order.Length;)
        {
            float threshold=score[order[i]];int j=i;
            while(j<order.Length&&score[order[j]]==threshold){if(positive[order[j]])tp++;else fp++;j++;}
            points.Add(new(tp/positives,tp/(tp+fp),fp/negatives,tp/positives));i=j;
        }
        if(points.Count<=maximum)return points.ToArray();
        return Enumerable.Range(0,maximum).Select(i=>points[(int)((long)i*(points.Count-1)/(maximum-1))]).ToArray();
    }
    // Sort by score and process all tied scores together.
    public static (double? PrAuc,double? AveragePrecision,double? RocAuc) Compute(bool[] positive,float[] score)
    {
        if(positive.Length!=score.Length||score.Any(v=>!float.IsFinite(v)))throw new InvalidDataException("Invalid scores.");
        int positives=positive.Count(v=>v),negatives=positive.Length-positives;
        if(positives==0)return (null,null,null);
        int[] order=Enumerable.Range(0,score.Length).ToArray();System.Array.Sort(order,(a,b)=>score[b].CompareTo(score[a]));
        double tp=0,fp=0,lastRecall=0,lastPrecision=1,lastFpr=0,lastTpr=0,pr=0,ap=0,roc=0;
        for(int i=0;i<order.Length;)
        {
            float threshold=score[order[i]];int j=i;
            while(j<order.Length&&score[order[j]]==threshold){if(positive[order[j]])tp++;else fp++;j++;}
            double recall=tp/positives,precision=tp/(tp+fp),fpr=negatives==0?0:fp/negatives;
            pr+=(recall-lastRecall)*(precision+lastPrecision)/2;ap+=(recall-lastRecall)*precision;
            roc+=(fpr-lastFpr)*(recall+lastTpr)/2;
            lastRecall=recall;lastPrecision=precision;lastFpr=fpr;lastTpr=recall;i=j;
        }
        return (pr,ap,negatives==0?null:roc);
    }
}
public sealed class Confusion
{
    public readonly long[][] Matrix=Enumerable.Range(0,4).Select(_=>new long[4]).ToArray();
    readonly List<int> labels=[];
    readonly List<float>[] scores=Enumerable.Range(0,4).Select(_=>new List<float>()).ToArray();
    readonly double rareThreshold;double lossSum;long count;
    public Confusion(double rareThreshold=.5){this.rareThreshold=rareThreshold;}
    public void Add(long[] truth,float[] probabilities,double meanLoss)
    {
        if(probabilities.Length!=truth.Length*4||!double.IsFinite(meanLoss))throw new InvalidDataException("Invalid metrics.");
        for(int i=0;i<truth.Length;i++)
        {
            if(truth[i]<0||truth[i]>=4)throw new InvalidDataException("Invalid class.");
            int predicted=0;double sum=0;
            for(int c=0;c<4;c++)
            {
                float p=probabilities[i*4+c];if(!float.IsFinite(p)||p<0||p>1)throw new InvalidDataException("Invalid probability.");
                scores[c].Add(p);sum+=p;if(p>probabilities[i*4+predicted])predicted=c;
            }
            if(Math.Abs(sum-1)>1e-4)throw new InvalidDataException("Probabilities do not sum to 1.");
            Matrix[truth[i]][predicted]++;labels.Add((int)truth[i]);
        }
        lossSum+=meanLoss*truth.Length;count+=truth.Length;
    }
    static double? Ratio(long a,long b)=>b==0?null:(double)a/b;
    public Evaluation Result()
    {
        if(count==0)throw new InvalidDataException("Empty evaluation set.");
        var classes=Enumerable.Range(0,4).Select(i=>{
            long tp=Matrix[i][i],support=Matrix[i].Sum(),predicted=Matrix.Sum(row=>row[i]),fn=support-tp,fp=predicted-tp,tn=count-support-fp;
            var curve=Curves.Compute(labels.Select(l=>l==i).ToArray(),scores[i].ToArray());
            return new ClassMetrics(Aih.Ecg.Dataset.Labels[i],support,count-support,Ratio(tp,predicted),Ratio(tp,support),Ratio(2*tp,2*tp+fp+fn),Ratio(fn,support),Ratio(fp,fp+tn),curve.PrAuc,curve.AveragePrecision,curve.RocAuc);
        }).ToArray();
        var rareLabels=labels.Select(l=>l is 1 or 2).ToArray();var rareScore=Enumerable.Range(0,labels.Count).Select(i=>scores[1][i]+scores[2][i]).ToArray();
        var rareCurve=Curves.Compute(rareLabels,rareScore);long rtp=0,rfp=0,rtn=0,rfn=0;
        for(int i=0;i<labels.Count;i++){bool predicted=rareScore[i]>=rareThreshold;if(rareLabels[i]){if(predicted)rtp++;else rfn++;}else{if(predicted)rfp++;else rtn++;}}
        double? Mean(IEnumerable<double?> xs){var x=xs.Where(v=>v.HasValue).Select(v=>v!.Value).ToArray();return x.Length==0?null:x.Average();}
        return new(count,lossSum/count,(double)Enumerable.Range(0,4).Sum(i=>Matrix[i][i])/count,classes.Average(c=>c.F1??0),Mean(classes.Select(c=>c.PrAuc)),Mean(classes.Select(c=>c.RocAuc)),classes.Count(c=>c.PrAuc.HasValue),classes.Count(c=>c.RocAuc.HasValue),Matrix,classes,new(rtp+rfn,rfp+rtn,rareThreshold,rtp,rfp,rtn,rfn,Ratio(rfn,rtp+rfn),Ratio(rfp,rfp+rtn),rareCurve.PrAuc,rareCurve.AveragePrecision,rareCurve.RocAuc),Curves.Sample(rareLabels,rareScore));
    }
}
