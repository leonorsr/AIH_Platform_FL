namespace Aih.Ecg;

public sealed record Quality(double Q,double P,double K,double Bas,double Score,string Grade);
public static class SignalProcessing
{
    // Kaiser FIR, 4/5 resampling, zero padding and group-delay compensation (50 samples at the upsampled rate).
    public static double[] Resample(double[] x)
    {
        int n=(x.Length*4+4)/5;var y=new double[n];var h=Coefficients.ResampleFir;
        for(int m=0;m<n;m++)
        {
            int center=m*5+50, first=Math.Max(0,(center-100+3)/4),last=Math.Min(x.Length-1,center/4);
            double sum=0;for(int j=first;j<=last;j++)sum+=x[j]*h[center-4*j];y[m]=sum;
        }
        return y;
    }
    public static double[] Filter(double[] x)=>FiltFilt(FiltFilt(x,Coefficients.HighB,Coefficients.HighA,Coefficients.HighZi),Coefficients.NotchB,Coefficients.NotchA,Coefficients.NotchZi);
    public static double[] FiltFilt(double[] x,double[] b,double[] a,double[] zi)
    {
        int edge=3*Math.Max(a.Length,b.Length);if(x.Length<=edge)throw new ArgumentException("Signal is too short for forward-backward filtering.");
        var padded=new double[x.Length+2*edge];
        for(int i=0;i<edge;i++){padded[i]=2*x[0]-x[edge-i];padded[edge+x.Length+i]=2*x[^1]-x[x.Length-2-i];}
        System.Array.Copy(x,0,padded,edge,x.Length);
        var y=LFilter(padded,b,a,zi,padded[0]);System.Array.Reverse(y);
        y=LFilter(y,b,a,zi,y[0]);System.Array.Reverse(y);return y[edge..(edge+x.Length)];
    }
    static double[] LFilter(double[] x,double[] b,double[] a,double[] zi,double initial)
    {
        var z=zi.Select(v=>v*initial).ToArray();var y=new double[x.Length];
        for(int i=0;i<x.Length;i++)
        {
            double v=b[0]*x[i]+z[0]; y[i]=v;
            for(int j=0;j<z.Length-1;j++)z[j]=b[j+1]*x[i]+z[j+1]-a[j+1]*v;
            z[^1]=b[^1]*x[i]-a[^1]*v;
        }
        return y;
    }
    static double[] Smooth(double[] x,int size)
    {
        var y=new double[x.Length];double sum=0;int left=size/2,right=(size-1)/2;
        for(int j=-left;j<=right;j++)sum+=x[Math.Clamp(j,0,x.Length-1)];
        for(int i=0;i<x.Length;i++) {y[i]=sum/size;sum-=x[Math.Clamp(i-left,0,x.Length-1)];sum+=x[Math.Clamp(i+right+1,0,x.Length-1)];}
        return y;
    }
    static List<int> Maxima(double[] x,int start,int end,bool plateaus=true)
    {
        var peaks=new List<int>();
        for(int i=start+1;i<end-1;i++)
        {
            if(x[i]<=x[i-1])continue;int right=i;
            while(right+1<end&&x[right+1]==x[i])right++;
            if(right+1<end&&x[right]>x[right+1]&&(plateaus||right==i))peaks.Add((i+right)/2);
            i=right;
        }
        return peaks;
    }
    public static int[] Peaks(double[] x)
    {
        var grad=new double[x.Length];
        for(int i=0;i<x.Length;i++)grad[i]=Math.Abs(i==0?x[1]-x[0]:i==x.Length-1?x[^1]-x[^2]:(x[i+1]-x[i-1])/2);
        var smooth=Smooth(grad,20);var average=Smooth(smooth,150);var starts=new List<int>();var ends=new List<int>();
        for(int i=0;i<x.Length-1;i++)
        {
            bool a=smooth[i]>1.5*average[i],b=smooth[i+1]>1.5*average[i+1];if(!a&&b)starts.Add(i);if(a&&!b)ends.Add(i);
        }
        if(starts.Count==0)return [];
        ends=ends.Where(e=>e>starts[0]).ToList();int n=Math.Min(starts.Count,ends.Count);if(n==0)return [];
        double minLength=Enumerable.Range(0,n).Average(i=>ends[i]-starts[i])*.4;var peaks=new List<int>();int last=0;
        for(int j=0;j<n;j++)
        {
            int begin=starts[j],end=ends[j];if(end-begin<minLength)continue;
            double best=double.NegativeInfinity;int chosen=-1;
            foreach(int p in Maxima(x,begin,end))
            {
                double l=x[p],r=x[p];
                for(int k=p;k>=begin&&x[k]<=x[p];k--)l=Math.Min(l,x[k]);
                for(int k=p;k<end&&x[k]<=x[p];k++)r=Math.Min(r,x[k]);
                double prominence=x[p]-Math.Max(l,r);if(prominence>best){best=prominence;chosen=p;}
            }
            if(chosen>=0&&chosen-last>60){peaks.Add(chosen);last=chosen;}
        }
        return peaks.ToArray();
    }
    public static int[] PanTompkins(double[] x)
    {
        var detection=new double[x.Length-1];double sum=0;
        for(int i=0;i<detection.Length;i++)
        {
            double d=x[i+1]-x[i];sum+=d*d;
            if(i>=24){double old=x[i-23]-x[i-24];sum-=old*old;}
            detection[i]=i<40?0:sum/Math.Min(i+1,24);
        }
        var candidates=Maxima(detection,0,detection.Length,false);var result=new List<int>();double spki=0,npki=0;int last=0,lastIndex=-1;
        for(int i=0;i<candidates.Count;i++)
        {
            int p=candidates[i];double value=detection[p],threshold=npki+.25*(spki-npki);
            if(value>threshold&&p>last+60)
            {
                result.Add(p);
                if(result.Count>9)
                {
                    int rr=(result[^2]-result[^10])/8;
                    if(p-last>(int)(1.66*rr))
                    {
                        int missed=-1;
                        for(int j=lastIndex+1;j<i;j++) {int q=candidates[j];if(q>last+50&&q<p-50&&detection[q]>.5*threshold&&(missed<0||detection[q]>detection[missed]))missed=q;}
                        if(missed>=0){result[^1]=missed;result.Add(p);}
                    }
                }
                last=p;lastIndex=i;spki=.125*value+.875*spki;
            }else npki=.125*value+.875*npki;
        }
        return result.ToArray();
    }
    public static float[] Beat(double[] x,int peak)
    {
        if(peak<100||peak+100>x.Length)return [];
        var y=new double[500];for(int i=0;i<500;i++){double pos=i*199.0/499;int a=(int)pos;int b=Math.Min(199,a+1);y[i]=x[peak-100+a]*(1-(pos-a))+x[peak-100+b]*(pos-a);}
        double min=y.Min(),range=y.Max()-min;if(!double.IsFinite(range)||range<=0)return [];
        return y.Select(v=>(float)((v-min)/range)).ToArray();
    }
    public static Dictionary<int,(Annotation Annotation,int TargetSample)> Match(int[] peaks,IReadOnlyList<Annotation> annotations)
    {
        var ann=annotations.Select(a=>(A:a,Sample:(int)Math.Round(a.Sample*.8))).OrderBy(a=>a.Sample).ToArray();
        var candidates=new List<(int Error,int Peak,int Ann)>();int left=0;
        for(int i=0;i<peaks.Length;i++)
        {
            while(left<ann.Length&&ann[left].Sample<peaks[i]-20)left++;
            for(int j=left;j<ann.Length&&ann[j].Sample<=peaks[i]+20;j++)candidates.Add((Math.Abs(peaks[i]-ann[j].Sample),i,j));
        }
        var matches=new Dictionary<int,(Annotation,int)>();var used=new HashSet<int>();
        foreach(var c in candidates.OrderBy(c=>c.Error).ThenBy(c=>c.Peak).ThenBy(c=>c.Ann))
            if(!matches.ContainsKey(c.Peak)&&used.Add(c.Ann))matches.Add(c.Peak,(ann[c.Ann].A,ann[c.Ann].Sample));
        return matches;
    }
    // Precompute the Hann periodogram basis for fixed 200-sample windows.
    static readonly double[,] Cos=new double[101,200],Sin=new double[101,200];
    static SignalProcessing()
    {
        for(int k=0;k<=100;k++)for(int j=0;j<200;j++){double w=.5-.5*Math.Cos(2*Math.PI*j/200);Cos[k,j]=w*Math.Cos(2*Math.PI*k*j/200);Sin[k,j]=w*Math.Sin(2*Math.PI*k*j/200);}
    }
    public static Quality Sqi(double[] window,int[] primary,int[] secondary)
    {
        if(window.Length!=200||window.Any(v=>!double.IsFinite(v)))throw new InvalidDataException("Invalid SQI window.");
        int i=0,j=0,matches=0;
        while(i<primary.Length&&j<secondary.Length){int d=primary[i]-secondary[j];if(Math.Abs(d)<=10){matches++;i++;j++;}else if(d<0)i++;else j++;}
        double q=primary.Length+secondary.Length==0?0:200.0*matches/(primary.Length+secondary.Length);
        var power=new double[101];
        for(int k=0;k<=100;k++) {double re=0,im=0;for(int t=0;t<200;t++){re+=window[t]*Cos[k,t];im+=window[t]*Sin[k,t];}power[k]=(re*re+im*im)*(k is 0 or 100?1:2);}
        double Band(int a,int b){double s=0;for(int k=a;k<b;k++)s+=(power[k]+power[k+1])/2;return s;}
        double denom=Band(5,40),total=Band(0,40),p=denom>double.Epsilon?Band(5,15)/denom:0,bas=total>double.Epsilon?100*(1-Band(0,1)/total):0;
        double mean=window.Average(),m2=0,m4=0;foreach(double v in window){double d=v-mean;m2+=d*d;m4+=d*d*d*d;}
        double kurt=window.Length*m4/(m2*m2);if(!double.IsFinite(kurt))throw new InvalidDataException("Non-finite kurtosis.");
        return Fuzzy(q,p,kurt,bas);
    }
    public static Quality Fuzzy(double q,double p,double k,double bas)
    {
        q=Math.Clamp(q,0,100);p=Math.Clamp(p,0,1);bas=Math.Clamp(bas,0,100);
        double Sq(double v)=>v*v;
        double qe=q<=80?0:q<90?1/(1+1/Sq(.3*(q-80))):1;
        double qu=q<=55?1:1/(1+Sq((q-55)/5)),qb=1/(1+Sq((q-75)/7.5));
        double pe=p<=.25?0:p<.35?10*(p-.25):1,pu=p<.15?1:p<=.25?10*(.25-p):0;
        double pb=p<.18?0:p<.22?25*(p-.18):p<.28?1:p<.32?25*(.32-p):0;
        double be=bas<=90?0:bas<95?1/(1+1/Sq(.8718*(bas-90))):1;
        double bu=bas<=85?1:1/(1+Sq((bas-85)/5)),bb=1/(1+Sq((bas-92)/2.5));
        double e=Math.Min(1,.4*qe+.4*pe+.1*(k>5?1:0)+.1*be),b=Math.Min(1,.4*qb+.4*pb+.1*bb),u=Math.Min(1,.4*qu+.4*pu+.1*(k>5?0:1)+.1*bu);
        double score=(e*e+2*b*b+3*u*u)/(e*e+b*b+u*u);
        if(!double.IsFinite(score))throw new InvalidDataException("Non-finite SQI.");
        return new(q,p,k,bas,score,score<=1.5?"Excellent":score<2.4?"Barely acceptable":"Unacceptable");
    }
}
