import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import zmc; from zmc import *
zmc.rng=np.random.default_rng(99); r=zmc.rng
def pk(m,bw,off):
    e=np.arange(LO-off,HI+bw,bw); h,e=np.histogram(m,e); i=h.argmax(); return .5*(e[i]+e[i+1])
out={}
for n in [500,2000,10000,50000]:
    v={'peak':[], 'peakI':[], 'gml':[], 'gfit':[]}
    for _ in range(2000):
        m=gen(n); v['peak'].append(pk(m,1.0,r.random())); v['peakI'].append(peak(m,1.0,True)); v['gml'].append(gml(m)); v['gfit'].append(gfit(m))
    out[n]={k:dict(bias=float(np.nanmean(x)-MZ),sd=float(np.nanstd(x,ddof=1)),rmse=float(np.sqrt(np.nanmean((np.array(x)-MZ)**2)))) for k,x in v.items()}
    print(n,{k:(round(s['bias'],3),round(s['sd'],3),round(s['rmse'],3)) for k,s in out[n].items()},flush=True)
json.dump(out,open('nscan.json','w'),indent=1)
Ns=list(out); fig,ax=plt.subplots(figsize=(5.2,3.2))
for k,l,c in [('peak','Histogram peak (1 GeV bins)','tab:red'),('peakI','Peak + parabolic interpolation','tab:orange'),('gml','Truncated mean (unbinned)','tab:purple'),('gfit','Gaussian fit (binned)','tab:blue')]:
    ax.plot(Ns,[out[n][k]['rmse'] for n in Ns],'o-',c=c,label=l,ms=4)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('events per pseudo-experiment'); ax.set_ylabel('RMSE of $m_Z$ [GeV]'); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('fig2.png',dpi=200)
