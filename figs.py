import numpy as np, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zmc
from zmc import *
R=json.load(open('mc_results.json'))
plt.rcParams.update({'font.size':9,'figure.dpi':200})
# Fig1 example
zmc.rng=np.random.default_rng(7); m=gen(10000)
fig,ax=plt.subplots(figsize=(5.6,3.2))
bw=1.0; e=np.arange(LO,HI+bw,bw); h,_=np.histogram(m,e); x=.5*(e[1:]+e[:-1])
ax.errorbar(x,h,np.sqrt(h),fmt='k.',ms=3,lw=.6,label='pseudo-data (N=10,000)')
pk=peak(m,1.0); g=gfit(m)
ax.axvline(pk,color='tab:red',ls='--',lw=1,label=f'histogram peak = {pk:.2f} GeV')
ax.axvline(g,color='tab:blue',lw=1,label=f'Gaussian fit = {g:.2f} GeV')
ax.axvline(MZ,color='g',ls=':',lw=1.2,label=f'true m$_Z$ = {MZ:.2f} GeV')
ax.set_xlim(80,102); ax.set_xlabel('dimuon invariant mass [GeV]'); ax.set_ylabel('events / 1 GeV'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig('fig1.png')
# Fig2 RMSE vs N
Ns=[500,2000,10000,50000]; fig,ax=plt.subplots(figsize=(5.2,3.2))
for k,l,c in [('peak1','Peak, 1 GeV bins','tab:red'),('peakI','Peak + parabolic interp.','tab:orange'),('gml','Truncated-mean (unbinned)','tab:purple'),('gfit','Gaussian fit (binned)','tab:blue')]:
    ax.plot(Ns,[R[f'N{n}'][k]['rmse'] for n in Ns],'o-',c=c,label=l,ms=4)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('events per pseudo-experiment'); ax.set_ylabel('RMSE of $m_Z$ [GeV]'); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('fig2.png')
# bin-width scan: fixed grid vs random grid offset
zmc.rng=np.random.default_rng(11)
def peak_off(m,bw,off):
    edges=np.arange(LO-off,HI+bw,bw); h,e=np.histogram(m,edges); i=np.argmax(h); return .5*(e[i]+e[i+1])
bws=[0.25,0.5,1,2,4]; fixed=[];rand=[]
data=[gen(10000) for _ in range(600)]
for bw in bws:
    f=np.array([peak(m,bw) for m in data]); r=np.array([peak_off(m,bw,zmc.rng.random()*bw) for m in data])
    fixed.append((f.mean()-MZ,f.std(),np.sqrt(((f-MZ)**2).mean()))); rand.append((r.mean()-MZ,r.std(),np.sqrt(((r-MZ)**2).mean())))
gf=np.array([gfit(m,bw=b) for b in bws for m in data[:200]]).reshape(len(bws),-1)
gfs=[(x.mean()-MZ,x.std(),np.sqrt(((x-MZ)**2).mean())) for x in gf]
json.dump(dict(bws=bws,fixed=fixed,rand=rand,gfit=gfs),open('binscan.json','w'),indent=1)
for i,b in enumerate(bws): print(b,'fixed rmse %.3f bias %+.3f | rand rmse %.3f bias %+.3f sd %.3f | gfit rmse %.3f bias %+.3f'%(fixed[i][2],fixed[i][0],rand[i][2],rand[i][0],rand[i][1],gfs[i][2],gfs[i][0]))
fig,ax=plt.subplots(figsize=(5.2,3.2))
ax.plot(bws,[r[2] for r in fixed],'o-',c='tab:red',label='Peak (fixed bin grid)',ms=4)
ax.plot(bws,[r[2] for r in rand],'s-',c='tab:orange',label='Peak (random grid offset)',ms=4)
ax.plot(bws,[g[2] for g in gfs],'^-',c='tab:blue',label='Gaussian fit',ms=4)
ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.set_xlabel('bin width [GeV]'); ax.set_ylabel('RMSE of $m_Z$ [GeV]'); ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('fig3.png')
# Fig4 systematics bars
sc=[('Nominal','N10000'),('FSR 15%','FSR15'),('Bkg 20%','BKG20'),('Res 1.0%','RES01'),('Res 2.5%','RES025')]
fig,ax=plt.subplots(figsize=(5.2,3.0)); w=.27; xs=np.arange(len(sc))
for j,(k,l,c) in enumerate([('peakI','Peak+interp.','tab:orange'),('gml','Trunc. mean','tab:purple'),('gfit','Gaussian fit','tab:blue')]):
    ax.bar(xs+(j-1)*w,[R[s][k]['bias']*1000 for _,s in sc],w,color=c,label=l)
ax.axhline(0,c='k',lw=.6); ax.set_xticks(xs); ax.set_xticklabels([a for a,_ in sc],fontsize=8); ax.set_ylabel('bias [MeV]'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig('fig4.png')
