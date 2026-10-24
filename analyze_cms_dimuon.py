"""
Real-data analysis for the IRIS paper.  Run on YOUR computer (needs internet once to download the CSV).

1. Download a CERN Open Data CSV, e.g.
     "Z to two muons from 2011"  (10,000 events)   https://opendata.cern.ch/record/5208
     "Dimuon events, Run2010B"   (100,000 events)  https://opendata.cern.ch/record/700   (file Mu_Run2010B.csv or similar)
2. python analyze_cms_dimuon.py  <file.csv>  [--lo 70 --hi 110] [--boots 1000]
Needs: numpy scipy pandas matplotlib.
Column handling: uses column 'M' (invariant mass) if present; otherwise computes M from E1,px1,py1,pz1,E2,px2,py2,pz2.
"""
import sys, json, argparse, numpy as np, pandas as pd
from scipy.optimize import curve_fit
ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--lo", type=float, default=70.0)
ap.add_argument("--hi", type=float, default=110.0); ap.add_argument("--boots", type=int, default=1000); a = ap.parse_args()
df = pd.read_csv(a.csv); df.columns = [c.strip() for c in df.columns]
if "M" in df.columns: m = df["M"].to_numpy(float)
else:
    E = df.E1+df.E2; px = df.px1+df.px2; py = df.py1+df.py2; pz = df.pz1+df.pz2
    m = np.sqrt(np.maximum(E**2-px**2-py**2-pz**2, 0)).to_numpy()
m = m[(m > a.lo) & (m < a.hi)]; N = len(m); print("events in window:", N)
rng = np.random.default_rng(1)

def peak(x, bw, off=0.0):
    e = np.arange(a.lo-off, a.hi+bw, bw); h, e = np.histogram(x, e); i = h.argmax(); return 0.5*(e[i]+e[i+1])
def gfit(x, bw=1.0, half=6.0):
    p0 = peak(x, 2.0); e = np.arange(p0-half, p0+half+bw, bw); h, e = np.histogram(x, e); c = 0.5*(e[1:]+e[:-1])
    f = lambda t, A, mu, s, k: A*np.exp(-0.5*((t-mu)/s)**2)+k
    p, cov = curve_fit(f, c, h, p0=[h.max(), p0, 2.0, 0], sigma=np.sqrt(np.maximum(h, 1)), maxfev=5000); return p[1], np.sqrt(cov[1, 1]), abs(p[2])
out = {"N": N}
mu, mu_err, sig = gfit(m); out["gauss_fit"] = dict(mass=mu, stat_err_fit=mu_err, sigma=sig)
for bw in [0.25, 0.5, 1.0, 2.0]:
    out[f"peak_bw{bw}"] = dict(fixed_grid=peak(m, bw), random_offset_mean=float(np.mean([peak(m, bw, rng.random()*bw) for _ in range(200)])))
# bootstrap uncertainties (resample events)
bp = {bw: [] for bw in [0.5, 1.0, 2.0]}; bg = []
for _ in range(a.boots):
    s = rng.choice(m, N)
    for bw in bp: bp[bw].append(peak(s, bw, rng.random()*bw))
    try: bg.append(gfit(s)[0])
    except Exception: pass
out["boot_gauss_sd"] = float(np.std(bg, ddof=1))
for bw in bp: out[f"boot_peak_bw{bw}_sd"] = float(np.std(bp[bw], ddof=1))
# window-size systematic for the Gaussian fit
out["gauss_window_scan"] = {str(h): gfit(m, 1.0, h)[0] for h in [4.0, 5.0, 6.0, 8.0]}
out["PDG_mZ"] = 91.1880; json.dump(out, open("real_data_results.json", "w"), indent=1); print(json.dumps(out, indent=1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
e = np.arange(a.lo, a.hi+1, 1.0); h, e = np.histogram(m, e); c = 0.5*(e[1:]+e[:-1])
plt.errorbar(c, h, np.sqrt(h), fmt="k.", label="CMS Open Data"); plt.axvline(mu, c="b", label=f"Gaussian fit {mu:.2f}")
plt.axvline(peak(m, 1.0), c="r", ls="--", label="peak (1 GeV bins)"); plt.axvline(91.1880, c="g", ls=":", label="PDG 91.188")
plt.xlim(80, 102); plt.xlabel("dimuon mass [GeV]"); plt.ylabel("events / GeV"); plt.legend(); plt.savefig("real_data_fit.png", dpi=200)
