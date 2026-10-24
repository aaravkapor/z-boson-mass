import numpy as np, json, time
from scipy.optimize import curve_fit
rng = np.random.default_rng(2026)
MZ, GZ = 91.1876, 2.4952          # PDG/LEP Breit-Wigner parameters (GeV)
LO, HI = 70.0, 110.0              # analysis window

def gen(n, res=0.015, bkg=0.05, fsr=0.0, shift=0.0):
    """Pseudo-dimuon masses: Cauchy (BW) x Gaussian resolution (+FSR tail) + falling background."""
    ns = int(round(n*(1-bkg))); nb = n-ns
    m = MZ + 0.5*GZ*np.tan(np.pi*(rng.random(ns*2)-0.5))   # BW (non-rel. approx)
    m = m[(m > 50) & (m < 130)][:ns]
    if len(m) < ns:
        m = np.concatenate([m, MZ + 0.5*GZ*np.tan(np.pi*(rng.random(ns)-0.5))])[:ns]
    if fsr > 0:                                            # radiative low-mass tail
        has = rng.random(ns) < fsr
        m = np.where(has, m*(1-rng.exponential(0.02, ns)), m)
    m = m*(1+shift) + rng.normal(0, res*m)                 # detector smearing
    b = LO + rng.exponential(35.0, nb*3)                   # smooth continuum
    b = b[b < HI][:nb]
    return np.concatenate([m, b])

def peak(m, bw=1.0, interp=False):
    edges = np.arange(LO, HI+bw, bw)
    h, e = np.histogram(m, edges)
    i = np.argmax(h); c = 0.5*(e[i]+e[i+1])
    if interp and 0 < i < len(h)-1:
        y0, y1, y2 = h[i-1], h[i], h[i+1]
        d = y0-2*y1+y2
        if d != 0: c += bw*0.5*(y0-y2)/d
    return c

def gfit(m, bw=1.0, k=2.0):
    """Binned least-squares Gaussian fit (+const) in +-k*sigma_guess window around the peak."""
    p0 = peak(m, bw=2.0)
    for _ in range(2):                                     # iterate window
        w = 4.0 if _ == 0 else k*max(sg, 1.0) if False else 4.0
        break
    lo, hi = p0-6.0, p0+6.0
    edges = np.arange(lo, hi+bw, bw); h, e = np.histogram(m, edges); x = 0.5*(e[1:]+e[:-1])
    f = lambda x, A, mu, s, c: A*np.exp(-0.5*((x-mu)/s)**2)+c
    try:
        p, _ = curve_fit(f, x, h, p0=[h.max(), p0, 2.0, 0.0], sigma=np.sqrt(np.maximum(h, 1)), maxfev=5000)
        return p[1]
    except Exception:
        return np.nan

def gml(m, k=1.5):
    """Unbinned truncated-Gaussian ML (iterative sigma-clipped mean) - bin-free reference."""
    mu = peak(m, 2.0); s = 2.0
    for _ in range(10):
        sel = np.abs(m-mu) < max(k*s, 1.5)
        mu, s = m[sel].mean(), max(m[sel].std(), 0.5)
    return mu

def study(n, nexp, **kw):
    out = {"peak1": [], "peak05": [], "peak2": [], "peakI": [], "gfit": [], "gml": []}
    for _ in range(nexp):
        m = gen(n, **kw)
        out["peak05"].append(peak(m, 0.5)); out["peak1"].append(peak(m, 1.0)); out["peak2"].append(peak(m, 2.0))
        out["peakI"].append(peak(m, 1.0, True)); out["gfit"].append(gfit(m)); out["gml"].append(gml(m))
    res = {}
    for k, v in out.items():
        v = np.array(v); v = v[~np.isnan(v)]
        res[k] = dict(bias=float(v.mean()-MZ), sd=float(v.std(ddof=1)), rmse=float(np.sqrt(((v-MZ)**2).mean())), n=int(len(v)))
    return res

if __name__ == "__main__":
    t = time.time(); R = {}
    NEXP = 2000
    for n in [500, 2000, 10000, 50000]:
        R[f"N{n}"] = study(n, NEXP); print("N", n, round(time.time()-t), "s", flush=True)
    R["FSR15"] = study(10000, NEXP, fsr=0.15); print("fsr", flush=True)
    R["BKG20"] = study(10000, NEXP, bkg=0.20); print("bkg", flush=True)
    R["RES01"] = study(10000, NEXP, res=0.010); R["RES025"] = study(10000, NEXP, res=0.025); print("res", flush=True)
    R["SHIFT"] = study(10000, NEXP, shift=-0.001)
    json.dump(R, open("mc_results.json", "w"), indent=1)
    print(json.dumps(R, indent=1))
