"""Worked example on published per-unit data: the sugar-coated tablets of Nagase et al. (1976), which aged in the
slow-release pattern (main text: "Published data" in Materials and Methods, "Published lots" in Results, Figure 4).

Source: Nagase I, Fujishiro T, Okamoto T, Nakajima S. Disintegration measurement of commercial sugar-coated tablets by
thermal analysis. Japanese Journal of Hospital Pharmacy 1976;1(4):201-207, doi:10.5649/jjphcs1975.1.201 (in Japanese,
freely available on J-STAGE). Figure 8, lower panel: individual dissolution curves of four tablets of one lot of a
commercial sugar-coated product, (A) on opening and (B) after 80 days at 25 degC / 81 % RH in the original strip
package. Dissolution was followed in the stirred cell of a thermal-analysis instrument, not in a pharmacopoeial
apparatus.

Digitisation: the curves were read by eye from the published figure (reading precision about +/- 1 min in time and
+/- 3 % in amount dissolved). For each tablet DATA holds the time at which release begins (onset), the time to 50 %
and the amount dissolved at 20, 30, 40, 50 and 60 min. These are the only measured values used here.

Steps
 1. Fit the unit model to each tablet: Q(t) = 100 (1 - exp(-k (t - L))) for t > L, giving the opening time L_i and the
    rate after opening k_i.
 2. For states A and B, compute the mean and SD of ln L and ln k (four tablets, so the estimates are rough).
 3. Along an ageing path obtained by linear interpolation from A to B (f = 0, 0.1, ..., 1), compare the probability of
    failing the acceptance procedures under the true unit model (ln L and ln k independent normal) with the normal
    calculation (normal distribution with the same mean and SD).
 4. The specification is nominal (not that of this product). Interpretation 2: individual limit R % at t min.
    Interpretation 1: Q = R - 5 (stage 1 at R). t = 20, 30 and 40 min with R = 75 % as the main case; R = 70 and 80 %
    as variations. Analytical error SD 2 %.
 5. Sensitivity: uncertainty of the SDs estimated from four tablets (sigma x 0.5 and x 2), and a correlation of -0.5 or
    +0.5 between ln L and ln k.
"""
import numpy as np
from scipy.optimize import least_squares

rng = np.random.default_rng(20260930)
N_MC = 200_000
S_MEAS = 2.0

# condition, onset_min, t50_min, pct at 20/30/40/50/60 min (values read from Figure 8 of Nagase et al. 1976)
DATA = {
    "A": [(5.5, 11.5, (100, 100, 100, 100, 100)), (7.5, 12.5, (99, 100, 100, 100, 100)),
          (8.5, 14.0, (91, 100, 100, 100, 100)), (9.0, 14.5, (89, 100, 100, 100, 100))],
    "B": [(13.0, 29.0, (19, 52, 76, 94, 99.5)), (14.0, 30.0, (15, 50, 72, 92, 99)),
          (15.5, 35.0, (15, 41, 60, 70, 85)), (20.0, 43.0, (0, 28, 47, 63, 81))],
}
TGRID = (20, 30, 40, 50, 60)


def model(t, L, k):
    t = np.asarray(t, float)
    return np.where(t > L, 100 * (1 - np.exp(-k * np.clip(t - L, 0, None))), 0.0)


def fit_unit(onset, t50, pcts):
    ts = np.array([t50, *TGRID]); ys = np.array([50.0, *pcts])
    def res(p):
        L, lk = p
        return np.r_[model(ts, L, np.exp(lk)) - ys, (L - onset) * 3.0]  # weak penalty tying L to the onset read from the figure
    k0 = np.log(2) / max(t50 - onset, 0.5)
    r = least_squares(res, x0=[onset, np.log(k0)])
    L, k = r.x[0], np.exp(r.x[1])
    rmse = np.sqrt(np.mean((model(ts, L, k) - ys) ** 2))
    return L, k, rmse


def summarize():
    par = {}
    print("1. Fit to each tablet (L in min, k in /min, RMSE of the fit in %)")
    for c, rows in DATA.items():
        Ls, ks = [], []
        for i, (o, t50, p) in enumerate(rows, 1):
            L, k, e = fit_unit(o, t50, p)
            Ls.append(L); ks.append(k)
            print(f"  {c}{i}: L {L:5.1f}  k {k:.3f}  RMSE {e:4.1f}")
        lL, lk = np.log(Ls), np.log(ks)
        par[c] = dict(mL=lL.mean(), sL=lL.std(ddof=1), mk=lk.mean(), sk=lk.std(ddof=1), r=np.corrcoef(lL, lk)[0, 1])
        p = par[c]
        print(f"  {c}: median L {np.exp(p['mL']):.1f} min, sigma_L {p['sL']:.3f}; median k {np.exp(p['mk']):.3f}/min, sigma_k {p['sk']:.3f}; "
              f"correlation {p['r']:+.2f} (4 tablets, for reference only)")
    return par


def unit_q(mL, sL, mk, sk, t, rho, size):
    z1 = rng.standard_normal(size); z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(size)
    L = np.exp(mL + sL * z1); k = np.exp(mk + sk * z2)
    return model(t, L, k) + S_MEAS * rng.standard_normal(size)


def verdicts(q, R):
    Q1 = R - 5
    f6 = (q[:, :6] < R).sum(1); f12 = (q[:, :12] < R).sum(1)
    pass2 = (f6 == 0) | ((f6 <= 2) & (f12 <= 2))
    s1 = (q[:, :6] >= Q1 + 5).all(1)
    s2 = (q[:, :12].mean(1) >= Q1) & (q[:, :12] >= Q1 - 15).all(1)
    s3 = (q.mean(1) >= Q1) & ((q < Q1 - 15).sum(1) <= 2) & (q >= Q1 - 25).all(1)
    return 1 - pass2.mean(), 1 - (s1 | s2 | s3).mean()


def path(par, t, R, sscale=1.0, rho=0.0, fs=np.round(np.arange(0, 1.01, 0.1), 2)):
    A, B = par["A"], par["B"]
    out = []
    for f in fs:
        g = lambda key: A[key] + f * (B[key] - A[key])
        mL, mk = g("mL"), g("mk")
        sL, sk = g("sL") * sscale, g("sk") * sscale
        q = unit_q(mL, sL, mk, sk, t, rho, (N_MC, 24))
        tr = verdicts(q, R)
        qn = q.mean() + q.std() * rng.standard_normal((N_MC, 24))
        nr = verdicts(qn, R)
        out.append((f, q.mean(), q.std(), (q < R).mean(), tr, nr))
    return out


def show(rows, label):
    print(label)
    print("   f    mean    SD  unit p |  Int. 2: true -> normal |  Int. 1: true -> normal")
    for f, m, v, p, tr, nr in rows:
        if tr[0] < 1e-4 and tr[1] < 1e-4 and nr[0] < 1e-4 and nr[1] < 1e-4:
            continue
        print(f"  {f:.1f}  {m:5.1f} {v:5.1f}  {p:.3f}  |  {tr[0]:.3f} -> {nr[0]:.3f}         |  {tr[1]:.3f} -> {nr[1]:.3f}")
    print()


if __name__ == "__main__":
    par = summarize()
    print()
    print("2. Probability of failing along the ageing path (linear interpolation from A to B); nominal specification")
    for t in (20, 30, 40):
        show(path(par, t, 75), f"  Specification at {t} min: 75 % (Interpretation 1: Q = 70)")
    print("3. Sensitivity (30 min, 75 %)")
    show(path(par, 30, 70), "  Limit 70 %")
    show(path(par, 30, 80), "  Limit 80 %")
    show(path(par, 30, 75, sscale=0.5), "  sigma x 0.5")
    show(path(par, 30, 75, sscale=2.0), "  sigma x 2")
    show(path(par, 30, 75, rho=-0.5), "  correlation of ln L and ln k -0.5")
    show(path(par, 30, 75, rho=+0.5), "  correlation of ln L and ln k +0.5")
