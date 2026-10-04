"""Stronger control: estimate the mean, SD and skewness from the 12 units of a single test, fit a reflected lognormal
(matched up to the third moment) and compute the probability of failing. Does it come close to the unit model?
(main text: Results, "Other single-peak distributions", last paragraph).

single_peak_controls.py showed that, given the true skewness, the reflected lognormal nearly recovers Interpretation 1
for the mixed type. In practice only the sample skewness of 12 units is available, and it varies widely (the sample
skewness of 12 units is bounded by |g| <= (n - 2)/sqrt(n - 1) ~ 3.0, and stays below 3.5 after bias correction).
Evaluation as in regime_diagnostic.py: median |log ratio| to the true probability of failing (0.001 added), 150 pulls.
Comparator (the same 12 units, 30-min values only): the normal calculation (mean and SD). For reference, the values of
the unit model fitted to three time points (D2b) are copied from output/regime_diagnostic.txt.
"""
import numpy as np
from scipy.stats import skew
from regime_diagnostic import qmod, verdicts, S_MEAS, T_SPEC, make_types
from single_peak_controls import reflected_lognormal

rng = np.random.default_rng(20261003)
N_REP, N_EST, N_TRUE = 150, 20_000, 200_000
D2B = {("S slow release", 0.05): (0.04, 1.21), ("S slow release", 0.10): (0.13, 1.07), ("S slow release", 0.20): (0.74, 0.55),
       ("O spread opening", 0.05): (0.83, 1.68), ("O spread opening", 0.10): (0.58, 1.31), ("O spread opening", 0.20): (0.26, 0.56),
       ("M mixed", 0.05): (1.49, 1.63), ("M mixed", 0.10): (1.03, 0.97), ("M mixed", 0.20): (0.86, 0.48)}  # D2b in output/regime_diagnostic.txt


def units(th, size):
    mL, sL, mk, sk = th
    L = np.exp(mL + sL * rng.standard_normal(size)); k = np.exp(mk + sk * rng.standard_normal(size))
    return qmod(T_SPEC, L, k) + S_MEAS * rng.standard_normal(size)


def err(est, true):
    return np.abs(np.log((est + 1e-3) / (true + 1e-3)))


if __name__ == "__main__":
    print(f"Estimated from 1 test of 12 units (30-min values): median |log ratio| of the probability of failing [Interpretation 1, Interpretation 2]. {N_REP} pulls")
    print("  type, p                    true             | normal       | reflected lognormal (sample skewness) | unit model, 3 time points (D2b, reference) | sample skewness: median [10-90 %], true skewness")
    for key, th in make_types().items():
        true = verdicts(units(th, (N_TRUE, 24)))
        ref = units(th, 1_000_000); g_true = skew(ref)
        en, er, gs = [], [], []
        for _ in range(N_REP):
            y = units(th, 12)
            m, s, g = y.mean(), y.std(ddof=1), skew(y, bias=False)
            gs.append(g)
            vn = verdicts(m + s * rng.standard_normal((N_EST, 24)))
            rl = reflected_lognormal(m, s, g, (N_EST, 24)) if g < -0.05 else None
            vr = verdicts(rl[0]) if rl is not None else vn
            en.append(err(vn, true)); er.append(err(vr, true))
        en, er, gs = np.median(en, 0), np.median(er, 0), np.array(gs)
        d = D2B[key]
        print(f"  {key[0]} {key[1]:.2f}   [{true[0]:.3f}, {true[1]:.3f}]   | [{en[0]:.2f}, {en[1]:.2f}] | [{er[0]:.2f}, {er[1]:.2f}]              | "
              f"[{d[0]:.2f}, {d[1]:.2f}]                     | {np.median(gs):+.2f} [{np.quantile(gs, .1):+.2f}, {np.quantile(gs, .9):+.2f}], {g_true:+.2f}")
