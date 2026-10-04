"""Gradual start: when the release of a unit starts along an S-shaped curve, how far off is the probability of failing
estimated by fitting the unit model (first-order from the moment the unit opens)? (Main text: Methods, one sentence on
the Weibull shape; Supplementary Section S5.)

A coat cracks and the crack widens gradually, so release may begin slowly (the crack stays narrow for a while). The true
curve of one unit is
  D(t) = 100 {1 - exp(-(k (t - L))^beta)}   (t > L),
built with beta = 1 (the same shape as the unit model; control), 1.5 and 2. L and k are lognormal with the same sigmas as
the three ageing types of regime_diagnostic.py (S slow release, O spread opening, M mixed); the medians are recalibrated
for each beta so that the fraction p of units below R is 0.05 or 0.10. One test is one pull of 12 units at 10, 20 and
30 min with an analytical error of SD 2 %. The fit is the unit model (beta = 1, likelihood.fit_single). Reported for the
probability of failing (Interpretation 1 and 2): median |log ratio|, proportion within a factor of two, proportion below
half; the normal calculation (from mean and SD) alongside. 40 tests per state. All values are assumed; no measured data.
Run with the argument "check" for a quick two-test run of the first state.
"""
import sys
import numpy as np
from scipy.optimize import brentq
import likelihood as LK
import regime_diagnostic as RD

S, TS, R2, TIMES = 2.0, 30.0, 75.0, (10.0, 20.0, 30.0)
NREP, N_TRUE, N_EST = 40, 200_000, 20_000
TYPES = {"S slow release": (0.13, 0.28, "k"), "O spread opening": (0.45, 0.10, "L"), "M mixed": (0.30, 0.20, "both")}


def Dw(t, L, k, beta):
    u = np.clip(t - L, 0, None)
    return np.where(t > L, 100 * (1 - np.exp(-(k * u) ** beta)), 0.0)


def th_of(which, x, sL, sk):
    if which == "k":
        return (np.log(8.0), sL, x, sk)
    if which == "L":
        return (x, sL, np.log(0.25), sk)
    return (np.log(8.0) + 0.6 * x, sL, np.log(0.25) - x, sk)


def units(th, n, beta, gen, times=TIMES):
    mL, sL, mk, sk = th
    L = np.exp(mL + sL * gen.standard_normal(n)); k = np.exp(mk + sk * gen.standard_normal(n))
    return np.stack([Dw(t, L, k, beta) + S * gen.standard_normal(n) for t in times], -1)


def p_unit(th, beta):
    g = np.random.default_rng(99)
    return (units(th, 200_000, beta, g, (TS,))[:, 0] < R2).mean()


def oc_true(th, beta, gen):
    v = units(th, N_TRUE * 24, beta, gen, (TS,))[:, 0].reshape(N_TRUE, 24)
    return RD.verdicts(v)


def oc_fit(thfit, gen):
    mL, lsL, mk, lsk = thfit
    L = np.exp(mL + np.exp(lsL) * gen.standard_normal((N_EST, 24))); k = np.exp(mk + np.exp(lsk) * gen.standard_normal((N_EST, 24)))
    return RD.verdicts(LK.qmod(TS, L, k) + S * gen.standard_normal((N_EST, 24)))


def oc_normal(v, gen):
    m, sd = v.mean(), v.std(ddof=1)
    return RD.verdicts(m + sd * gen.standard_normal((N_EST, 24)))


def score(est, tru):
    r = (est + 1e-3) / (tru + 1e-3)
    return np.median(np.abs(np.log(r))), np.mean((r >= 0.5) & (r <= 2)), np.mean(r < 0.5)


def run(args):
    beta, name, p_target, seed = args
    sL, sk, which = TYPES[name]
    lo, hi = {"k": (np.log(0.3), np.log(0.01)), "L": (np.log(5.0), np.log(60.0)), "both": (0.0, 3.0)}[which]
    x = brentq(lambda x: p_unit(th_of(which, x, sL, sk), beta) - p_target, lo, hi, xtol=1e-3)
    th = th_of(which, x, sL, sk)
    gen = np.random.default_rng(seed)
    tru = oc_true(th, beta, gen)
    est_m, est_n = [], []
    for _ in range(NREP):
        y = units(th, 12, beta, gen)
        fit = RD.fit(y, TIMES)
        est_m.append(oc_fit((fit[0], np.log(fit[1]), fit[2], np.log(fit[3])), gen))
        est_n.append(oc_normal(y[:, -1], gen))
    est_m, est_n = np.array(est_m), np.array(est_n)
    cols = []
    for j in range(2):
        a = score(est_m[:, j], tru[j]); b = score(est_n[:, j], tru[j])
        cols.append(f"Int. {j + 1}: true {tru[j]:.3f} model {a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f} normal {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f}")
    return f"β={beta:.1f} {name} p={p_target:.2f} (median L {np.exp(th[0]):.1f} min, median k {np.exp(th[2]):.3f}) | " + " | ".join(cols)


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(b, n, p, 20261003 + 97 * i) for i, (b, n, p) in enumerate((b, n, p) for b in (1.0, 1.5, 2.0) for n in TYPES for p in (0.05, 0.10))]
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        NREP = 2
        print(run(jobs[0])); sys.exit()
    print(f"{NREP} tests each (12 units, 10/20/30 min). median |log ratio| / within 2x / below half. β = 1 is the control (same shape as the unit model)")
    with Pool(6) as pool:
        for line in pool.imap(run, jobs):
            print(line, flush=True)
