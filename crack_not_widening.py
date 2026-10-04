"""Cracks that do not widen: when the slow units are a separate subpopulation rather than the lognormal tail, how far off
is the probability of failing estimated by fitting the unit model (L and k both lognormal)? (Main text: limitations;
Supplementary Section S6.)

Interpretation 2 is decided by the one or two slow units among six (the tail of the distribution). If storage produces a
fixed fraction of units whose crack stays narrow and does not widen, the tail can be heavier than the lognormal
extrapolation.
True units: the bulk follows one of the three ageing types of regime_diagnostic.py (the state with a fraction p = 0.05
of units below R; first-order from the moment the unit opens). A fraction pi of "crack-not-widening" units is mixed in:
L has the same distribution as the bulk, and ln k ~ N(ln 0.015, 0.3^2) (around 30 % dissolved at 30 min).
pi = 0 (control), 0.03 and 0.10. One test is one pull of 12 units at 10, 20 and 30 min with an analytical error of
SD 2 %. The fit is the unit model (likelihood.fit_single). Reported for the probability of failing (Interpretation 1
and 2): median |log ratio| / proportion within a factor of two / proportion below half; the normal calculation (from
mean and SD) alongside. 40 tests per state. All values are assumed; no measured data.
Run with the argument "check" for a quick two-test run of one state.
"""
import sys
import numpy as np
import likelihood as LK
import regime_diagnostic as RD

S, TS, R2, TIMES = 2.0, 30.0, 75.0, (10.0, 20.0, 30.0)
NREP, N_TRUE, N_EST = 40, 200_000, 20_000
K_SLOW, SK_SLOW = 0.015, 0.3


def units(th, n, pi, gen, times=TIMES):
    mL, sL, mk, sk = th
    L = np.exp(mL + sL * gen.standard_normal(n)); k = np.exp(mk + sk * gen.standard_normal(n))
    slow = gen.random(n) < pi
    k = np.where(slow, np.exp(np.log(K_SLOW) + SK_SLOW * gen.standard_normal(n)), k)
    return np.stack([LK.qmod(t, L, k) + S * gen.standard_normal(n) for t in times], -1)


def oc_fit(thfit, gen):
    mL, lsL, mk, lsk = thfit
    L = np.exp(mL + np.exp(lsL) * gen.standard_normal((N_EST, 24))); k = np.exp(mk + np.exp(lsk) * gen.standard_normal((N_EST, 24)))
    return RD.verdicts(LK.qmod(TS, L, k) + S * gen.standard_normal((N_EST, 24)))


def score(est, tru):
    r = (est + 1e-3) / (tru + 1e-3)
    return np.median(np.abs(np.log(r))), np.mean((r >= 0.5) & (r <= 2)), np.mean(r < 0.5)


def run(args):
    name, pi, seed = args
    th = RD.make_types()[(name, 0.05)]
    gen = np.random.default_rng(seed)
    v = units(th, N_TRUE * 24, pi, gen, (TS,))[:, 0]
    p_true = (v < R2).mean()
    tru = RD.verdicts(v.reshape(N_TRUE, 24))
    em, en = [], []
    for _ in range(NREP):
        y = units(th, 12, pi, gen)
        f = RD.fit(y, TIMES)
        em.append(oc_fit((f[0], np.log(f[1]), f[2], np.log(f[3])), gen))
        m, sd = y[:, -1].mean(), y[:, -1].std(ddof=1)
        en.append(RD.verdicts(m + sd * gen.standard_normal((N_EST, 24))))
    em, en = np.array(em), np.array(en)
    cols = []
    for j in range(2):
        a = score(em[:, j], tru[j]); b = score(en[:, j], tru[j])
        cols.append(f"Int. {j + 1}: true {tru[j]:.3f} model {a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f} normal {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f}")
    return f"{name} π={pi:.2f} (fraction below R {p_true:.3f}) | " + " | ".join(cols)


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(n, pi, 20261004 + 31 * i) for i, (n, pi) in enumerate((n, pi) for n in ("S slow release", "O spread opening", "M mixed") for pi in (0.0, 0.03, 0.10))]
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        NREP = 2
        print(run(jobs[1])); sys.exit()
    print(f"{NREP} tests each (12 units, 10/20/30 min). median |log ratio| / within 2x / below half. π = 0 is the control")
    with Pool(6) as pool:
        for line in pool.imap(run, jobs):
            print(line, flush=True)
