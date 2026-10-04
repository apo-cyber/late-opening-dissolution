"""Detecting the start of ageing while all units pass: from pulls at 0, 3 and 6 months that all meet the specification,
how often can a trend that has begun be detected? (Supplementary Section S7.) All values are assumed; no measured data.

Question: when the mean dissolution falls from 0 to 3 to 6 months, or when all units were near 90 % at 0 months and one
unit at 6 months comes close to the lower limit, can the start of ageing be inferred although the specification is met?

True path: as in pooled_pulls.py. The parameters move linearly from theta0 at release (median L 8 min, sigma_L 0.13,
median k 0.25 /min, sigma_k 0.10) towards the direction of an ageing type (the p ~ 0.20 state of regime_diagnostic.py:
S slow release, O spread opening, M mixed). The speed c is set so that the fraction of units below R (75 % at 30 min)
reaches 0.05 at T* months, T* = 12, 24 or 36 (slow ageing that reaches 5 % by the end of the shelf life); c = 0 (no
change) is the control. The speed was first set by the fraction below R at 6 months (0.002 to 0.03), but that was so
strong that all seven check cases gave a likelihood ratio of 28 or more (detection almost certain), so it was set by the
month at which 0.05 is reached instead.
Each pull: 12 units at 10, 20 and 30 min, analytical error SD 2 %. 200 runs of the control (false alarms), 60 of each
other path.

Methods (each at the 5 % level)
 M           the test of no change: likelihood ratio, on the 10-, 20- and 30-min values of all pulls, of the pooled
             linear fit (parameters linear in time, 8 parameters) against the fit with parameters constant in time
             (4 parameters); alarm if above 9.49, the 5 % point of chi-squared(4)
 N30 mean    the 36 values at 30 min regressed on the month of storage; slope negative (one-sided)
 N30 spread  the distance of each 30-min value from the median of its pull, |y - median|, regressed on the month of
             storage; slope positive (one-sided; the spread widens)
 N30 either  either of the two above (each at 2.5 %)
 N10 mean    the 36 values at 10 min regressed on the month of storage; slope negative (one-sided)
 eye         the lowest 30-min value at 6 months is at least 5 % below the lowest at 0 months (no significance level;
             its false-alarm rate is read from the control)
Which parameter moved, when M gives an alarm: at the 8-parameter fit, z = slope / standard error of each slope from a
numerical Hessian, and count whether the parameter with the largest |z| is on the L side (mu_L, sigma_L) or on the
k side (mu_k, sigma_k).
Usage: python3 trend_detection.py [NPROC] [nocontrol] (default 5 processes; nocontrol skips the control, whose output is
saved in output/trend_detection_control.txt; the results are saved in output/trend_detection.pkl).
Run with the argument "check" for a quick three-run test of one path.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.optimize import minimize, brentq
import likelihood as LK
from regime_diagnostic import qmod, S_MEAS, T_SPEC, R2, L0, TIMES, make_types

OUTDIR = Path(__file__).resolve().parent / "output"
PULLS = np.array([0.0, 3.0, 6.0])
N_UNITS, NREP, N_ALT = 12, 200, 60
TH0 = np.array([np.log(L0), np.log(0.13), np.log(0.25), np.log(0.10)])
TSTAR = (12.0, 24.0, 36.0)
CHI2_4 = 9.488
NAMES = ("μ_L", "σ_L", "μ_k", "σ_k")
TH20 = {}


def set_directions():
    for (kind, p), th in make_types().items():
        if p == 0.20:
            mL, sL, mk, sk = th
            TH20[kind] = np.array([mL, np.log(sL), mk, np.log(sk)])


def theta(kind, c, tau):
    return TH0 + c * (tau / 6.0) * (TH20[kind] - TH0)


def calibrate(kind, tstar):
    z = np.random.default_rng(7)
    zl, zk, ze = z.standard_normal(400_000), z.standard_normal(400_000), z.standard_normal(400_000)
    def p(c):
        mL, lsL, mk, lsk = theta(kind, c, tstar)
        q = qmod(T_SPEC, np.exp(mL + np.exp(lsL) * zl), np.exp(mk + np.exp(lsk) * zk)) + S_MEAS * ze
        return (q < R2).mean()
    return brentq(lambda c: p(c) - 0.05, 0.0, 1.5, xtol=1e-5)


def p_unit(th):
    z = np.random.default_rng(8)
    mL, lsL, mk, lsk = th
    q = qmod(T_SPEC, np.exp(mL + np.exp(lsL) * z.standard_normal(400_000)), np.exp(mk + np.exp(lsk) * z.standard_normal(400_000))) + S_MEAS * z.standard_normal(400_000)
    return (q < R2).mean()


def draw(th, n, gen):
    mL, lsL, mk, lsk = th
    L = np.exp(mL + np.exp(lsL) * gen.standard_normal(n)); k = np.exp(mk + np.exp(lsk) * gen.standard_normal(n))
    return np.stack([qmod(t, L, k) + S_MEAS * gen.standard_normal(n) for t in TIMES], -1)


def fit_const(ug, starts):
    ng = len(PULLS)
    def f(p):
        ll, g = ug.loglik(np.repeat(p[None], ng, 0), grad=True)
        return -ll, -g.sum(0)
    best = None
    for s0 in starts:
        r = minimize(f, s0, jac=True, method="L-BFGS-B", bounds=LK.BOUND4)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def slope_z(ug, a, b):
    """z of the four slopes, from the numerical Hessian (central differences of the analytical gradient) of the
    negative log-likelihood of the 8 parameters."""
    def grad(p):
        ll, g = ug.loglik(p[:4][None] + PULLS[:, None] * p[4:][None], grad=True)
        return -np.r_[g.sum(0), (g * PULLS[:, None]).sum(0)]
    p0, h = np.r_[a, b], np.r_[[1e-3] * 4, [1e-4] * 4]
    H = np.zeros((8, 8))
    for j in range(8):
        e = np.zeros(8); e[j] = h[j]
        H[:, j] = (grad(p0 + e) - grad(p0 - e)) / (2 * h[j])
    H = 0.5 * (H + H.T)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov)[4:], 1e-12, None))
    except np.linalg.LinAlgError:
        return np.full(4, np.nan)
    return b / se


def ols_slope_p(x, y, side):
    r = stats.linregress(x, y)
    t = r.slope / r.stderr if r.stderr > 0 else 0.0
    df = len(x) - 2
    return stats.t.sf(t, df) if side > 0 else stats.t.cdf(t, df)


def one_rep(kind, c, gen):
    ys = [draw(theta(kind, c, tau), N_UNITS, gen) for tau in PULLS]
    tau = np.repeat(PULLS, N_UNITS)
    y30 = np.concatenate([y[:, 2] for y in ys]); y10 = np.concatenate([y[:, 0] for y in ys])
    dev = np.concatenate([np.abs(y[:, 2] - np.median(y[:, 2])) for y in ys])
    p_mean = ols_slope_p(tau, y30, -1); p_wid = ols_slope_p(tau, dev, +1); p10 = ols_slope_p(tau, y10, -1)
    eye = ys[-1][:, 2].min() <= ys[0][:, 2].min() - 5.0
    allpass = bool((y30 >= R2).all())
    ug = LK.UnitGrids(np.concatenate(ys), TIMES, group=np.repeat(np.arange(len(PULLS)), N_UNITS))
    th_s = [LK.fit_single(LK.UnitGrids(y, TIMES))[0] for y in (ys[0], ys[-1])]
    a0, f0 = fit_const(ug, LK.STARTS4 + [th_s[0], th_s[1]])
    starts = [np.r_[a0, np.zeros(4)], np.r_[th_s[0], (th_s[1] - th_s[0]) / 6.0], np.r_[TH0, np.zeros(4)]]
    a, b, f1 = LK.fit_pooled(ug, PULLS, starts)
    LR = 2 * (f0 - f1)
    z = slope_z(ug, a, b) if LR > CHI2_4 else np.full(4, np.nan)
    return dict(LR=LR, M=LR > CHI2_4, N30m=p_mean < 0.05, N30w=p_wid < 0.05, N30b=(p_mean < 0.025) | (p_wid < 0.025),
                N10m=p10 < 0.05, eye=eye, allpass=allpass, z=z, m6=ys[-1][:, 2].mean(), min6=ys[-1][:, 2].min(),
                m0=ys[0][:, 2].mean(), min0=ys[0][:, 2].min(), m10_0=ys[0][:, 0].mean(), m10_6=ys[-1][:, 0].mean())


def simulate(args):
    kind, tstar, seed, nrep = args
    set_directions()
    c = 0.0 if tstar == 0 else calibrate(kind, tstar)
    gen = np.random.default_rng(seed)
    return (kind, tstar), c, [one_rep(kind if tstar else "O spread opening", c, gen) for _ in range(nrep)]


def summarize(kind, tstar, c, rs):
    col = lambda k: np.array([r[k] for r in rs])
    LR = col("LR"); M = col("M")
    zs = np.array([r["z"] for r in rs if r["M"]])
    if len(zs):
        top = np.nanargmax(np.abs(zs), 1)
        side = f"L side {np.mean(top < 2):.2f}, k side {np.mean(top >= 2):.2f}"
        sig = ", ".join(f"{n} {np.mean(np.abs(zs[:, j]) > 1.96):.2f}" for j, n in enumerate(NAMES))
    else:
        side, sig = "—", "—"
    head = "No change (control)" if tstar == 0 else f"{kind} T*={tstar:.0f} months (c={c:.4f}, fraction below R at 6 months {p_unit(theta(kind, c, 6.0)):.4f})"
    lines = [f"{head}: 30-min values: 0 months mean {col('m0').mean():.1f}, min {col('min0').mean():.1f} → 6 months mean {col('m6').mean():.1f}, min {col('min6').mean():.1f}"
             f" (median over runs {np.median(col('min6')):.1f}); 10-min values: mean {col('m10_0').mean():.1f} → {col('m10_6').mean():.1f}; runs with all 36 units at or above 75 %: {col('allpass').mean():.2f}",
             f"   Proportion of alarms  M {M.mean():.2f}  N30 mean {col('N30m').mean():.2f}  N30 spread {col('N30w').mean():.2f}  N30 either {col('N30b').mean():.2f}"
             f"  N10 mean {col('N10m').mean():.2f}  eye {col('eye').mean():.2f}   | LR median {np.median(LR):.2f}, 95th percentile {np.quantile(LR, 0.95):.2f}, negative {np.mean(LR < -1e-6):.2f}",
             f"   Parameter with the largest |z| on an M alarm: {side}  /  proportion with |z|>1.96: {sig}",
             f"   Alarms in the runs with all units at or above 75 %  M {M[col('allpass')].mean():.2f}  N30 either {col('N30b')[col('allpass')].mean():.2f}  N10 mean {col('N10m')[col('allpass')].mean():.2f}"]
    return "\n".join(lines)


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [("control", 0, 20261005 + j, NREP // 4) for j in range(4)] + [(k, p, 20261005 + 13 * (i + 1), N_ALT)
                                                                          for i, (k, p) in enumerate((k, p) for k in ("S slow release", "O spread opening", "M mixed") for p in TSTAR)]
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        import time
        t0 = time.time()
        (k, p), c, rs = simulate(("O spread opening", 24.0, 1, 3))
        print(summarize(k, p, c, rs), f"\n{time.time() - t0:.1f} s for 3 runs")
        sys.exit()
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    set_directions()  # summarize runs in the parent process, so the directions of the types are loaded here as well
    if len(sys.argv) > 2 and sys.argv[2] == "nocontrol":  # the control is saved in output/trend_detection_control.txt
        jobs = [j for j in jobs if j[1] != 0]
    import pickle
    print(f"Pulls at {PULLS.astype(int).tolist()} months, {N_UNITS} units each. Control {NREP} runs, the others {N_ALT} runs each. Specification at {T_SPEC:.0f} min: {R2:.0f} %. "
          f"Significance level 5 % (M: χ²₄ {CHI2_4})")
    acc = {}
    with Pool(nproc) as pool:
        for key, c, rs in pool.imap(simulate, jobs):
            acc.setdefault(key, [c, []])[1].extend(rs)
            with open(OUTDIR / "trend_detection.pkl", "wb") as fh:  # saved after each finished job, so the results survive an interruption
                pickle.dump(acc, fh)
            if key[1] != 0 or len(acc[key][1]) == NREP:
                print(summarize(key[0], key[1], c, acc[key][1]), flush=True); print(flush=True)
