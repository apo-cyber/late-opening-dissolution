"""Pooling the stability pulls: how much does fitting all pulls together improve the estimated probability of failing?
(main text: Section "Pooling the pulls" and its table). All values are assumed; no measured data.

At one pull of 12 units the probability of failing is known only to its order of magnitude (regime_diagnostic.py).
Here the unit model is fitted to the pulls of a stability study (0, 3, 6, 9 and 12 months, 12 units each, values at
10, 20 and 30 min), and the probability of failing is estimated (a) at 12 months and (b) at 15 months (extrapolation
three months ahead; at 18 months the true value is close to 1).

True path: the parameters (mu_L, ln sigma_L, mu_k, ln sigma_k) of the unit model (ln L ~ N(mu_L, sigma_L^2),
ln k ~ N(mu_k, sigma_k^2), independent) move from theta0 at release (median L 8 min, sigma_L 0.13, median k 0.25 /min,
sigma_k 0.10) in the direction D = theta_type - theta0 of an ageing type (theta_type is the state with p ~ 0.20 in
regime_diagnostic.py: S slow release, O spread opening, M mixed): theta(tau) = theta0 + c g(tau) D.
The speed c is set so that the fraction p of units failing at the last pull (12 months) is 0.05 (early ageing, when an
early warning is needed).
 path lin: g = tau/12 (matches the fitted model; the best case)
 path acc: g = (tau/12)^2 (accelerating; the fitted model stays linear, i.e. its form is wrong)
Specification (nominal) as in regime_diagnostic.py (75 % at 30 min, Q = 70). Analytical error SD 2 %.

Estimation methods
 N1 normal calculation, last pull only: mean and SD of the values at the specification time at 12 months
    (no estimate at 15 months)
 N5 normal calculation, pulls pooled: the mean and the SD of the values at the specification time at each pull are each
    regressed linearly on time, and evaluated at 12 and 15 months
 M1 unit model, last pull only: 4 parameters fitted by maximum likelihood to the 10-, 20- and 30-min values at
    12 months (no estimate at 15 months)
 M5 unit model, pulls pooled (pooled linear fit): 8 parameters (intercept and slope of each of the 4 parameters) fitted
    by maximum likelihood to the 10-, 20- and 30-min values of all pulls
Evaluation (at 12 and at 15 months), for each of Interpretation 1 and 2: median |log ratio| to the true probability of
failing (0.001 added), the proportion within a factor of two, and the proportion underestimated by more than half (the
unsafe side).
"""
import numpy as np
from regime_diagnostic import qmod, verdicts, oc_normal, S_MEAS, T_SPEC, R2, Q1, L0, TIMES, make_types
import likelihood as LK

rng = np.random.default_rng(20261001)
PULLS = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
N_UNITS, NREP, N_TRUE, N_EST = 12, 100, 200_000, 20_000
TH0 = np.array([np.log(L0), np.log(0.13), np.log(0.25), np.log(0.10)])
TH24 = {}  # end point of the direction of each type = the p ~ 0.20 state of regime_diagnostic.make_types() (set at run time)


def set_directions():
    for (kind, p), th in make_types().items():
        if p == 0.20:
            mL, sL, mk, sk = th
            TH24[kind] = np.array([mL, np.log(sL), mk, np.log(sk)])


C = {}


def theta(kind, shape, tau, c=None):
    c = C[(kind, shape)] if c is None else c
    g = tau / 12.0 if shape == "lin" else (tau / 12.0) ** 2
    return TH0 + c * g * (TH24[kind] - TH0)  # (mu_L, ln sigma_L, mu_k, ln sigma_k)


def calibrate():
    from scipy.optimize import brentq
    z = np.random.default_rng(7)
    zl, zk, ze = z.standard_normal(200_000), z.standard_normal(200_000), z.standard_normal(200_000)
    def p12(c, kind, shape):
        mL, lsL, mk, lsk = theta(kind, shape, 12.0, c)
        q = qmod(T_SPEC, np.exp(mL + np.exp(lsL) * zl), np.exp(mk + np.exp(lsk) * zk)) + S_MEAS * ze
        return (q < R2).mean()
    for kind in TH24:
        for shape in ("lin", "acc"):
            C[(kind, shape)] = brentq(lambda c: p12(c, kind, shape) - 0.05, 0.05, 1.5, xtol=1e-4)


def draw(th, size, times):
    mL, lsL, mk, lsk = th
    L = np.exp(mL + np.exp(lsL) * rng.standard_normal(size))
    k = np.exp(mk + np.exp(lsk) * rng.standard_normal(size))
    return np.stack([qmod(t, L, k) + S_MEAS * rng.standard_normal(size) for t in times], -1)


def oc_th(th, n):
    return verdicts(draw(th, (n, 24), (T_SPEC,))[..., 0])


def fit_single(y):
    th, _ = LK.fit_single(LK.UnitGrids(y, TIMES))
    return th


def fit_pooled(ys):
    """ys: one (12, 3) array per pull. Parameters = 4 intercepts + 4 slopes (per month). Likelihood from likelihood.py."""
    ug = LK.UnitGrids(np.concatenate(ys), TIMES, group=np.repeat(np.arange(len(ys)), [len(y) for y in ys]))
    th_last = fit_single(ys[-1]); th_first = fit_single(ys[0])
    starts = [np.r_[th_first, (th_last - th_first) / 12.0], np.r_[TH0, np.zeros(4)], np.r_[th_last, np.zeros(4)]]
    a, b, fval = LK.fit_pooled(ug, PULLS, starts)
    return a, b, fval, ug


def score(est, true):
    fl = 1e-3
    return np.abs(np.log((est + fl) / (true + fl))), (est + fl) / (true + fl)


def run_scenario(args):
    """Compute one (path, type) and return the lines to print (unit of parallel work)."""
    global rng
    shape, kind, seed = args
    rng = np.random.default_rng(seed)
    set_directions(); calibrate()
    out = []
    t12, t15 = oc_th(theta(kind, shape, 12), N_TRUE), oc_th(theta(kind, shape, 15), N_TRUE)
    gaps = []
    res = {k: [] for k in ("N1@12", "N5@12", "M1@12", "M5@12", "N5@15", "M5@15")}
    for _ in range(NREP):
        ys = [draw(theta(kind, shape, tau), N_UNITS, TIMES) for tau in PULLS]
        y45 = [y[:, -1] for y in ys]  # values at the specification time
        m = np.array([v.mean() for v in y45]); s = np.array([v.std(ddof=1) for v in y45])
        bm, bs = np.polyfit(PULLS, m, 1), np.polyfit(PULLS, s, 1)
        res["N1@12"].append(oc_normal(m[-1], s[-1], N_EST))
        res["N5@12"].append(oc_normal(np.polyval(bm, 12), max(np.polyval(bs, 12), 0.5), N_EST))
        res["N5@15"].append(oc_normal(np.polyval(bm, 15), max(np.polyval(bs, 15), 0.5), N_EST))
        res["M1@12"].append(oc_th(fit_single(ys[-1]), N_EST))
        a, b, fval, ug = fit_pooled(ys)
        ftrue = -ug.loglik(np.array([theta(kind, shape, tau) for tau in PULLS]))
        gaps.append(ftrue - fval)
        res["M5@12"].append(oc_th(a + 12 * b, N_EST))
        res["M5@15"].append(oc_th(a + 15 * b, N_EST))
    out.append(f"Path {shape}, {kind}: true probability of failing [Interpretation 1, 2] at 12 months [{t12[0]:.3f}, {t12[1]:.3f}], at 15 months [{t15[0]:.3f}, {t15[1]:.3f}]")
    gaps = np.array(gaps)
    out.append(f"   Check: negative log-likelihood at the true parameters minus that at the fit: median {np.median(gaps):.1f}, range [{gaps.min():.1f}, {gaps.max():.1f}]"
               f" (for a linear path at least 0, and of the order of the number of parameters, 8)")
    out.append("   method    Int. 1: median |log ratio|  within 2x  below half | Int. 2: median |log ratio|  within 2x  below half")
    for key, arr in res.items():
        arr = np.array(arr); tru = t12 if key.endswith("12") else t15
        cols = []
        for j in range(2):
            e, r = score(arr[:, j], tru[j])
            cols.append(f"{np.median(e):5.2f}      {np.mean(r >= 0.5) - np.mean(r > 2):.2f}      {np.mean(r < 0.5):.2f}")
        out.append(f"   {key:7s}   {cols[0]}  |      {cols[1]}")
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    set_directions(); calibrate()
    print(f"Pulls at {PULLS.astype(int).tolist()} months, {N_UNITS} units each, {NREP} repetitions. Specification at {T_SPEC:.0f} min: {R2:.0f} % (Interpretation 1: Q = {Q1:.0f})")
    print("Path speed c, set so that at 12 months the fraction of units failing is p = 0.05: " + "; ".join(f"{k[0]} ({k[1]}) {v:.3f}" for k, v in C.items()) + "\n")
    jobs = [(shape, kind, 20261001 + 100 * i) for i, (shape, kind) in enumerate((sh, k) for sh in ("lin", "acc") for k in ("S slow release", "O spread opening", "M mixed"))]
    with Pool(6) as pool:
        for lines in pool.map(run_scenario, jobs):
            print("\n".join(lines)); print()
