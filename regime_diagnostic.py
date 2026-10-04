"""Regime diagnosis at one pull: can the per-vessel values of 12 units tell a lot in which the normal calculation of
the probability of failing is adequate (single peak) from one in which it is not (two clusters)? Also defines the three
ageing types, the acceptance procedures and the normal calculation used by the other scripts (main text: Table 1,
Section "Diagnosis at one pull" and its table). All values are assumed; no measured data.

Unit model as in likelihood.py, with D_max = 100 % and an analytical error of SD 2 %. Nominal specification (not taken
from any product): 75 % at 30 min. Interpretation 1 (harmonized three-stage procedure) with Q = 70 (stage 1 at Q + 5 =
75); Interpretation 2 (Japanese Pharmacopoeia individual-value procedure) with R = 75 % (6 units, then 12, at least 10
of 12 meeting R). Median opening time of a new unit L0 = 8 min.

True regime: the normal calculation (normal distribution with the true mean and SD) is off by more than a factor of two,
and by at least 0.02, for either procedure -> "two-cluster"; otherwise "single peak".

Ageing types, each at three levels of the fraction p of units below R (0.05, 0.10, 0.20):
 S slow release:   L tight (median L0, sigma_L 0.13); k falls and spreads (sigma_k 0.28).
 O spread opening: k tight (median 0.25 /min, sigma_k 0.10); L lengthens and spreads (sigma_L 0.45).
 M mixed:          sigma_L 0.30, sigma_k 0.20; L lengthens and k falls.

Rules for declaring a pull of 12 units two-cluster:
 D1  simple flag: any unit below Q - 25 (= 45 %) at the specification time.
 D2  fitted model: fit the unit model (4 parameters) by maximum likelihood; two-cluster if the probability of failing
     computed from the fit and that computed from the sample mean and SD differ by more than a factor of two and
     0.02 for either procedure. D2a uses the 30-min values only, D2b the values at 10, 20 and 30 min.
Also reported: the error of the estimated probability of failing (median |log ratio|, 0.001 added) for the normal
calculation and for D2b.
"""
import numpy as np
from scipy.optimize import brentq
from likelihood import UnitGrids, fit_single, qmod  # plateau of qmod: likelihood.DMAX (default 100)

rng = np.random.default_rng(20260930)
S_MEAS, T_SPEC, R2, Q1 = 2.0, 30.0, 75.0, 70.0
TIMES = (10.0, 20.0, 30.0)
L0 = 8.0  # median opening time of a new unit (min)
N_TRUE, N_EST, NREP, N_UNITS = 200_000, 20_000, 150, 12


def draw_units(th, size, times=(T_SPEC,)):
    mL, sL, mk, sk = th
    L = np.exp(mL + sL * rng.standard_normal(size)); k = np.exp(mk + sk * rng.standard_normal(size))
    return np.stack([qmod(t, L, k) + S_MEAS * rng.standard_normal(size) for t in times], -1)


def verdicts(q):
    f6 = (q[:, :6] < R2).sum(1); f12 = (q[:, :12] < R2).sum(1)
    pass2 = (f6 == 0) | ((f6 <= 2) & (f12 <= 2))
    s1 = (q[:, :6] >= Q1 + 5).all(1)
    s2 = (q[:, :12].mean(1) >= Q1) & (q[:, :12] >= Q1 - 15).all(1)
    s3 = (q.mean(1) >= Q1) & ((q < Q1 - 15).sum(1) <= 2) & (q >= Q1 - 25).all(1)
    return np.array([1 - (s1 | s2 | s3).mean(), 1 - pass2.mean()])  # probability of failing [Interpretation 1, Interpretation 2]


def oc_model(th, n):
    return verdicts(draw_units(th, (n, 24))[..., 0])


def oc_normal(m, v, n):
    return verdicts(m + v * rng.standard_normal((n, 24)))


def unreliable(a, b):
    """a: reference, b: compared. True if they differ by more than a factor of two (and 0.02) for either procedure."""
    for x, y in zip(a, b):
        if abs(x - y) >= 0.02 and (y < x / 2 or y > 2 * x):
            return True
    return False


def p_unit(th):
    return (draw_units(th, 200_000)[:, 0] < R2).mean()


def make_types():
    types = {}
    for name, (sL, sk, which) in {"S slow release": (0.13, 0.28, "k"), "O spread opening": (0.45, 0.10, "L"),
                                  "M mixed": (0.30, 0.20, "both")}.items():
        for p_target in (0.05, 0.10, 0.20):
            def th_of(x):
                if which == "k":
                    return (np.log(L0), sL, x, sk)
                if which == "L":
                    return (x, sL, np.log(0.25), sk)
                return (np.log(L0) + 0.6 * x, sL, np.log(0.25) - x, sk)  # L lengthens and k falls
            lo, hi = {"k": (np.log(0.3), np.log(0.01)), "L": (np.log(5.0), np.log(60.0)), "both": (0.0, 3.0)}[which]
            x = brentq(lambda x: p_unit(th_of(x)) - p_target, lo, hi, xtol=1e-3)
            types[(name, p_target)] = th_of(x)
    return types


def fit(y, times):
    """Maximum-likelihood fit of the unit model (local-grid integration of likelihood.py). Returns (mu_L, sigma_L, mu_k, sigma_k)."""
    th, _ = fit_single(UnitGrids(y, times))
    mL, lsL, mk, lsk = th
    return (mL, np.exp(lsL), mk, np.exp(lsk))


def run_state(args):
    """Diagnosis and estimation for one state; returns the printed line (unit of parallel work)."""
    global rng
    key, th, t_oc, lab, seed = args
    rng = np.random.default_rng(seed)
    f1 = f2a = f2b = 0
    err_n, err_m = [], []
    for _ in range(NREP):
        y = draw_units(th, N_UNITS, TIMES)
        ys = y[:, -1]
        f1 += (ys < Q1 - 25).any()
        n_oc = oc_normal(ys.mean(), ys.std(ddof=1), N_EST)
        tha = fit(ys[:, None], (T_SPEC,))
        f2a += unreliable(oc_model(tha, N_EST), n_oc)
        thb = fit(y, TIMES)
        m_oc = oc_model(thb, N_EST)
        f2b += unreliable(m_oc, n_oc)
        fl = 1e-3
        err_n.append(np.abs(np.log((n_oc + fl) / (t_oc + fl))))
        err_m.append(np.abs(np.log((m_oc + fl) / (t_oc + fl))))
    en, em = np.array(err_n), np.array(err_m)
    return (f"  {key[0]} p {key[1]:.2f}  {'two-cluster ' if lab else 'single peak '}   |  {f1/NREP:.2f}   |  {f2a/NREP:.2f}         |  {f2b/NREP:.2f}          "
            f"|  {np.median(en[:, 0]):.2f} / {np.median(em[:, 0]):.2f}                     |  {np.median(en[:, 1]):.2f} / {np.median(em[:, 1]):.2f}")


if __name__ == "__main__":
    from multiprocessing import Pool
    types = make_types()
    print(f"Specification at {T_SPEC:.0f} min: {R2:.0f} % (Interpretation 1: Q = {Q1:.0f}), {N_UNITS} units, {NREP} pulls, time points {TIMES}\n")
    print("1. True probability of failing [Interpretation 1, Interpretation 2] by type, the normal calculation, and the true regime")
    truth = {}
    for key, th in types.items():
        t = oc_model(th, N_TRUE)
        y = draw_units(th, N_TRUE)[:, 0]
        n = oc_normal(y.mean(), y.std(), N_TRUE)
        lab = unreliable(t, n)
        low = (y < Q1 - 25).mean()
        truth[key] = (t, lab)
        print(f"  {key[0]} p {key[1]:.2f}: median L {np.exp(th[0]):5.1f} min, sigma_L {th[1]:.2f}, median k {np.exp(th[2]):.3f}, sigma_k {th[3]:.2f}, "
              f"below Q-25 {low:.3f} | true [{t[0]:.3f}, {t[1]:.3f}] normal [{n[0]:.3f}, {n[1]:.3f}] -> {'two-cluster' if lab else 'single peak'}")
    print()
    print("2. Proportion of pulls declared two-cluster by each rule, and the error of the estimated probability of failing")
    print("  type, p                 true regime  | D1 flag | D2a 30 min only | D2b 3 times | median |log ratio| Int. 1: normal / D2b | Int. 2: normal / D2b")
    jobs = [(key, th, truth[key][0], truth[key][1], 20260930 + 7 * i) for i, (key, th) in enumerate(types.items())]
    with Pool(9) as pool:
        for line in pool.map(run_state, jobs):
            print(line)
