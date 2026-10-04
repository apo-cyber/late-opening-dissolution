"""Robustness of the test of form (LR_quad) to the optimizer: the linear and quadratic fits are redone from more
starting values (main text: Section "Pooling the pulls", the test of form and the test-then-choose rule). All values are
assumed.

Why: the likelihood of the quadratic (and of the pooled linear) fit can have several local maxima, and the fit with
three starting values (L-BFGS-B) used in form_test.py does not always find the best one (in a check on 6 simulated
data sets, one fit was worse by 2.3 in negative log-likelihood, LR 9.1 against 14.1). This script measures how much the
false alarms, detections and test-then-choose results of form_test.py depend on such misses.

Method: the data are regenerated in the same random order as in form_test.py (the fits use no random numbers, and the
number of draws used by oc_th does not depend on the parameters, so drawing and discarding arrays of the same size keeps
the generator state aligned). For each simulated study:
 - the original fit (three starting values) is repeated and checked against the saved LR_quad (reproduction check)
 - pooled linear fit: the original three starting values + one from a linear regression of the per-pull fitted theta
   on time + three restarts with perturbed time terms
 - quadratic fit: the original three starting values + ones from quadratic and linear regressions of the per-pull
   fitted theta + six restarts with perturbed time terms
The best solution is kept, and LR, the alarm, and the 15-month prediction of the rule (quadratic fit on an alarm) are
recomputed.
Usage: python3 form_test_restart.py check (reproduction check on one study) / python3 form_test_restart.py (all);
needs output/form_test.pkl.
"""
from pathlib import Path
import sys, pickle
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import likelihood as LK
import pooled_pulls as PP
import form_test as LD
from pooled_pulls import theta, fit_pooled, set_directions, calibrate, PULLS, N_UNITS, N_TRUE, N_EST, TIMES

OUTDIR = Path(__file__).resolve().parent / "output"
CRIT = chi2.ppf(0.95, 4)
U = PULLS / 12.0


def skip_oc(n):
    """Draw and discard as many numbers from PP.rng as oc_th(th, n) uses (L, k and the error: 3 arrays of shape (n, 24))."""
    for _ in range(3):
        PP.rng.standard_normal((n, 24))


def poly_start(thetas, x, deg):
    X = np.stack([x ** r for r in range(deg + 1)], 1)
    coef, *_ = np.linalg.lstsq(X, np.asarray(thetas), rcond=None)
    return coef  # (deg + 1, 4)


def fit_lin_more(ug, ys, a0, b0, singles, gen):
    taus = PULLS
    bounds = LK.BOUND4 + [(-0.5, 0.5)] * 4
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    def f(p):
        a, b = p[:4], p[4:]
        ll, g = ug.loglik(a[None] + taus[:, None] * b[None], grad=True)
        return -ll, -np.r_[g.sum(0), (g * taus[:, None]).sum(0)]
    c = poly_start(singles, taus, 1)
    th_first, th_last = singles[0], singles[-1]
    starts = [np.r_[th_first, (th_last - th_first) / 12.0], np.r_[PP.TH0, np.zeros(4)], np.r_[th_last, np.zeros(4)], np.r_[c[0], c[1]], np.r_[a0, b0]]
    best = None
    for s0 in starts:
        r = minimize(f, np.clip(s0, lo, hi), jac=True, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    for _ in range(3):
        x0 = best.x.copy(); x0[4:] += (hi[4:] - lo[4:]) / 4 * gen.standard_normal(4)
        r = minimize(f, np.clip(x0, lo, hi), jac=True, method="L-BFGS-B", bounds=bounds)
        if r.fun < best.fun - 1e-6:
            best = r
    return best.x[:4], best.x[4:], best.fun


def fit_quad_more(ug, a, b, singles, P0, gen):
    Um = np.stack([np.ones_like(U), U, U ** 2], 1)
    def f(p):
        P = p.reshape(3, 4)
        ll, g = ug.loglik(Um @ P, grad=True)
        return -ll, -(Um.T @ g).ravel()
    lo = np.r_[[x[0] for x in LK.BOUND4], [-8] * 8]; hi = np.r_[[x[1] for x in LK.BOUND4], [8] * 8]
    b12 = 12 * b
    c2 = poly_start(singles, U, 2).ravel(); c1 = np.r_[poly_start(singles, U, 1).ravel(), np.zeros(4)]
    starts = [np.r_[a, b12, np.zeros(4)], np.r_[a, np.zeros(4), b12], np.r_[a, 0.5 * b12, 0.5 * b12], c2, c1, P0.ravel()]
    best = None
    for s0 in starts:
        r = minimize(f, np.clip(s0, lo, hi), jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or r.fun < best.fun:
            best = r
    for _ in range(6):
        x0 = best.x.copy(); x0[4:] += (hi[4:] - lo[4:]) / 4 * gen.standard_normal(8)
        r = minimize(f, np.clip(x0, lo, hi), jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if r.fun < best.fun - 1e-6:
            best = r
    return best.x.reshape(3, 4), best.fun


def run(args):
    shape, kind, seed, nrep, stored = args
    PP.rng = np.random.default_rng(seed)
    gen_fit = np.random.default_rng(seed + 11)   # for the perturbed restarts (separate from PP.rng)
    gen_oc = np.random.default_rng(seed + 13)    # for the predicted probabilities (separate from PP.rng)
    set_directions(); calibrate()
    skip_oc(N_TRUE); skip_oc(N_TRUE)             # the original t12, t15
    out = []
    for i in range(nrep):
        ys = [PP.draw(theta(kind, shape, tau), N_UNITS, TIMES) for tau in PULLS]
        skip_oc(N_EST)                           # m1 (last pull only)
        a, b, f5, ug = fit_pooled(ys)
        skip_oc(N_EST); skip_oc(N_EST)           # m5, m5_15
        P, fq = LD.fit_quad(ug, a, b)            # original quadratic fit (for the reproduction check)
        lr_orig = 2 * (f5 - fq)
        singles = [LK.fit_single(LK.UnitGrids(y, TIMES))[0] for y in ys]
        a2, b2, f5b = fit_lin_more(ug, ys, a, b, singles, gen_fit)
        P2, fqb = fit_quad_more(ug, a2, b2, singles, P, gen_fit)
        lr_new = 2 * (f5b - fqb)
        th15_lin = a2 + 15 * b2
        th15_q = P2[0] + 1.25 * P2[1] + 1.25 ** 2 * P2[2]
        f15_lin = LD.oc_th2(th15_lin, N_EST, gen_oc); f15_q = LD.oc_th2(th15_q, N_EST, gen_oc)
        out.append(dict(lr_stored=stored[i]["lr_quad"] if stored else np.nan, lr_orig=lr_orig, lr_new=lr_new,
                        d_lin=f5b - f5, d_quad=fqb - fq, f15_lin=f15_lin, f15_q=f15_q))
        print(f"{shape} {kind} {i}: LR saved {out[-1]['lr_stored']:.3f} reproduced {lr_orig:.3f} restarted {lr_new:.3f}  "
              f"(linear {f5b - f5:+.3f}, quadratic {fqb - fq:+.3f})", flush=True)
    return shape, kind, out


if __name__ == "__main__":
    from multiprocessing import Pool
    with open(OUTDIR / "form_test.pkl", "rb") as fh:
        saved = pickle.load(fh)
    by = {(sh, k): (t12, t15, rows) for sh, k, t12, t15, rows in saved}
    order = [(sh, k) for sh in ("lin", "acc") for k in ("S slow release", "O spread opening", "M mixed")]
    jobs = [(sh, k, 20261001 + 100 * i, len(by[(sh, k)][2]), by[(sh, k)][2]) for i, (sh, k) in enumerate(order)]
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sh, k, seed, n, st = jobs[1]
        run((sh, k, seed, 1, st))
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "--from-pkl":  # print the summary from the saved results only
        with open(OUTDIR / "form_test_restart.pkl", "rb") as fh:
            results = pickle.load(fh)
    else:
        with Pool(6) as pool:
            results = pool.map(run, jobs)
        with open(OUTDIR / "form_test_restart.pkl", "wb") as fh:
            pickle.dump(results, fh)
    print("\nSummary (50 studies each, 95 %% point of χ²₄ for LR_quad = %.2f)" % CRIT)
    for sh, k, rows in results:
        t15 = by[(sh, k)][1]
        lo_, ln_ = np.array([r["lr_orig"] for r in rows]), np.array([r["lr_new"] for r in rows])
        alarm = ln_ > CRIT
        fl = np.array([r["f15_lin"] for r in rows]); fq_ = np.array([r["f15_q"] for r in rows])
        rule = np.where(alarm[:, None], fq_, fl)
        cols = []
        for j in range(2):
            e, r = PP.score(rule[:, j], t15[j])
            cols.append(f"{np.median(e):.2f} / {np.mean((r >= 0.5) & (r <= 2)):.2f} / {np.mean(r < 0.5):.2f}")
        print(f"{sh}, {k}: alarm original {np.mean(lo_ > CRIT):.2f} -> restarted {alarm.mean():.2f}  median LR {np.median(lo_):.2f} -> {np.median(ln_):.2f}  "
              f"improvement of linear fit median {np.median([-r['d_lin'] for r in rows]):.3f}, max {max(-r['d_lin'] for r in rows):.2f}  "
              f"improvement of quadratic fit median {np.median([-r['d_quad'] for r in rows]):.3f}, max {max(-r['d_quad'] for r in rows):.2f}  "
              f"rule at 15 months [Interpretation 1 | 2] {cols[0]} | {cols[1]}")
