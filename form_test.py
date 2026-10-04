"""Test of form: can a wrong (linear) form of the ageing path be detected from the pulls themselves?
(main text: Section "Pooling the pulls", the test of form and the test-then-choose rule). All values are assumed.

In pooled_pulls.py the ranking of the 12-month estimates reversed with the form of the path (O spread opening,
Interpretation 1, median |log ratio|: on the linear path the pooled M5 0.60 < last pull only M1 0.94; on the
accelerating path M5 1.76 > M1 0.88). This script computes how well a disagreement between M1 and M5, or a likelihood
ratio test, flags a wrong assumed form.

With the same paths and the same random stream (draws in the same order as run_scenario in pooled_pulls.py), each
simulated study gives:
 D1, D2   disagreement: |log{(M5@12 + 0.001)/(M1@12 + 0.001)}| (Interpretation 1 and 2). D = max(D1, D2)
 LR_last  fit at the last pull: 2 {l_12(M1 parameters) - l_12(M5 parameters at 12 months)} (log-likelihood of the
          12 units of the 12-month pull only)
 LR_free  test of form 1: 2 {sum_tau l_tau(each pull fitted separately, 20 parameters) - l(M5, 8 parameters)},
          reference chi-squared(12)
 LR_quad  test of form 2: 2 {l(quadratic fit, 12 parameters) - l(M5, 8 parameters)}, reference chi-squared(4).
          Quadratic fit: theta(tau) = a + b u + c u^2, u = tau/12
An alarm on the linear path is a false alarm; an alarm on the accelerating path is a detection. Thresholds: (i) fixed in
advance (D > ln 2, i.e. a factor of two; the 5 % point of chi-squared), and (ii) the value giving 10 % false alarms on
the linear path (for comparison only; in practice the distribution under the linear path is unknown).
Also reported: the rule that switches the 12-month estimate to M1 on an alarm, and the 15-month prediction of the
quadratic fit (M5q@15).
Usage: python3 form_test.py [NREP] (runs the simulation, saves output/form_test.pkl) /
       python3 form_test.py --from-pkl (rebuilds the tables from the saved results)
"""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import pooled_pulls as PP
from pooled_pulls import (theta, oc_th, fit_pooled, score, set_directions, calibrate,
                          PULLS, N_UNITS, N_TRUE, N_EST, TIMES, C)
from regime_diagnostic import verdicts, qmod, S_MEAS, T_SPEC
import likelihood as LK

OUTDIR = Path(__file__).resolve().parent / "output"
NREP = PP.NREP
FL = 1e-3


def oc_th2(th, n, gen):
    """Same computation as oc_th, with a separate generator (so that the PP.rng stream stays the same as in pooled_pulls.py)."""
    mL, lsL, mk, lsk = th
    L = np.exp(mL + np.exp(lsL) * gen.standard_normal((n, 24)))
    k = np.exp(mk + np.exp(lsk) * gen.standard_normal((n, 24)))
    return verdicts(qmod(T_SPEC, L, k) + S_MEAS * gen.standard_normal((n, 24)))


def fit_quad(ug, a, b):
    """Quadratic fit theta_j = a + b u + c u^2, u = tau/12. Starting values: the linear solution (c = 0) and two that pass through the end points."""
    u = PULLS / 12.0
    U = np.stack([np.ones_like(u), u, u**2], 1)  # (pull, 3)
    def f(p):
        P = p.reshape(3, 4)
        ll, g = ug.loglik(U @ P, grad=True)
        return -ll, -(U.T @ g).ravel()
    lo = np.r_[[x[0] for x in LK.BOUND4], [-8] * 8]; hi = np.r_[[x[1] for x in LK.BOUND4], [8] * 8]
    b12 = 12 * b
    starts = [np.r_[a, b12, np.zeros(4)], np.r_[a, np.zeros(4), b12], np.r_[a, 0.5 * b12, 0.5 * b12]]
    best = None
    for s0 in starts:
        r = minimize(f, np.clip(s0, lo, hi), jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or r.fun < best.fun:
            best = r
    P = best.x.reshape(3, 4)
    return P, best.fun


def run_scenario(args):
    shape, kind, seed, nrep = args
    PP.rng = np.random.default_rng(seed)
    gen2 = np.random.default_rng(seed + 7)
    set_directions(); calibrate()
    t12 = oc_th(theta(kind, shape, 12), N_TRUE); t15 = oc_th(theta(kind, shape, 15), N_TRUE)
    rows = []
    for _ in range(nrep):
        # --- PP.rng is used in the same order as in pooled_pulls.run_scenario ---
        ys = [PP.draw(theta(kind, shape, tau), N_UNITS, TIMES) for tau in PULLS]
        ug_last = LK.UnitGrids(ys[-1], TIMES)
        th1, f1 = LK.fit_single(ug_last)
        m1 = oc_th(th1, N_EST)
        a, b, f5, ug = fit_pooled(ys)
        m5 = oc_th(a + 12 * b, N_EST)
        m5_15 = oc_th(a + 15 * b, N_EST)
        # --- additions from here on (using gen2) ---
        f_free = sum(LK.fit_single(LK.UnitGrids(y, TIMES))[1] for y in ys[:-1]) + f1
        P, fq = fit_quad(ug, a, b)
        th_q12 = P[0] + P[1] + P[2]; th_q15 = P[0] + 1.25 * P[1] + 1.25**2 * P[2]
        mq12 = oc_th2(th_q12, N_EST, gen2); mq15 = oc_th2(th_q15, N_EST, gen2)
        lr_last = 2 * (-f1 - ug_last.loglik((a + 12 * b)[None]))
        rows.append(dict(m1=m1, m5=m5, m5_15=m5_15, mq12=mq12, mq15=mq15,
                         lr_free=2 * (f5 - f_free), lr_quad=2 * (f5 - fq), lr_last=lr_last))
    return shape, kind, t12, t15, rows


def summarize(results):
    out = []
    by = {(sh, k): (t12, t15, rows) for sh, k, t12, t15, rows in results}
    kinds = ("S slow release", "O spread opening", "M mixed")
    stat_names = ("D1", "D2", "D", "LR_last", "LR_free", "LR_quad")

    def stats(rows):
        m1 = np.array([r["m1"] for r in rows]); m5 = np.array([r["m5"] for r in rows])
        d = np.abs(np.log((m5 + FL) / (m1 + FL)))
        return {"D1": d[:, 0], "D2": d[:, 1], "D": d.max(1),
                "LR_last": np.array([r["lr_last"] for r in rows]),
                "LR_free": np.array([r["lr_free"] for r in rows]),
                "LR_quad": np.array([r["lr_quad"] for r in rows])}

    pre = {"D1": np.log(2), "D2": np.log(2), "D": np.log(2), "LR_last": chi2.ppf(0.95, 4),
           "LR_free": chi2.ppf(0.95, 12), "LR_quad": chi2.ppf(0.95, 4)}
    out.append(f"{NREP} simulated studies. Thresholds fixed in advance: D > ln 2 (a 2-fold disagreement), LR_last > 95 % point of χ²₄ ({pre['LR_last']:.2f}), "
               f"LR_free > χ²₁₂ ({pre['LR_free']:.2f}), LR_quad > χ²₄ ({pre['LR_quad']:.2f})\n")
    out.append("1. Alarm rate (linear = false alarm / accelerating = detection), detection at the threshold giving 10 % false alarms on the linear path, and AUC (linear vs accelerating)")
    for kind in kinds:
        sl, sa = stats(by[("lin", kind)][2]), stats(by[("acc", kind)][2])
        out.append(f"  {kind}")
        for nm in stat_names:
            x0, x1 = sl[nm], sa[nm]
            thr10 = np.quantile(x0, 0.90)
            auc = (x1[:, None] > x0[None, :]).mean() + 0.5 * (x1[:, None] == x0[None, :]).mean()
            out.append(f"    {nm:8s} median lin {np.median(x0):6.2f} acc {np.median(x1):6.2f} | fixed threshold: false alarm {np.mean(x0 > pre[nm]):.2f} "
                       f"detection {np.mean(x1 > pre[nm]):.2f} | threshold for 10 % false alarms {thr10:6.2f}: detection {np.mean(x1 > thr10):.2f} | AUC {auc:.2f}")
    out.append("\n2. Estimate at 12 months (median |log ratio| [Interpretation 1, 2]): M5 throughout / M1 throughout / switch to M1 on an alarm (D > ln 2, LR_quad > χ²₄) / deg-2 (quadratic fit)")
    for shape in ("lin", "acc"):
        for kind in kinds:
            t12, t15, rows = by[(shape, kind)]
            st = stats(rows)
            m1 = np.array([r["m1"] for r in rows]); m5 = np.array([r["m5"] for r in rows]); mq = np.array([r["mq12"] for r in rows])
            swD = np.where((st["D"] > pre["D"])[:, None], m1, m5)
            swQ = np.where((st["LR_quad"] > pre["LR_quad"])[:, None], m1, m5)
            def med(arr):
                return "[" + ", ".join(f"{np.median(score(arr[:, j], t12[j])[0]):.2f}" for j in range(2)) + "]"
            out.append(f"  {shape}, {kind}: true [{t12[0]:.3f}, {t12[1]:.3f}]  M5 {med(m5)}  M1 {med(m1)}  switch(D) {med(swD)}  "
                       f"switch(LR_quad) {med(swQ)}  deg-2 {med(mq)}")
    out.append("\n3. Prediction at 15 months: linear M5 and deg-2 (quadratic) M5q (median |log ratio| / within 2x / below half, [Interpretation 1 | Interpretation 2])")
    for shape in ("lin", "acc"):
        for kind in kinds:
            t12, t15, rows = by[(shape, kind)]
            def line(arr):
                cols = []
                for j in range(2):
                    e, r = score(arr[:, j], t15[j])
                    cols.append(f"{np.median(e):.2f} / {np.mean(r >= 0.5) - np.mean(r > 2):.2f} / {np.mean(r < 0.5):.2f}")
                return " | ".join(cols)
            out.append(f"  {shape}, {kind}: true [{t15[0]:.3f}, {t15[1]:.3f}]  M5 {line(np.array([r['m5_15'] for r in rows]))}   "
                       f"M5q {line(np.array([r['mq15'] for r in rows]))}")
    out.append("\n4. Test-then-choose rule: if LR_quad > χ²₄, the deg-2 (quadratic) M5q, otherwise the linear M5, predicts 15 months (median |log ratio| / within 2x / below half)")
    for shape in ("lin", "acc"):
        for kind in kinds:
            t12, t15, rows = by[(shape, kind)]
            alarm = np.array([r["lr_quad"] for r in rows]) > pre["LR_quad"]
            sw = np.where(alarm[:, None], np.array([r["mq15"] for r in rows]), np.array([r["m5_15"] for r in rows]))
            cols = []
            for j in range(2):
                e, r = score(sw[:, j], t15[j])
                cols.append(f"{np.median(e):.2f} / {np.mean(r >= 0.5) - np.mean(r > 2):.2f} / {np.mean(r < 0.5):.2f}")
            out.append(f"  {shape}, {kind}: alarm {alarm.mean():.2f}  rule {cols[0]} | {cols[1]}")
    out.append("\nCheck: M1@12 and M5@12 use the same random stream as pooled_pulls.txt, so their median |log ratio| should agree with it (O spread opening, Interpretation 1: linear 0.94 / 0.60, accelerating 0.88 / 1.76)")
    return out


if __name__ == "__main__":
    import sys, pickle
    from multiprocessing import Pool
    if len(sys.argv) > 1 and sys.argv[1] == "--from-pkl":  # rebuild the tables from saved results
        with open(OUTDIR / "form_test.pkl", "rb") as fh:
            results = pickle.load(fh)
        NREP = len(results[0][4])
        print("\n".join(summarize(results))); sys.exit()
    if len(sys.argv) > 1:
        NREP = int(sys.argv[1])
    set_directions(); calibrate()
    jobs = [(sh, k, 20261001 + 100 * i, NREP) for i, (sh, k) in enumerate((sh, k) for sh in ("lin", "acc") for k in ("S slow release", "O spread opening", "M mixed"))]
    with Pool(6) as pool:
        results = pool.map(run_scenario, jobs)
    with open(OUTDIR / "form_test.pkl", "wb") as fh:
        pickle.dump(results, fh)
    print("\n".join(summarize(results)))
