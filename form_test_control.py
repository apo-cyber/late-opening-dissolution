"""Control for form_test.py: the test of form when the accelerating path is not quadratic.
(main text: Section "Pooling the pulls", the test-then-choose rule; Supplementary material). All values are assumed.

In form_test.py the accelerating path was exactly quadratic in parameter space (g = (tau/12)^2), so the quadratic fit
had the right form and the rule "alarm on LR_quad -> predict with the quadratic fit" was favoured by construction. Here
the same evaluation is repeated with accelerating paths that are not quadratic:
 cub:   g = (tau/12)^3 (starts late and then moves fast; the quadratic fit has the wrong form)
 onset: g = max(0, tau - 6)/6 (no change up to 6 months, linear afterwards; a broken line that no polynomial represents)
All have g(12) = 1, so the path speed c is the same as for lin (fraction of units failing p = 0.05 at 12 months).
The false-alarm side (linear path) is taken from the saved results of form_test.py (output/form_test.pkl).
Usage: python3 form_test_control.py [NREP] (default 50); needs output/form_test.pkl.
"""
from pathlib import Path
import pickle
import numpy as np
from scipy.stats import chi2
import pooled_pulls as PP
import form_test as LD

OUTDIR = Path(__file__).resolve().parent / "output"
_theta0, _cal0 = PP.theta, PP.calibrate
G = {"cub": lambda u: u**3, "onset": lambda u: np.maximum(0.0, 12.0 * u - 6.0) / 6.0}


def theta(kind, shape, tau, c=None):
    if shape not in G:
        return _theta0(kind, shape, tau, c)
    c = PP.C[(kind, "lin")] if c is None else c
    return PP.TH0 + c * G[shape](tau / 12.0) * (PP.TH24[kind] - PP.TH0)


def calibrate():
    _cal0()


PP.theta = LD.theta = theta  # replaced at import time, so also in the worker processes
LD.calibrate = calibrate


def run(args):
    """Wrapper, so that the worker processes import this module (and the replacement above takes effect)."""
    return LD.run_scenario(args)


def summarize(results, lin_rows):
    thr = chi2.ppf(0.95, 4)
    out = [f"{len(results[0][4])} simulated studies. Alarm = LR_quad > 95 % point of χ²₄ ({thr:.2f}). Prediction at 15 months: median |log ratio| / within 2x / below half [Interpretation 1 | Interpretation 2]"]
    for sh, k, t12, t15, rows in results:
        lr = np.array([r["lr_quad"] for r in rows]); x0 = np.array([r["lr_quad"] for r in lin_rows[k]])
        auc = (lr[:, None] > x0[None, :]).mean()
        alarm = lr > thr
        m5 = np.array([r["m5_15"] for r in rows]); mq = np.array([r["mq15"] for r in rows])
        sw = np.where(alarm[:, None], mq, m5)
        def f(arr):
            cols = []
            for j in range(2):
                e, r = PP.score(arr[:, j], t15[j])
                cols.append(f"{np.median(e):.2f} / {np.mean(r >= 0.5) - np.mean(r > 2):.2f} / {np.mean(r < 0.5):.2f}")
            return " | ".join(cols)
        out.append(f"  {sh}, {k}: true at 15 months [{t15[0]:.3f}, {t15[1]:.3f}]  detection {alarm.mean():.2f} (AUC vs linear {auc:.2f})")
        out.append(f"      linear M5 {f(m5)}")
        out.append(f"      rule      {f(sw)}")
        out.append(f"      deg-2 M5q {f(mq)}")
    return out


if __name__ == "__main__":
    import sys
    from multiprocessing import Pool
    nrep = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    PP.set_directions(); calibrate()
    jobs = [(sh, k, 20261002 + 100 * i, nrep) for i, (sh, k) in enumerate((sh, k) for sh in ("cub", "onset") for k in ("S slow release", "O spread opening", "M mixed"))]
    with Pool(6) as pool:
        results = pool.map(run, jobs)
    with open(OUTDIR / "form_test_control.pkl", "wb") as fh:
        pickle.dump(results, fh)
    with open(OUTDIR / "form_test.pkl", "rb") as fh:
        lin_rows = {k: rows for sh, k, _, _, rows in pickle.load(fh) if sh == "lin"}
    print("\n".join(summarize(results, lin_rows)))
