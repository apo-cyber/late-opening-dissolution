"""Plateau D_max below 100 %, pooled linear fit: how far off is the prediction of the probability of failing at 12 and
15 months when the pooled linear fit is made with D_max fixed at 100 % for a product that levels off below 100 %?
(Main text: limitations, plateau D_max; Supplementary Information.)

Same ageing paths (linear, p = 0.05 at 12 months) and pulls (0-12 months, 12 units each, 10/20/30 min) as
pooled_pulls.py, regenerated under a true plateau TRUE_D (95 and 90) and fitted twice: with D_max fixed at 100 and with
the true value. The states (direction of each ageing type, speed c of the path) are recalibrated under the true plateau.
Accelerating paths are not examined. Reported per fit: median |log ratio| / proportion within a factor of two /
proportion below half of the true probability of failing (Interpretation 1 and 2), and the median estimated k at
12 months. All values are assumed; no measured data.
"""
import numpy as np
import likelihood as LK
import pooled_pulls as PP

NREP = 40


def run(args):
    true_d, kind, seed = args
    LK.DMAX = true_d
    PP.set_directions(); PP.calibrate()
    PP.rng = np.random.default_rng(seed)
    th12, th15 = PP.theta(kind, "lin", 12), PP.theta(kind, "lin", 15)
    t12, t15 = PP.oc_th(th12, PP.N_TRUE), PP.oc_th(th15, PP.N_TRUE)
    res = {100.0: {"@12": [], "@15": [], "k12": []}, true_d: {"@12": [], "@15": [], "k12": []}}
    for _ in range(NREP):
        LK.DMAX = true_d
        ys = [PP.draw(PP.theta(kind, "lin", tau), PP.N_UNITS, PP.TIMES) for tau in PP.PULLS]
        for d in (100.0, true_d):
            LK.DMAX = d
            a, b, _, _ = PP.fit_pooled(ys)
            res[d]["@12"].append(PP.oc_th(a + 12 * b, PP.N_EST))
            res[d]["@15"].append(PP.oc_th(a + 15 * b, PP.N_EST))
            res[d]["k12"].append(np.exp((a + 12 * b)[2]))
    out = [f"true D_max {true_d:.0f}, {kind}, linear: true probability of failing [Int. 1, 2] 12 months [{t12[0]:.3f}, {t12[1]:.3f}], 15 months [{t15[0]:.3f}, {t15[1]:.3f}], "
           f"true median k at 12 months {np.exp(th12[2]):.3f}"]
    for d in (100.0, true_d):
        cols = []
        for key, tru in (("@12", t12), ("@15", t15)):
            arr = np.array(res[d][key])
            for j in range(2):
                e, r = PP.score(arr[:, j], tru[j])
                cols.append(f"{key} Int. {j + 1} {np.median(e):.2f}/{np.mean((r >= 0.5) & (r <= 2)):.2f}/{np.mean(r < 0.5):.2f}")
        out.append(f"   fitted with D_max={d:5.1f}: " + "  ".join(cols) + f"  median estimated k@12 {np.median(res[d]['k12']):.3f}")
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    print(f"{NREP} tests each. median |log ratio| / within 2x / below half")
    jobs = [(d, k, 20261003 + 100 * i) for i, (d, k) in enumerate((d, k) for d in (95.0, 90.0) for k in ("S slow release", "O spread opening", "M mixed"))]
    with Pool(6) as pool:
        for lines in pool.imap(run, jobs):
            print("\n".join(lines), flush=True)
