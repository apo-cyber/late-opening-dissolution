"""Plateau D_max below 100 %, one pull: how far off is the estimated probability of failing when the unit model is fitted
with D_max fixed at 100 % instead of the true plateau? (Main text: limitations, plateau D_max; Supplementary Information.)

The product levels off at TRUE_D % (default 95; pass another value as the first argument, e.g. 90). The three ageing
types (S slow release, O spread opening, M mixed) are recalibrated under the true plateau to the state with a fraction
p = 0.10 of units below R (regime_diagnostic.make_types()). Each test is one pull of 12 units measured at 10, 20 and
30 min. The unit model is fitted twice, with D_max = 100 and with D_max = TRUE_D, and the probability of failing
(Interpretation 1 and Interpretation 2) computed from each fit is compared with the true one: median |log ratio|
(0.001 added), and the median estimated k. 40 tests per type. All values are assumed; no measured data.
"""
import sys, numpy as np
sys.path.insert(0, '.')
import likelihood as LK
import regime_diagnostic as RD
TRUE_D = float(sys.argv[1]) if len(sys.argv) > 1 else 95.0
gen = np.random.default_rng(20261003)
def pfail(th, dmax, n=40_000):
    LK.DMAX = dmax
    mL, sL, mk, sk = th
    L = np.exp(mL + sL * gen.standard_normal((n, 24))); k = np.exp(mk + sk * gen.standard_normal((n, 24)))
    return RD.verdicts(RD.qmod(RD.T_SPEC, L, k) + RD.S_MEAS * gen.standard_normal((n, 24)))
LK.DMAX = TRUE_D
types = RD.make_types()
NREP = 40
for name in ("S slow release", "O spread opening", "M mixed"):
    th = types[(name, 0.10)]
    truth = pfail(th, TRUE_D, 200_000)
    err = {100.0: [], TRUE_D: []}; kest = {100.0: [], TRUE_D: []}
    for r in range(NREP):
        LK.DMAX = TRUE_D
        y = RD.draw_units(th, 12, RD.TIMES)
        for d in (100.0, TRUE_D):
            LK.DMAX = d
            est = RD.fit(y, RD.TIMES)
            pf = pfail(est, d)
            err[d].append(np.abs(np.log((pf + 1e-3) / (truth + 1e-3))))
            kest[d].append(np.exp(est[2]))
    print(f"{name} p=0.10 true [{truth[0]:.3f}, {truth[1]:.3f}]  true median k {np.exp(th[2]):.3f}")
    for d in (100.0, TRUE_D):
        e = np.array(err[d]); print(f"   fitted with D_max={d:5.1f}: median |log ratio| [Int. 1 {np.median(e[:,0]):.2f}, Int. 2 {np.median(e[:,1]):.2f}]  median estimated k {np.median(kest[d]):.3f}", flush=True)
