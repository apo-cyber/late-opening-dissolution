"""Comparison at equal quality, in the framework of Novick et al. 2009 (main text: Results, "Lots of equal quality",
its table and figure).

Novick et al. 2009 (delivered-dose uniformity of inhalation products) compared the probability of acceptance between
distributions with the same fraction outside the target interval (the same tail fraction), asking whether a lot of the
same quality passes more easily when the shape of the distribution differs (the consumer's risk).
Here the same is done for dissolution. The measure of quality is the unit failure fraction p = P(unit value < limit R).
All values are assumed.

Distribution types (each with p set to 0.01-0.30)
 normal SD 3 / SD 6: an ordinary immediate-release product (only the mean moves: mu = R - SD * Phi^{-1}(p))
 S slow release, O spread opening, M mixed: the unit model (the same types as regime_diagnostic.py; p is matched
     through the extent of ageing)
Specification (nominal, typical values): R = 75 % at 30 min. Interpretation 1 with Q = 70 (S1 at Q + 5 = 75 = R),
Interpretation 2 with the limit R. Analytical error SD 2 %.

p_Q: since Interpretation 1 is written in terms of Q, a comparison with quality matched as P(unit value < Q) is also given.
Expectation (written before the calculation): Interpretation 2 depends only on the number of units below R, so its
probability of passing is a function of p alone and does not depend on the shape (exactly equal to the binomial value).
Interpretation 1 has the mean clause and the Q - 15 and Q - 25 clauses, so the shape matters.
"""
import numpy as np
from scipy.stats import norm, binom
from scipy.optimize import brentq
from regime_diagnostic import qmod, verdicts, S_MEAS, R2, Q1, T_SPEC, L0

rng = np.random.default_rng(20261001)
N_LOT = 200_000
P_GRID = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
Z = np.random.default_rng(11)
ZL, ZK, ZE = Z.standard_normal(400_000), Z.standard_normal(400_000), Z.standard_normal(400_000)


def th_of(kind, x):
    if kind == "S slow release":
        return (np.log(L0), 0.13, x, 0.28)
    if kind == "O spread opening":
        return (x, 0.45, np.log(0.25), 0.10)
    return (np.log(L0) + 0.6 * x, 0.30, np.log(0.25) - x, 0.20)


def p_unit(th, limit=R2):
    mL, sL, mk, sk = th
    q = qmod(T_SPEC, np.exp(mL + sL * ZL), np.exp(mk + sk * ZK)) + S_MEAS * ZE
    return (q < limit).mean(), (q < Q1 - 25).mean()


def solve(kind, p, limit=R2):
    lo, hi = {"S slow release": (np.log(0.3), np.log(0.01)), "O spread opening": (np.log(5.0), np.log(80.0)), "M mixed": (0.0, 3.0)}[kind]
    return th_of(kind, brentq(lambda x: p_unit(th_of(kind, x), limit)[0] - p, lo, hi, xtol=1e-4))


def oc_lag(th):
    mL, sL, mk, sk = th
    L = np.exp(mL + sL * rng.standard_normal((N_LOT, 24))); k = np.exp(mk + sk * rng.standard_normal((N_LOT, 24)))
    q = qmod(T_SPEC, L, k) + S_MEAS * rng.standard_normal((N_LOT, 24))
    return 1 - verdicts(q)  # probability of passing [Interpretation 1, Interpretation 2]


def oc_normal(p, sd, limit=R2):
    mu = limit - sd * norm.ppf(p)
    q = mu + sd * rng.standard_normal((N_LOT, 24))
    return 1 - verdicts(q), mu


def pass2_exact(p):
    return (1 - p) ** 6 + 6 * p * (1 - p) ** 5 * binom.cdf(1, 6, p) + 15 * p**2 * (1 - p) ** 4 * (1 - p) ** 6


if __name__ == "__main__":
    print(f"Measure of quality: unit failure fraction p = P({T_SPEC:.0f}-min value < {R2:.0f} %). Probability of passing = 1 - probability of failing. {N_LOT} lots\n")
    rows = {}
    for p in P_GRID:
        r = {}
        for sd in (3.0, 6.0):
            oc, mu = oc_normal(p, sd)
            r[f"normal SD{sd:.0f}"] = (oc, f"mean {mu:.1f}")
        for kind in ("S slow release", "O spread opening", "M mixed"):
            th = solve(kind, p)
            deep = p_unit(th)[1]
            r[kind] = (oc_lag(th), f"below Q-25 {deep:.3f}")
        rows[p] = r
    for j, name in ((1, "Interpretation 2 (individual values, 6 -> 12)"), (0, "Interpretation 1 (Q value, S1-S3)")):
        print(f"Probability of passing {name}")
        print("   p     binomial (exact)" + "".join(f" {k:>12s}" for k in rows[P_GRID[0]]))
        for p in P_GRID:
            print(f"  {p:.2f}   {pass2_exact(p):.3f}           " + "".join(f" {rows[p][k][0][j]:12.3f}" for k in rows[p]))
        print()
    print("Reference: state of each type (normal: mean; unit-model types: fraction of units below Q - 25)")
    for p in P_GRID:
        print(f"  p {p:.2f}: " + ", ".join(f"{k} {v[1]}" for k, v in rows[p].items()))
    print()
    print(f"Interpretation 1 compared at equal quality measured by its own specification Q = {Q1:.0f} %: p_Q = P({T_SPEC:.0f}-min value < Q)")
    print("   p_Q   " + "".join(f" {k:>12s}" for k in rows[P_GRID[0]]))
    for p in P_GRID:
        vals = []
        for sd in (3.0, 6.0):
            vals.append(oc_normal(p, sd, Q1)[0][0])
        for kind in ("S slow release", "O spread opening", "M mixed"):
            vals.append(oc_lag(solve(kind, p, Q1))[0])
        print(f"  {p:.2f}  " + "".join(f" {v:12.3f}" for v in vals))
