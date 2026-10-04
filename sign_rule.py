"""Numerical checks of the propositions behind the sign rule and the four-class expression (main text: Results,
"The direction of the error"; Supplementary Section S4). The derivations are in sign_rule.md.

Two-point limit model: a unit value X lies, with probability ε, at a low value a (unit not opened), and with probability
1 − ε in an upper cluster N(m, s²). Gap D = m − a, relative depth of a threshold c δ = (m − c)/D (0 < δ < 1), κ = s/D.
The normal distribution with the same mean and SD: μ = m − εD, σ² = D²{(1 − ε)κ² + ε(1 − ε)}.

Proposition 1 (sign rule for the threshold depth): as κ → 0, the true fraction is F(c) = ε and the normal fraction is
  G(c) = Φ(−(δ − ε)/√(ε(1 − ε))). G(c) > F(c) ⇔ δ < δ*(ε) = ε + z_{1−ε}·√(ε(1 − ε)). The normal calculation overstates
  the fraction below a shallow threshold and understates it below a deep one. For fixed δ and ε → 0, G/F → 0
  (G vanishes at the rate exp(−δ²/2ε)).
Proposition 2 (Interpretation 2): the probability of an OOS result is an increasing function h(p) of p (the fraction
  below the limit R), so the sign of the error of the normal calculation is the sign of G(R) − F(R), decided by
  Proposition 1 through δ_R versus δ*(ε). In the two-point limit h(ε) ≈ 200 ε³.
Proposition 3 (Interpretation 1, two-cluster limit): if a < Q − 25 and m − s·(a large number) ≥ Q + 5, the true
  probability of failing tends to 1 − (1 − ε)^6 (once an unopened unit is among the first 6, S2 and S3 cannot rescue
  the lot). For the normal calculation
  P_N(fail) ≤ Φ(−√24 (μ − Q)/σ) + 24·G(Q − 25) + C(24, 3)·G(Q − 15)³ → 0 (ε → 0, faster than any power).
Corollary (four-class expression): when the mean clauses do not act (mean well above Q), the probability of passing
  Interpretation 1 is a closed-form function of the probabilities (pA, pB, pC, pD) of the four classes of unit values
  A ≥ Q + 5, B ∈ [Q − 15, Q + 5), C ∈ [Q − 25, Q − 15), D < Q − 25 only.

Checks
 1. Table of δ*(ε), and the exact (numerical) position of the sign change for κ > 0.
 2. In the unit model (the types of regime_diagnostic.py and 400 random states), the proportion of cases in which the
    sign predicted from the two-point fit (ε, δ) agrees with the simulated sign.
 3. The limit of Proposition 3 and the four-class expression against simulation (200,000 lots).
 4. Accuracy of the asymptotic form in Proposition 1 (G ≈ √ε·φ(δ/√ε)/δ).
"""
import numpy as np
from math import comb
from scipy.stats import norm, binom
from scipy.optimize import brentq
from regime_diagnostic import qmod, verdicts, S_MEAS, T_SPEC, R2, Q1, make_types

rng = np.random.default_rng(20261001)


def delta_star(eps):
    return eps + norm.ppf(1 - eps) * np.sqrt(eps * (1 - eps))


def FG_two_point(eps, delta, kappa):
    """True fraction F and normal fraction G for the two-point model with upper-cluster SD κD (scaled to D = 1).
    The lower cluster is a point."""
    F = eps + (1 - eps) * norm.cdf(-delta / kappa) if kappa > 0 else eps
    sig = np.sqrt((1 - eps) * kappa**2 + eps * (1 - eps))
    G = norm.cdf(-(delta - eps) / sig)
    return F, G


def h2(p):
    """Probability of an OOS result under Interpretation 2 (p = fraction below the limit)."""
    ok = (1 - p) ** 6 + 6 * p * (1 - p) ** 5 * binom.cdf(1, 6, p) + 15 * p**2 * (1 - p) ** 4 * (1 - p) ** 6
    return 1 - ok


def pass1_four_class(pA, pB, pC, pD):
    """Probability of passing Interpretation 1 when the mean clauses do not act. Classes as in the main text."""
    AB = pA + pB
    p1 = pA**6
    p2 = AB**12 - pA**6 * AB**6
    p3 = 0.0
    blk = lambda n, c: comb(n, c) * pC**c * AB ** (n - c)
    for c1 in range(3):
        for c2 in range(3):
            for c3 in range(3):
                if c1 + c2 < 1 or c1 + c2 + c3 > 2:
                    continue
                first = blk(6, c1) if c1 >= 1 else (AB**6 - pA**6)
                p3 += first * blk(6, c2) * blk(12, c3)
    return p1 + p2 + p3


def units(th, n):
    mL, sL, mk, sk = th
    L = np.exp(mL + sL * rng.standard_normal(n)); k = np.exp(mk + sk * rng.standard_normal(n))
    return qmod(T_SPEC, L, k) + S_MEAS * rng.standard_normal(n)


def two_point_fit(x, split):
    """Split the values into two clusters at split and return ε, a, m and s."""
    lo, hi = x[x < split], x[x >= split]
    eps = lo.size / x.size
    a = lo.mean() if lo.size else 0.0
    return eps, a, hi.mean(), hi.std()


def part1():
    print("1. Critical threshold depth δ*(ε) = ε + z_{1-ε}√(ε(1-ε)) (κ → 0), and the position of the sign change for κ > 0 (numerical)")
    print("   ε       δ*(κ→0)   κ=0.02    κ=0.05    κ=0.10")
    for eps in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20):
        row = [delta_star(eps)]
        for kappa in (0.02, 0.05, 0.10):
            f = lambda d: np.subtract(*FG_two_point(eps, d, kappa)[::-1])  # G - F
            try:
                row.append(brentq(f, 1e-4 + 3 * kappa, 0.999))
            except ValueError:
                row.append(np.nan)
        print(f"  {eps:.3f}   " + "   ".join(f"{v:7.3f}" for v in row))
    a, m = 0.0, 98.0
    print(f"   Reference: with a = {a:.0f} and m = {m:.0f}, the limit R = {R2:.0f} has δ_R = {(m - R2) / (m - a):.3f}, "
          f"Q - 15 = {Q1 - 15:.0f} has δ = {(m - Q1 + 15) / (m - a):.3f}, Q - 25 = {Q1 - 25:.0f} has δ = {(m - Q1 + 25) / (m - a):.3f}")
    print()


def part2(n_random=400):
    print("2. Unit model: sign predicted from the two-point fit (ε, δ) against the simulated sign")
    states = list(make_types().values())
    r = np.random.default_rng(3)
    for _ in range(n_random):
        states.append((np.log(r.uniform(6, 30)), r.uniform(0.1, 0.6), np.log(r.uniform(0.03, 0.4)), r.uniform(0.05, 0.35)))
    ok_R = ok_deep = n_R = n_deep = 0
    rows = []
    for th in states:
        x = units(th, 200_000)
        F = lambda c: (x < c).mean()
        mu, sd = x.mean(), x.std()
        G = lambda c: norm.cdf((c - mu) / sd)
        eps, a, m, s = two_point_fit(x, (Q1 - 25 + Q1 - 15) / 2)  # boundary between the lower and upper clusters at Q - 20
        if eps < 0.002 or eps > 0.4:
            continue
        D = m - a
        dstar = delta_star(eps)
        for c, kind in ((R2, "R"), (Q1 - 15, "deep")):
            tF, tG = F(c), G(c)
            if max(tF, tG) < 0.002 or abs(tF - tG) < 0.2 * max(tF, tG):
                continue  # states in which the two hardly differ are not counted
            pred_over = (m - c) / D < dstar
            if kind == "R":
                n_R += 1; ok_R += pred_over == (tG > tF)
            else:
                n_deep += 1; ok_deep += pred_over == (tG > tF)
        rows.append((eps, (m - R2) / D, dstar, F(R2), G(R2), F(Q1 - 15), G(Q1 - 15)))
    print(f"   Limit R: prediction correct in {ok_R}/{n_R}; deep threshold Q - 15: {ok_deep}/{n_deep} (states with a difference below 20 % excluded)")
    print("   Examples (the first 9 states fitted by two points; the slow-release states have no lower cluster and are excluded by ε < 0.002):  ε   δ_R   δ*   F(R)  G(R)  F(Q-15)  G(Q-15)")
    for r_ in rows[:9]:
        print("    " + "  ".join(f"{v:.3f}" for v in r_))
    print()


def part3():
    print("3. Probability of failing Interpretation 1: simulation, four-class expression, limit 1 - (1 - ε)^6 (ε = fraction below Q - 25)")
    print("   type, p                simul.   four-class  limit     Interpretation 2: simul.  h(p_R)   (p_R = fraction below R)")
    for (kind, p), th in make_types().items():
        x = units(th, 400_000)
        pA = (x >= Q1 + 5).mean(); pD = (x < Q1 - 25).mean(); pC = ((x >= Q1 - 25) & (x < Q1 - 15)).mean(); pB = 1 - pA - pC - pD
        mL, sL, mk, sk = th
        L = np.exp(mL + sL * rng.standard_normal((200_000, 24))); k = np.exp(mk + sk * rng.standard_normal((200_000, 24)))
        q = qmod(T_SPEC, L, k) + S_MEAS * rng.standard_normal((200_000, 24))
        sim1, sim2 = verdicts(q)
        print(f"  {kind} {p:.2f}   {sim1:.3f}    {1 - pass1_four_class(pA, pB, pC, pD):.3f}     {1 - (1 - pD) ** 6:.3f}"
              f"          {sim2:.3f}   {h2((x < R2).mean()):.3f}")
    print("   Interpretation 2, two-point limit h(ε) and 200ε³: " + ", ".join(f"ε {e}: {h2(e):.4f} / {200 * e**3:.4f}" for e in (0.01, 0.03, 0.05)))
    print()


def part4():
    print("4. Asymptotic form in Proposition 1: G = Φ(-(δ-ε)/√(ε(1-ε))) and the Mills-ratio approximation φ(x)/x (x = (δ-ε)/√(ε(1-ε)))")
    for delta in (0.45, 0.55):
        for eps in (0.005, 0.01, 0.03):
            x = (delta - eps) / np.sqrt(eps * (1 - eps))
            G = norm.cdf(-x); approx = norm.pdf(x) / x
            print(f"   δ {delta:.2f}  ε {eps:.3f}: G = {G:.3e}, approximation {approx:.3e}, ratio to the true fraction ε: G/ε = {G / eps:.2e}")
    print()


if __name__ == "__main__":
    print(f"Specification (nominal): {T_SPEC:.0f} min, R = {R2:.0f} %; Interpretation 1: Q = {Q1:.0f}\n")
    part1(); part3(); part4(); part2()
