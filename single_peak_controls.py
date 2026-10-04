"""Control: is the error of the normal calculation a general consequence of non-normality, or specific to two clusters?
(main text: Results, "Other single-peak distributions" and its table).

For the nine states of the three ageing types (regime_diagnostic.make_types), the unit values at the specification time
are compared with single-peak distributions having the same mean and SD, through the probability of failing the
staged procedures:
 normal               mean and SD
 lognormal            mean and SD (tail to the right; the shape compared with the normal by Saccone et al. 2004)
 reflected lognormal  mean, SD and skewness (c - Y with Y lognormal; tail to the left; matched up to the third moment)
 skew-normal          mean, SD and skewness (the skew-normal can only represent skewness |gamma| < 0.995, so larger
                      values are truncated at the boundary)
200,000 lots x 24 units each. The analytical error is included in the true values (they are values drawn from the
unit model as they are).
"""
import numpy as np
from scipy.stats import skew, skewnorm
from scipy.optimize import brentq
from regime_diagnostic import qmod, verdicts, S_MEAS, T_SPEC, R2, Q1, make_types

rng = np.random.default_rng(20261002)
N_LOT, N_REF = 200_000, 2_000_000


def unit_values(th, size):
    mL, sL, mk, sk = th
    L = np.exp(mL + sL * rng.standard_normal(size)); k = np.exp(mk + sk * rng.standard_normal(size))
    return qmod(T_SPEC, L, k) + S_MEAS * rng.standard_normal(size)


def lognormal(m, s, size):
    v = np.log(1 + (s / m) ** 2)
    return np.exp(np.log(m) - v / 2 + np.sqrt(v) * rng.standard_normal(size))


def reflected_lognormal(m, s, g, size):
    """c - Y with Y lognormal, matched to mean m, SD s and skewness g (< 0). The lognormal skewness
    (e^v + 2) sqrt(e^v - 1) = |g| is solved for v."""
    if g >= 0:
        return None
    v = brentq(lambda v: (np.exp(v) + 2) * np.sqrt(np.exp(v) - 1) - abs(g), 1e-10, 20)
    mY = s / np.sqrt(np.exp(v) - 1)
    c = m + mY
    Y = np.exp(np.log(mY) - v / 2 + np.sqrt(v) * rng.standard_normal(size))
    return c - Y, c


def skew_normal(m, s, g, size):
    gmax = 0.99
    g_used = np.clip(g, -gmax, gmax)
    # solve the skew-normal skewness gamma(delta) for delta; shape a = delta / sqrt(1 - delta^2)
    def gam(d):
        b = d * np.sqrt(2 / np.pi)
        return (4 - np.pi) / 2 * b**3 / (1 - b**2) ** 1.5
    d = brentq(lambda d: gam(d) - g_used, -0.999999, 0.999999)
    a = d / np.sqrt(1 - d**2)
    b = d * np.sqrt(2 / np.pi)
    scale = s / np.sqrt(1 - b**2)
    loc = m - scale * b
    return skewnorm.rvs(a, loc=loc, scale=scale, size=size, random_state=rng), g_used


def classes(x):
    return [(x >= Q1 + 5).mean(), ((x >= Q1 - 15) & (x < Q1 + 5)).mean(), ((x >= Q1 - 25) & (x < Q1 - 15)).mean(), (x < Q1 - 25).mean()]


if __name__ == "__main__":
    print(f"Specification at {T_SPEC:.0f} min: R = {R2:.0f} %, Q = {Q1:.0f} %. {N_LOT:,} lots each. Values: probability of failing [Interpretation 1, Interpretation 2]")
    print("Second block: probabilities of the 4 classes of unit values [>=Q+5, Q-15 to Q+5, Q-25 to Q-15, <Q-25] (they alone decide Interpretation 1 when the mean clauses do not act)\n")
    for (kind, p), th in make_types().items():
        ref = unit_values(th, N_REF)
        m, s, g = ref.mean(), ref.std(), skew(ref)
        out = {"true": verdicts(unit_values(th, (N_LOT, 24)))}
        cls = {"true": classes(ref)}
        x = m + s * rng.standard_normal((N_LOT, 24)); out["normal"] = verdicts(x); cls["normal"] = classes(m + s * rng.standard_normal(N_REF))
        out["lognormal"] = verdicts(lognormal(m, s, (N_LOT, 24))); cls["lognormal"] = classes(lognormal(m, s, N_REF))
        rl = reflected_lognormal(m, s, g, (N_LOT, 24))
        if rl is not None:
            out["reflected lognormal"] = verdicts(rl[0]); cls["reflected lognormal"] = classes(reflected_lognormal(m, s, g, N_REF)[0])
        sn, g_used = skew_normal(m, s, g, (N_LOT, 24))
        out["skew-normal"] = verdicts(sn); cls["skew-normal"] = classes(skew_normal(m, s, g, N_REF)[0])
        print(f"{kind} p {p:.2f}: mean {m:.1f}, SD {s:.1f}, skewness {g:+.2f}" + (f" (skew-normal truncated at {g_used:+.2f})" if abs(g) > 0.99 else ""))
        for name in out:
            v = out[name]; c = cls[name]
            print(f"   {name:8s} [{v[0]:.3f}, {v[1]:.3f}]   classes [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.4f}, {c[3]:.4f}]")
        print()
