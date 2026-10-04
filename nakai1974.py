"""Worked example from published summary statistics: the sugar-coated tablets of Nakai et al. (1974), whose opening times
spread on storage, a public example of the two-cluster regime (main text: "Published data" in Materials and Methods,
"Published lots" in Results).

Source: Nakai Y, Nakajima S, Kakizawa H. Disintegration measurement of sugar coated tablets by thermal analysis.
Chem Pharm Bull 1974;22(12):2910-2915, doi:10.1248/cpb.22.2910, Table III: time to complete dissolution t_e (min, acid
medium) read from the heat curves of a twin calorimeter. Sugar-coated tablet A (with a cellulose acetate phthalate
moisture-barrier subcoat): 56 +/- 7 min (14 tablets) initially -> 76 +/- 37 min (6 tablets) after 18 months at room
temperature in a well-closed container. Tablet B (no subcoat): 23 +/- 3 -> 24 +/- 4 min. Individual values are not
reported; only these means and SDs are used.

Reconstruction (assumptions stated)
 - t_e = L + Delta, where L is the opening time and Delta the time from opening to complete dissolution (95 % for
   first-order release: Delta = ln 20 / k). Delta is taken to be the same for every unit. Delta is unknown and is set
   to 10 or 20 min (the 23 min of tablet B is a rough upper guide).
 - Two distributions of L:
   (i)  lognormal: mean = mean of t_e - Delta, SD = SD of t_e (matched moments).
   (ii) mixture: most units keep the initial distribution shifted, and a fraction pi of units opens much later (units
        whose coat does not come off or open). An SD of 37 from 6 tablets can arise from a single outlying tablet
        (e.g. 5 tablets at 62 min and 1 at 146 min give mean 76 and SD 34). pi = 1/6; L of the late units is chosen to
        match the mean and SD.
 - The 18-month SD comes from 6 tablets, so the lower end of its 90 % interval (chi-squared; 25 min) is also checked
   (the side unfavourable to the claim).
 - Intermediate pulls: the parameters are interpolated linearly from the initial state to 18 months (f = 0 ... 1).
   This is an assumption about the shape of the path, not a measurement.
 - The specification is nominal (not that of this product): 75 % at t min, t = 90 and 120 min. Interpretation 1 with
   Q = 70 (stage 1 at 75). Analytical error SD 2 %; no release before opening.
"""
import numpy as np
from scipy.stats import chi2
from nagase1976 import model, verdicts

rng = np.random.default_rng(20260930)
N_MC = 200_000
S_MEAS = 2.0
TE0, SD0, TE18, SD18, N18 = 56.0, 7.0, 76.0, 37.0, 6


def lognorm_par(m, s):
    s2 = np.log(1 + (s / m) ** 2)
    return np.log(m) - s2 / 2, np.sqrt(s2)


def draw_L(f, delta, shape, sd18, size):
    """f: 0 on opening -> 1 at 18 months."""
    m0 = TE0 - delta
    mu0, s0 = lognorm_par(m0, SD0)
    if shape == "lognormal":
        mu1, s1 = lognorm_par(TE18 - delta, sd18)
        mu, s = mu0 + f * (mu1 - mu0), s0 + f * (s1 - s0)
        return np.exp(mu + s * rng.standard_normal(size))
    # Mixture: most units keep the initial shape and only shift; a fraction pi opens much later. Mean and SD matched at 18 months.
    pi = 1 / 6
    # Main component: the initial distribution shifted by +d. Late component: mean m_late, SD as initially.
    # Mean: (1-pi)(m0+d) + pi m_late = TE18 - Delta; variance: SD0^2 + pi(1-pi)(m_late - m0 - d)^2 = sd18^2
    gap = np.sqrt(max(sd18**2 - SD0**2, 0) / (pi * (1 - pi)))
    d = (TE18 - delta) - m0 - pi * gap
    base = np.exp(mu0 + s0 * rng.standard_normal(size)) + f * d
    is_late = rng.random(size) < pi
    return base + f * gap * is_late


def run(t_spec, delta, shape, sd18, R=75.0, fs=np.round(np.arange(0, 1.01, 0.1), 2)):
    k = np.log(20) / delta
    rows = []
    for f in fs:
        L = draw_L(f, delta, shape, sd18, (N_MC, 24))
        q = model(t_spec, L, k) + S_MEAS * rng.standard_normal(L.shape)
        unopened = (L > t_spec - np.log((100) / (100 - R)) / k).mean()  # units that cannot reach R % by the specification time
        tr = verdicts(q, R)
        qn = q.mean() + q.std() * rng.standard_normal(q.shape)
        nr = verdicts(qn, R)
        low = (q < 20).mean()  # units with almost nothing released (the lower of the two clusters)
        rows.append((f, q.mean(), q.std(), unopened, low, tr, nr))
    return rows


def show(rows, label):
    print(label)
    print("   f    mean    SD  unit p  <20 % |  Int. 2: true -> normal |  Int. 1: true -> normal")
    for f, m, v, p, low, tr, nr in rows:
        if max(tr + nr) < 1e-4:
            continue
        print(f"  {f:.1f}  {m:5.1f} {v:5.1f}  {p:.3f}  {low:.3f}  |  {tr[0]:.3f} -> {nr[0]:.3f}         |  {tr[1]:.3f} -> {nr[1]:.3f}")
    print()


if __name__ == "__main__":
    lo = SD18 * np.sqrt((N18 - 1) / chi2.ppf(0.95, N18 - 1))
    hi = SD18 * np.sqrt((N18 - 1) / chi2.ppf(0.05, N18 - 1))
    print(f"18-month SD 37 min (6 tablets), 90 % interval: {lo:.1f} to {hi:.1f} min")
    for delta in (10.0, 20.0):
        m0, s0 = lognorm_par(TE0 - delta, SD0); m1, s1 = lognorm_par(TE18 - delta, SD18)
        print(f"Delta {delta:.0f} min (k {np.log(20)/delta:.3f}/min): median L on opening {np.exp(m0):.1f} min, sigma {s0:.3f} -> at 18 months {np.exp(m1):.1f} min, sigma {s1:.3f}")
    print()
    for t_spec in (90.0, 120.0):
        for delta in (10.0, 20.0):
            show(run(t_spec, delta, "lognormal", SD18), f"Lognormal, specification at {t_spec:.0f} min: 75 %, Delta {delta:.0f} min")
    print("Sensitivity")
    show(run(90.0, 10.0, "lognormal", lo), f"Lognormal, SD at the lower end of the 90 % interval, {lo:.1f} min (the side unfavourable to the claim), 90 min, Delta 10")
    show(run(90.0, 10.0, "mixture", SD18), "Mixture (1/6 of the units open much later), 90 min, Delta 10")
    show(run(120.0, 20.0, "mixture", SD18), "Mixture, 120 min, Delta 20")
    show(run(90.0, 10.0, "lognormal", SD18, R=80.0), "Lognormal, 90 min: 80 %, Delta 10")
