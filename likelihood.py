"""Likelihood of the unit model, integrated on local grids, and its maximization.

Unit model: D(t) = D_max {1 - exp(-k (t - L))} for t > L and 0 otherwise, with ln L ~ N(mu_L, sigma_L^2) and
ln k ~ N(mu_k, sigma_k^2) independent across units; each observed value has an analytical error N(0, S_MEAS^2).
Parameters theta = (mu_L, ln sigma_L, mu_k, ln sigma_k).

The likelihood of one unit is the integral over (ln L, ln k) of prod_t phi((y_t - D(t; L, k)) / s) times the lognormal
densities. With an analytical error of 2 % the integrand is sharply peaked, and a 16 x 16 Gauss-Hermite rule did not
converge (at the true parameters the negative log-likelihood of one data set moved from 361 to 240 as the nodes per axis
rose from 16 to 160). For each unit we therefore locate, on a coarse grid over a wide box, the region where the data
likelihood G is within exp(-CUT) of its maximum, place a fine grid over that region and evaluate G once (it does not
depend on theta). Each evaluation for a new theta only reweights those nodes by the normal densities and sums
log sum_j G_ij phi_theta(node j) dA_j over units; the gradient is analytical. check_convergence() refines the grid
and shows that the value at the true parameters and the fitted value do not change (Supplementary Section S3).
"""
import numpy as np
from scipy.optimize import minimize

S_MEAS = 2.0
BOX_L = (np.log(1.0), np.log(200.0))
BOX_K = (np.log(0.002), np.log(3.0))
N_COARSE, N_FINE, CUT = 200, 400, 20.0
MAX_NODES, N_FALLBACK = 60_000, 300
LOG2PI = np.log(2 * np.pi)


DMAX = 100.0  # plateau D_max (% of label claim); set e.g. LK.DMAX = 95.0 for a product that levels off at 95 %


def qmod(t, L, k, dmax=None):
    d = DMAX if dmax is None else dmax
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(t > L, d * (1 - np.exp(-k * np.clip(t - L, 0, None))), 0.0)


def _logG(y, times, lL, lk):
    L, k = np.exp(lL), np.exp(lk)
    out = np.zeros(np.broadcast(lL, lk).shape)
    for yt, t in zip(y, times):
        out += -0.5 * ((yt - qmod(t, L, k)) / S_MEAS) ** 2
    return out


class UnitGrids:
    """Local grids, one per unit. units: (n, number of time points). group: index of the stability pull of each unit
    (units in different pulls get different parameters). An n_fine x n_fine grid is placed over the local box and only
    the nodes whose data likelihood is within exp(-CUT) of the maximum are kept, so that narrow ridges are resolved
    and the number of nodes scales with the area of the ridge. Where more than MAX_NODES nodes remain (a flat
    plateau, which is smooth), an n_fallback x n_fallback grid is used instead."""

    def __init__(self, units, times, group=None, n_fine=N_FINE, n_coarse=N_COARSE, max_nodes=None, n_fallback=None):
        max_nodes = MAX_NODES if max_nodes is None else max_nodes
        n_fallback = N_FALLBACK if n_fallback is None else n_fallback
        units = np.atleast_2d(units)
        self.n = units.shape[0]
        self.group = np.zeros(self.n, int) if group is None else np.asarray(group)
        cl = np.linspace(*BOX_L, n_coarse); ck = np.linspace(*BOX_K, n_coarse)
        CL, CK = np.meshgrid(cl, ck, indexing="ij")
        dl, dk = cl[1] - cl[0], ck[1] - ck[0]
        xs, zs, gs, idx = [], [], [], []
        for i, y in enumerate(units):
            g = _logG(y, times, CL, CK)
            keep = g > g.max() - CUT
            lo_l, hi_l = CL[keep].min() - dl, CL[keep].max() + dl
            lo_k, hi_k = CK[keep].min() - dk, CK[keep].max() + dk
            for nf in (n_fine, n_fallback):
                fl = np.linspace(lo_l, hi_l, nf); fk = np.linspace(lo_k, hi_k, nf)
                FL, FK = np.meshgrid(fl, fk, indexing="ij")
                G = _logG(y, times, FL, FK).ravel()
                sel = G > G.max() - CUT
                if sel.sum() <= max_nodes or nf == n_fallback:
                    break
            area = (fl[1] - fl[0]) * (fk[1] - fk[0])
            xs.append(FL.ravel()[sel]); zs.append(FK.ravel()[sel]); gs.append(G[sel] + np.log(area))
            idx.append(np.full(sel.sum(), i))
        self.x, self.z, self.g = np.concatenate(xs), np.concatenate(zs), np.concatenate(gs)
        self.idx = np.concatenate(idx)
        self.starts = np.r_[0, np.cumsum([a.size for a in gs])[:-1]]

    def loglik(self, thetas, grad=False):
        """thetas: (number of groups, 4); unit i uses thetas[group[i]]. Returns the log-likelihood (and the gradient per group)."""
        th = np.asarray(thetas)[self.group][self.idx]
        mL, sL, mk, sk = th[:, 0], np.exp(th[:, 1]), th[:, 2], np.exp(th[:, 3])
        uL, uk = (self.x - mL) / sL, (self.z - mk) / sk
        lw = self.g - 0.5 * (uL**2 + uk**2) - np.log(sL) - np.log(sk) - LOG2PI
        mx = np.maximum.reduceat(lw, self.starts)
        e = np.exp(lw - mx[self.idx])
        s = np.add.reduceat(e, self.starts)
        ll = np.sum(np.log(s) + mx)
        if not grad:
            return ll
        w = e / s[self.idx]  # posterior weights of the nodes within each unit
        d = np.stack([uL / sL, uL**2 - 1, uk / sk, uk**2 - 1], 1) * w[:, None]
        per_unit = np.add.reduceat(d, self.starts)
        ng = np.asarray(thetas).shape[0]
        gr = np.zeros((ng, 4))
        np.add.at(gr, self.group, per_unit)
        return ll, gr


BOUND4 = [(np.log(2), np.log(120)), (np.log(0.03), np.log(1.2)), (np.log(0.005), np.log(2)), (np.log(0.03), np.log(1.2))]
STARTS4 = [(np.log(8), np.log(0.13), np.log(0.08), np.log(0.28)), (np.log(12), np.log(0.13), np.log(0.08), np.log(0.28)), (np.log(25), np.log(0.45), np.log(0.25), np.log(0.10)),
           (np.log(18), np.log(0.30), np.log(0.12), np.log(0.20)), (np.log(10), np.log(0.13), np.log(0.25), np.log(0.10))]


def fit_single(ug):
    def f(p):
        ll, g = ug.loglik(p[None], grad=True)
        return -ll, -g[0]
    best = None
    for s0 in STARTS4:
        r = minimize(f, s0, jac=True, method="L-BFGS-B", bounds=BOUND4)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def fit_pooled(ug, taus, starts):
    """Pooled linear fit: parameters of group j = a + b * taus[j]. starts: list of 8-parameter starting values."""
    taus = np.asarray(taus, float)
    def f(p):
        a, b = p[:4], p[4:]
        ll, g = ug.loglik(a[None] + taus[:, None] * b[None], grad=True)
        return -ll, -np.r_[g.sum(0), (g * taus[:, None]).sum(0)]
    bounds = BOUND4 + [(-0.5, 0.5)] * 4
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    best = None
    for s0 in starts:
        r = minimize(f, np.clip(s0, lo, hi), jac=True, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return best.x[:4], best.x[4:], best.fun


def check_convergence(seed=5):
    """How the negative log-likelihood at the true parameters and the fitted value change with the fineness of the grid."""
    rng = np.random.default_rng(seed)
    times3 = (10.0, 20.0, 30.0)
    for label, th in (("spread opening", (np.log(14.0), np.log(0.45), np.log(0.25), np.log(0.10))),
                      ("slow release", (np.log(8.0), np.log(0.13), np.log(0.071), np.log(0.28)))):
        L = np.exp(th[0] + np.exp(th[1]) * rng.standard_normal(12)); k = np.exp(th[2] + np.exp(th[3]) * rng.standard_normal(12))
        y3 = np.stack([qmod(t, L, k) + S_MEAS * rng.standard_normal(12) for t in times3], 1)
        for design, y, times in (("10, 20, 30 min", y3, times3), ("30 min only", y3[:, 2:], (30.0,))):
            row = []
            for nf in (200, 400, 800):
                ug = UnitGrids(y, times, n_fine=nf)
                p, fval = fit_single(ug)
                row.append(f"n{nf}: true {-ug.loglik(np.array(th)[None]):7.2f} fitted {fval:7.2f}")
            print(f"  {label}, {design}: " + " | ".join(row))


if __name__ == "__main__":
    check_convergence()
