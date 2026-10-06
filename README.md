# Late-opening units and the probability of failing dissolution tests — code

Code for the paper *Late-opening units and the probability of failing dissolution tests* (Y. Arai, 2026, submitted).
Self-contained Python scripts (NumPy + SciPy; Matplotlib for the figures) produce every table and figure of the paper
and of its Supplementary Information. All simulated values are assumed; no measured data are used except the published
values of Nagase et al. (1976) and Nakai et al. (1974), transcribed in `nagase1976.py` and `nakai1974.py`.

The unit model: each unit releases nothing until it opens at time $L$ and then follows
$D(t) = D_{\max}\{1 - e^{-k(t - L)}\}$, with $\ln L$ and $\ln k$ normal and independent across units and an analytical
error of SD 2 % on each observed value. `likelihood.py` is the single source of the model and of its likelihood;
`regime_diagnostic.py` defines the three ageing types, the two acceptance procedures of the Japanese Pharmacopoeia
(Interpretation 1, the harmonized three-stage procedure; Interpretation 2, the individual-value procedure) and the
normal calculation from the mean and standard deviation.

| Script | Paper | What it does |
|---|---|---|
| `likelihood.py` | Methods (estimation), SI S3 | Unit model; likelihood of per-vessel values integrated on local grids, with analytical gradient; maximum-likelihood fits for one pull and for pooled pulls (parameters linear in time). Run alone, it refines the grid and prints the change in the likelihood and in the fit |
| `regime_diagnostic.py` | Tables 1, 2 and 5 | Ageing types (slow release, spread opening, mixed) at three degrees of ageing; true and normal probabilities of failing both procedures; diagnosis of the regime from 12 units of one pull (simple flag, model with 30-min values only, model with 10, 20 and 30 min) and error of the estimated probability of failing |
| `single_peak_controls.py`, `skew_baseline.py` | Table 3 | Other single-peak distributions matched to the unit values (lognormal, reflected lognormal, skew-normal) |
| `sign_rule.py`, `sign_rule.md` | Direction of the error; SI S4 | Two-point limit: threshold depth, the sign rule and the four-class expression for Interpretation 1 (derivation in `sign_rule.md`), checked on the states of the unit model |
| `equal_quality.py` | Table 4, Figure 3 | Probability of passing for lots of equal quality (fraction of units below $Q$) and different shapes |
| `pooled_pulls.py` | Table 6 | Forecasts of the probability of failing at 12 and 15 months from pulls at 0–12 months: normal trend versus the pooled unit model |
| `form_test.py`, `form_test_control.py`, `form_test_restart.py` | SI S1 | Likelihood-ratio test of the form of the path (quadratic against linear), the rule that switches to the quadratic forecast, two further forms of acceleration, and the same fits repeated from more starting values |
| `plateau_single.py`, `plateau_pooled.py` | SI S2 | Plateau of 95 % or 90 % fitted with $D_{\max}$ = 100 % or the true value, from one pull and from pooled pulls |
| `gradual_start.py` | SI S5 | Units whose release starts gradually (Weibull shape 1.5 and 2) fitted with the unit model |
| `crack_not_widening.py` | SI S6 | A small fraction of units whose crack does not widen, added to each ageing type |
| `trend_detection.py` | SI S7 | Pulls at 0, 3 and 6 months while all units pass: the test of no change (pooled linear against constant), regressions of the 30-min and 10-min values on time, and which parameter moved |
| `nagase1976.py`, `nakai1974.py` | Published data and published lots, Figure 4 | Curves of four tablets digitised from Figure 8 of Nagase et al. (1976) and the summary statistics of Nakai et al. (1974); fits and probabilities of failing |
| `figures.py`, `graphical_abstract.py`, `graphical_abstract_ddip.py` | Figures 1–4, graphical abstract (full size; 525 pixels wide for the journal) | Figures (PNG, written to `../figures` or `$PAPERE_FIG_DIR`; the exported copy keeps them in `figures/`) |

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
ONLY_FAST=1 ./run_all.sh   # a few minutes: Tables 3 and 4, the sign rule, the published lots, the figures
./run_all.sh               # everything; the simulations of Tables 1, 2, 5, 6 and SI S1, S2 and S5–S7 take hours each
```

The `.txt` files in `output/` are the runs used for the paper; the seeds are fixed in the scripts. The simulations use
`multiprocessing` (up to 9 processes). Tested with Python 3.12, NumPy 2.4, SciPy 1.17 and Matplotlib 3.10.

## Data

`nagase1976.py` holds the dissolution curves of four tablets on opening and after 80 days at 25 °C and 81 % RH, read by
eye from Figure 8 (lower panel) of Nagase I, Fujishiro T, Okamoto T, Nakajima S. *Jpn J Hosp Pharm* 1976;1:201–207
(https://doi.org/10.5649/jjphcs1975.1.201); the reading error is about ±1 min and ±3 %. `nakai1974.py` uses the means and
standard deviations of the time to complete dissolution reported in Table III of Nakai Y, Nakajima S, Kakizawa H.
*Chem Pharm Bull* 1974;22:2910–2915 (https://doi.org/10.1248/cpb.22.2910). The articles themselves are not redistributed.

## Licence and citation

MIT licence (see `LICENSE`). Please cite the paper and the Zenodo archive of this repository (`CITATION.cff`).
