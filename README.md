# Late-opening units and the probability of failing dissolution tests — code

Code for the paper *Late-opening units and the probability of failing dissolution tests* (Y. Arai, 2026, submitted).
Self-contained Python scripts (NumPy + SciPy; Matplotlib for the figures) produce every number, table and figure of the
paper and the further checks summarized in its text. All simulated values are assumed; no measured data are used except the published
values of Nagase et al. (1976) and Nakai et al. (1974), transcribed in `nagase1976.py` and `nakai1974.py`.

The unit model: each unit releases nothing until it opens at time $L$ and then follows
$D(t) = D_{\max}\{1 - e^{-k(t - L)}\}$, with $\ln L$ and $\ln k$ normal and independent across units and an analytical
error of SD 2 % on each observed value. `likelihood.py` is the single source of the model and of its likelihood;
`regime_diagnostic.py` defines the three ageing types, the two acceptance procedures of the Japanese Pharmacopoeia
(Interpretation 1, the harmonized three-stage procedure; Interpretation 2, the individual-value procedure) and the
normal calculation from the mean and standard deviation.

Numbers and section names below follow the version of the paper submitted to *Dissolution Technologies* (2026), which
has no supplementary material: the checks that versions 1.0.1–1.0.3 of this code listed as Supplementary Sections
S1–S8 are summarized in its text, and the scripts that produce them are kept here (the former section is given in
brackets).

| Script | Paper | What it does |
|---|---|---|
| `likelihood.py` | Methods, "Calculations" [S3] | Unit model; likelihood of per-vessel values integrated on local grids, with analytical gradient; maximum-likelihood fits for one pull and for pooled pulls (parameters linear in time). Run alone, it refines the grid and prints the change in the likelihood and in the fit |
| `regime_diagnostic.py` | Table 1; Results, "True and normal probabilities of failing"; Table 2 | Ageing types (slow release, spread opening, mixed) at three degrees of ageing; true and normal probabilities of failing both procedures; diagnosis of the regime from 12 units of one pull (simple flag, model with 30-min values only, model with 10, 20 and 30 min) and error of the estimated probability of failing |
| `single_peak_controls.py`, `skew_baseline.py` | Results, "True and normal probabilities of failing" (other single-peak families) [S8, Table S4] | Other single-peak distributions matched to the unit values (lognormal, reflected lognormal, skew-normal), and the reflected lognormal fitted to the sample skewness of one pull |
| `sign_rule.py`, `sign_rule.md` | Results, "The direction of the error" [S4] | Two-point limit: threshold depth, the sign rule and the four-class expression for Interpretation 1 (derivation in `sign_rule.md`), checked on the states of the unit model |
| `equal_quality.py` | Results, "Lots of equal quality" | Probability of passing for lots of equal quality and different shapes (its table and figure, Table 3 and Figure 3 in versions 1.0.1–1.0.3, are not in the current version) |
| `pooled_pulls.py` | Table 3 | Forecasts of the probability of failing at 12 and 15 months from pulls at 0–12 months: normal trend versus the pooled unit model |
| `form_test.py`, `form_test_control.py`, `form_test_restart.py` | Results, "Forecasts"; Discussion, limitations [S1] | Likelihood-ratio test of the form of the path (quadratic against linear), the rule that switches to the quadratic forecast, two further forms of acceleration, and the same fits repeated from more starting values |
| `plateau_pooled.py`, `plateau_single.py` | Results, "Forecasts" (pooled pulls; the one-pull result is not reported) [S2] | Plateau of 95 % or 90 % fitted with $D_{\max}$ = 100 % or the true value, from pooled pulls and from one pull |
| `gradual_start.py` | Discussion, limitations [S5] | Units whose release starts gradually (Weibull shape 1.5 and 2) fitted with the unit model |
| `crack_not_widening.py` | Discussion, limitations [S6] | A small fraction of units whose crack does not widen, added to each ageing type |
| `trend_detection.py` | Not reported in the current version [S7] | Pulls at 0, 3 and 6 months while all units pass: the test of no change (pooled linear against constant), regressions of the 30-min and 10-min values on time, and which parameter moved |
| `nagase1976.py`, `nakai1974.py` | Methods, "Published data"; Results, "Published lots"; Figure 3 | Curves of four tablets digitised from Figure 8 of Nagase et al. (1976) and the summary statistics of Nakai et al. (1974); fits and probabilities of failing |
| `figures.py` | Figures 1–3 (`PAPERE_FIG_STYLE=dt`) | With `PAPERE_FIG_STYLE=dt`: Figures 1–3 of the current version, about 5 inches wide with 14-point text and the panels stacked (`figures/dt/Figure1–3.png`). Without it: Figures 1–4 as numbered in versions 1.0.1–1.0.3 (`figures/fig1–fig4_*.png`; Figures 1, 2 and 4 there are Figures 1, 2 and 3 now). The simulated values are the same in both styles |
| `graphical_abstract.py`, `graphical_abstract_ddip.py` | Not used in the current version | Graphical abstract of the earlier version (full size, and 525 pixels wide in `figures/ddip/`) |

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
ONLY_FAST=1 ./run_all.sh   # a few minutes: other single-peak families, the sign rule, lots of equal quality, the published lots, the figures
./run_all.sh               # everything; the simulations of Tables 1–3 and of the checks on form, plateau, gradual start, cracks and trend take hours each
```

The `.txt` files in `output/` are the runs used for the paper; the seeds are fixed in the scripts. The simulations use
`multiprocessing` (up to 9 processes). Tested with Python 3.12, NumPy 2.4, SciPy 1.17 and Matplotlib 3.10.

## Data

`nagase1976.py` holds the dissolution curves of four tablets on opening and after 80 days at 25 °C and 81 % RH, read by
eye from Figure 8 (lower panel) of Nagase I, Fujishiro T, Okamoto T, Nakajima S. *Jpn J Hosp Pharm* 1976;1:201–207
(https://doi.org/10.5649/jjphcs1975.1.201); the reading error is about ±1 min and ±3 %. `nakai1974.py` uses the means and
standard deviations of the time to complete dissolution reported in Table III of Nakai Y, Nakajima S, Kakizawa H.
*Chem Pharm Bull* 1974;22:2910–2915 (https://doi.org/10.1248/cpb.22.2910). The articles themselves are not redistributed.

## Versions

- 1.0.4 (2026-10-24): numbers and section names in this README, `run_all.sh` and the docstrings follow the version of the paper submitted to *Dissolution Technologies*, which has no supplementary material (Tables 4 and 5 became Tables 2 and 3; Figure 4 became Figure 3; Tables 2 and 3 and Figure 3 of the earlier version are not in it; the former Supplementary Sections S1–S8 are listed by topic, with the old number in brackets). `figures.py`: `PAPERE_FIG_STYLE=dt` draws the figures of that version (`figures/dt/`); the default style is unchanged. No simulated value or script output changed.
- 1.0.3 (2026-10-22): table numbers in this README and `run_all.sh` follow the resubmitted paper, in which the comparison with other single-peak distributions moved to the supplementary material (SI S8, Table S4; Tables 4–6 of the paper became Tables 3–5). No code or output changed.
- 1.0.2 (2026-10-16): `sign_rule.md`, corollary (four-class expression): the error of the normal calculation comes *mainly*, not entirely, from the two deep classes; a lot with none in them always passes, but with units in class C the split between classes A and B also matters (worked example added). Graphical abstract at 525 pixels wide (`graphical_abstract_ddip.py`, `figures/ddip/`). No script output changed.
- 1.0.1 (2026-10-04): first archived version.

## Licence and citation

MIT licence (see `LICENSE`). Please cite the paper and the Zenodo archive of this repository (`CITATION.cff`).
