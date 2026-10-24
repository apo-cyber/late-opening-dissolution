#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper and the further checks summarized in its text.
# Fast scripts take seconds to minutes; the simulations marked (hours) take several hours each on a laptop with 6 cores
# (they use multiprocessing). Set ONLY_FAST=1 to skip them; the saved outputs in output/ are the runs used for the paper.
# [Sn] gives the Supplementary Section of versions 1.0.1-1.0.3 of this code; the current paper has no supplementary material.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
# Figures go to ./figures in the standalone copy and to ../figures (the paper directory) in the paper repository.
if [ -z "${PAPERE_FIG_DIR:-}" ]; then
  if [ -d figures ]; then export PAPERE_FIG_DIR="$PWD/figures"; else export PAPERE_FIG_DIR="$PWD/../figures"; fi
fi
mkdir -p output
$PY likelihood.py           > output/likelihood.txt           # Methods, Calculations [S3]: convergence of the local-grid likelihood
$PY single_peak_controls.py > output/single_peak_controls.txt # Results: other single-peak families [S8, Table S4]
$PY skew_baseline.py        > output/skew_baseline.txt        # Results: reflected lognormal fitted to one pull [S8]
$PY equal_quality.py        > output/equal_quality.txt        # Results: lots of equal quality
$PY sign_rule.py            > output/sign_rule.txt            # Results: direction of the error (sign rule, four-class expression) [S4]
$PY nagase1976.py           > output/nagase1976.txt           # Published lots: Nagase et al. (1976), Figure 3
$PY nakai1974.py            > output/nakai1974.txt            # Published lots: Nakai et al. (1974)
PAPERE_FIG_STYLE=dt $PY figures.py                            # Figures 1-3 of the current version ($PAPERE_FIG_DIR/dt)
$PY figures.py                                                # Figures 1-4 of versions 1.0.1-1.0.3 (default style)
$PY graphical_abstract.py                                     # graphical abstract (earlier version)
$PY graphical_abstract_ddip.py                                # graphical abstract at 525 px (earlier version)
if [ "${ONLY_FAST:-0}" = "1" ]; then echo "done (fast scripts only)"; exit 0; fi
$PY regime_diagnostic.py    > output/regime_diagnostic.txt    # Tables 1 and 2 (types, true vs normal, diagnosis at one pull) (hours)
$PY pooled_pulls.py         > output/pooled_pulls.txt         # Table 3 (forecasts across pulls) (hours)
$PY form_test.py            > output/form_test.txt            # Forecasts: test of the form of the path [S1] (hours; writes output/form_test.pkl)
$PY form_test_control.py    > output/form_test_control.txt    # Forecasts: two further forms of acceleration [S1] (hours)
$PY form_test_restart.py    > output/form_test_restart.txt    # Forecasts: the same fits from more starting values [S1] (hours; reads output/form_test.pkl)
$PY plateau_single.py 95    > output/plateau_single_95.txt    # plateau 95 %, one pull [S2] (not reported)
$PY plateau_single.py 90    > output/plateau_single_90.txt    # plateau 90 %, one pull [S2] (not reported)
$PY plateau_pooled.py       > output/plateau_pooled.txt       # Forecasts: plateau below 100 %, pooled pulls [S2] (hours)
$PY gradual_start.py        > output/gradual_start.txt        # limitations: gradual start of release [S5] (hours)
$PY crack_not_widening.py   > output/crack_not_widening.txt   # limitations: units whose crack does not widen [S6] (hours)
$PY trend_detection.py > output/trend_detection.txt   # detecting the start of ageing at 0, 3 and 6 months [S7] (not reported) (hours)
echo "done: output/*.txt and the figures"
