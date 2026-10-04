#!/usr/bin/env bash
# Reproduce every table and figure of the paper and the Supplementary Information.
# Fast scripts take seconds to minutes; the simulations marked (hours) take several hours each on a laptop with 6 cores
# (they use multiprocessing). Set ONLY_FAST=1 to skip them; the saved outputs in output/ are the runs used for the paper.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
# Figures go to ./figures in the standalone copy and to ../figures (the paper directory) in the paper repository.
if [ -z "${PAPERE_FIG_DIR:-}" ]; then
  if [ -d figures ]; then export PAPERE_FIG_DIR="$PWD/figures"; else export PAPERE_FIG_DIR="$PWD/../figures"; fi
fi
mkdir -p output
$PY likelihood.py           > output/likelihood.txt           # SI S3: convergence of the local-grid likelihood
$PY single_peak_controls.py > output/single_peak_controls.txt # Table 3 (other single-peak distributions)
$PY skew_baseline.py        > output/skew_baseline.txt        # Table 3 (skew-normal baseline)
$PY equal_quality.py        > output/equal_quality.txt        # Table 4, Figure 3 (lots of equal quality)
$PY sign_rule.py            > output/sign_rule.txt            # direction of the error (sign rule, four-class expression), SI S4
$PY nagase1976.py           > output/nagase1976.txt           # Published lots: Nagase et al. (1976), Figure 4
$PY nakai1974.py            > output/nakai1974.txt            # Published lots: Nakai et al. (1974)
$PY figures.py                                                # Figures 1-4 (written to $PAPERE_FIG_DIR)
$PY graphical_abstract.py                                     # graphical abstract
if [ "${ONLY_FAST:-0}" = "1" ]; then echo "done (fast scripts only)"; exit 0; fi
$PY regime_diagnostic.py    > output/regime_diagnostic.txt    # Tables 1, 2 and 5 (types, true vs normal, diagnosis at one pull) (hours)
$PY pooled_pulls.py         > output/pooled_pulls.txt         # Table 6 (forecasts across pulls) (hours)
$PY form_test.py            > output/form_test.txt            # SI S1: test of the form of the path (hours; writes output/form_test.pkl)
$PY form_test_control.py    > output/form_test_control.txt    # SI S1: two further forms of acceleration (hours)
$PY form_test_restart.py    > output/form_test_restart.txt    # SI S1: the same fits from more starting values (hours; reads output/form_test.pkl)
$PY plateau_single.py 95    > output/plateau_single_95.txt    # SI S2: plateau 95 %, one pull
$PY plateau_single.py 90    > output/plateau_single_90.txt    # SI S2: plateau 90 %, one pull
$PY plateau_pooled.py       > output/plateau_pooled.txt       # SI S2: plateau below 100 %, pooled pulls (hours)
$PY gradual_start.py        > output/gradual_start.txt        # SI S5: gradual start of release (hours)
$PY crack_not_widening.py   > output/crack_not_widening.txt   # SI S6: units whose crack does not widen (hours)
$PY trend_detection.py > output/trend_detection.txt   # SI S7: detecting the start of ageing at 0, 3 and 6 months (hours)
echo "done: output/*.txt and the figures"
