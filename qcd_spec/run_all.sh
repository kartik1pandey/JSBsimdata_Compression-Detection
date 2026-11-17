#!/usr/bin/env bash
# Spec run #1 grid, serial (one GPU, ~1.3 GB free RAM). Each run skips itself if its results json exists.
#   raw:  q in {tf, gru, tcn, lin}                       (detectability ceiling, K_X)
#   cae:  enc = q in {tf, gru, tcn}, k = 8, lambda in {0, 0.25, 0.5, 0.75, 1}   (lambda = 0: unaware AE)
# Seed 0 for every configuration first, then seeds 1 and 2.
cd "$(dirname "$0")"
P=../../.venv/Scripts/python
export PYTHONIOENCODING=utf-8
mkdir -p results/logs
for s in 0 1 2; do
  for q in tf gru tcn lin; do
    $P -u spec_run.py --pipeline raw --q $q --seed $s --e1 30 > results/logs/raw_q${q}_s${s}.log 2>&1
    echo "raw $q s$s exit $? $(date +%T)"
  done
  for f in tf gru tcn; do
    for lam in 0 0.25 0.5 0.75 1; do
      $P -u spec_run.py --pipeline cae --enc $f --q $f --k 8 --lam $lam --seed $s \
        --e0 10 --e1 10 --e2 15 --e3 10 > results/logs/cae_${f}_lam${lam}_s${s}.log 2>&1
      echo "cae $f lam $lam s$s exit $? $(date +%T)"
    done
  done
done
echo done
