#!/usr/bin/env bash
# Time-boxed pass (40 min): seed 0, lambda in {0, 1, 0.5}, shortened schedule, three lanes in parallel.
cd "$(dirname "$0")"
P=../../.venv/Scripts/python
export PYTHONIOENCODING=utf-8 QCD_RES=results/quick
E="--e0 5 --e1 5 --e2 8 --e3 6"
lane() { f=$1; shift
  for lam in 0 1 0.5; do
    $P -u spec_run.py --pipeline cae --enc $f --q $f --k 8 --lam $lam --seed 0 $E > results/quick/logs/cae_${f}_lam${lam}.log 2>&1
    echo "cae $f lam $lam exit $? $(date +%T)"
  done; }
lane tf &
( $P -u spec_run.py --pipeline raw --q tcn --seed 0 --e1 30 > results/quick/logs/raw_tcn.log 2>&1; echo "raw tcn exit $? $(date +%T)"; lane gru ) &
lane tcn &
wait
echo done $(date +%T)
