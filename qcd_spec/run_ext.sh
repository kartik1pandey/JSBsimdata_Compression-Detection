#!/usr/bin/env bash
# Extra experiments (seed 0, full schedule), second lane next to run_all.sh. Results -> results/ext/.
#   (order: B first � the main question is the encoder output size k)
#   A  bit budget: quantise the trained k = 8 codes to b bits per number (quant_eval.py), lam 0 and 0.25
#   B  latent size: k in {2, 4, 16, 32, 64, 128} (k = 8 is in the main grid), lam 0 and 0.25
#   C  model pairing: encoder family != density family (q0, q1), lam 0 and 0.25
#   D  decoder: per-step MLP and bidirectional GRU instead of the conv decoder, lam 0 and 0.25
# Every step skips itself if its json exists, so the script can be re-run after an interruption.
cd "$(dirname "$0")"
P=../../.venv/Scripts/python
export PYTHONIOENCODING=utf-8 QCD_RES=results/ext
mkdir -p results/ext/logs
E="--e0 10 --e1 10 --e2 15 --e3 10"
note() { echo "$1 exit $2 $(date +%T)"; }
# ---- B: latent size
for f in gru tf; do
  for k in 2 4 16 32 64 128; do
    for lam in 0 0.25; do
      $P -u -W ignore spec_run.py --pipeline cae --enc $f --q $f --k $k --lam $lam --seed 0 $E \
        > results/ext/logs/k${k}_${f}_lam${lam}.log 2>&1
      note "k $k $f lam $lam" $?
    done
  done
done
# ---- A: bit budget on the main-grid seed-0 checkpoints
for f in tf gru; do
  for lam in 0 0.25; do
    t=cae_e${f}_q${f}_k8_lam${lam}_s0
    $P -u -W ignore quant_eval.py --tag $t --bits 1 2 3 4 6 8 >> results/ext/logs/quant_$t.log 2>&1
    note "quant $t" $?
  done
done
# ---- C: encoder x density pairings (diagonal is in the main grid)
for f in gru tf; do
  for q in tf gru tcn lin; do
    [ "$f" = "$q" ] && continue
    for lam in 0 0.25; do
      $P -u -W ignore spec_run.py --pipeline cae --enc $f --q $q --k 8 --lam $lam --seed 0 $E \
        > results/ext/logs/pair_e${f}_q${q}_lam${lam}.log 2>&1
      note "pair enc $f q $q lam $lam" $?
    done
  done
done
# ---- D: decoder family
for dec in mlp bigru; do
  for f in gru tf; do
    for lam in 0 0.25; do
      $P -u -W ignore spec_run.py --pipeline cae --enc $f --q $f --k 8 --lam $lam --seed 0 --dec $dec $E \
        > results/ext/logs/dec_${dec}_${f}_lam${lam}.log 2>&1
      note "dec $dec $f lam $lam" $?
    done
  done
done
echo done
