#!/usr/bin/env bash
# Server lane: run every job of a job generator (default server_jobs.py), N at a time on the one GPU.
# Safe to re-run: jobs skip when their result json exists. A job that dies with CUDA out-of-memory (the GPU is
# shared by several runs) is retried up to 6 times with a growing pause; the real exit code is logged.
#   tmux new -d -s qcd 'bash ~/qcd/code/run_server.sh 6 server_jobs.py'
N=${1:-6}
GEN=${2:-server_jobs.py}
cd "$(dirname "$0")"
export QCD_DATA=$HOME/qcd/data/qcd_pure PYTHONIOENCODING=utf-8 MPLBACKEND=Agg OMP_NUM_THREADS=4 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/qcd/venv/bin/python
JOBS=jobs_${GEN%.py}.txt
$PY "$GEN" > "$JOBS"
one() {
  res=$1 ckpt=$2 log=$3 script=$4; shift 4
  mkdir -p "$res/logs"
  for try in 1 2 3 4 5 6; do
    QCD_RES=$res QCD_CKPT=$ckpt $PY -u -W ignore "$script" "$@" > "$res/logs/$log.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q OutOfMemoryError "$res/logs/$log.log"; then
      echo "$(date +%F_%T) oom-retry $try $res $log" >> progress.log
      sleep $((90 * try))
      continue
    fi
    break
  done
  echo "$(date +%F_%T) exit $rc $res $log" >> progress.log
}
export -f one; export PY
echo "$(date +%F_%T) start $(wc -l < "$JOBS") jobs from $GEN, $N parallel" >> progress.log
xargs -P "$N" -L 1 -a "$JOBS" bash -c 'one "$@"' _
echo "$(date +%F_%T) ALL DONE $GEN" >> progress.log
