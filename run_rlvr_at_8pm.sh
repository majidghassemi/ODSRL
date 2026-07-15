#!/usr/bin/env bash
# One-shot launcher for the real-verifier RLVR Q5 run.
# Sleeps until 8:00 PM America/Toronto on 2026-07-13, then runs
# exp_rlvr_real.py on the GPU. The experiment checkpoints all LLM scores to
# rlvr_real_cache.json, so if it is interrupted it resumes from where it stopped.
# This wrapper additionally retries on non-zero exit (transient download/OOM),
# each retry resuming from the cache.
#
# Launch it detached so it survives this shell / session:
#   nohup setsid bash run_rlvr_at_8pm.sh > rlvr_run.log 2>&1 &
set -u

cd "$(dirname "$0")"
LOG() { echo "[$(TZ=America/Toronto date '+%Y-%m-%d %H:%M:%S %Z')] $*"; }

TARGET=$(TZ=America/Toronto date -d '2026-07-13 20:00:00' +%s)
NOW=$(date +%s)
WAIT=$(( TARGET - NOW ))
if [ "$WAIT" -gt 0 ]; then
  LOG "waiting ${WAIT}s until 8:00 PM Toronto to start the RLVR run"
  sleep "$WAIT"
else
  LOG "8 PM Toronto already passed (${WAIT}s); starting immediately"
fi

LOG "starting exp_rlvr_real.py"
ATTEMPT=0
MAX_ATTEMPTS=6
until python3 exp_rlvr_real.py --n-v 120 --n-n 120 --batch-size 16; do
  CODE=$?
  ATTEMPT=$(( ATTEMPT + 1 ))
  LOG "exp_rlvr_real.py exited ${CODE}; attempt ${ATTEMPT}/${MAX_ATTEMPTS}"
  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
    LOG "giving up after ${MAX_ATTEMPTS} attempts"
    exit 1
  fi
  LOG "retrying in 300s (cache preserves progress)"
  sleep 300
done
LOG "RLVR run complete; figure q5_rlvr_real.png and rlvr_real_results.json written"
