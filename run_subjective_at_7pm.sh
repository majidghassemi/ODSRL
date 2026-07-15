#!/usr/bin/env bash
# Scheduled launcher for the HEAVY subjective-N Q5 run.
# Sleeps until 7:00 PM America/Toronto on 2026-07-14, then runs
# exp_rlvr_subjective.py on the GPU:
#   Phase 1: Qwen2.5-32B-Instruct (4-bit) gold-vets TruthfulQA N items (then freed).
#   Phase 2: Qwen2.5-7B panel grades V (GSM8K) + N; ESA transfer + measured drift.
# Checkpointed: N gold labels -> rlvr_subjective_N.json, panel scores ->
# rlvr_real_cache.json, so any interruption resumes. Retries on non-zero exit.
#
# Launch detached so it survives this session:
#   nohup setsid bash run_subjective_at_7pm.sh > subjective_run.log 2>&1 &
set -u
cd "$(dirname "$0")"
LOG() { echo "[$(TZ=America/Toronto date '+%F %T %Z')] $*"; }

TARGET=$(TZ=America/Toronto date -d '2026-07-14 19:00:00' +%s)
NOW=$(date +%s); WAIT=$(( TARGET - NOW ))
if [ "$WAIT" -gt 0 ]; then
  LOG "waiting ${WAIT}s until 7:00 PM Toronto to start the subjective-N run"
  sleep "$WAIT"
else
  LOG "7 PM Toronto already passed (${WAIT}s); starting immediately"
fi

# guard: confirm CUDA is intact before a long run (an install can break it)
python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { LOG "CUDA unavailable — aborting (fix torch: pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121)"; exit 1; }

LOG "starting exp_rlvr_subjective.py"
ATTEMPT=0; MAX=6
until python3 exp_rlvr_subjective.py --n-v 120 --n-n 100 --batch-size 16; do
  ATTEMPT=$(( ATTEMPT + 1 ))
  LOG "exited non-zero; attempt ${ATTEMPT}/${MAX}"
  [ "$ATTEMPT" -ge "$MAX" ] && { LOG "giving up after ${MAX} attempts"; exit 1; }
  LOG "retry in 300s (caches preserve progress)"; sleep 300
done
LOG "subjective-N run complete; figure q5_rlvr_subjective.png + results written"
