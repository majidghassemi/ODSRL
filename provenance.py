"""Shared provenance stamp for every emitted JSON (AAAI reproducibility)."""
import os
import subprocess
from datetime import datetime, timezone

HONEST_MODEL = "Qwen/Qwen2.5-32B-Instruct"
SYCO_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GOLD_MODEL = "Qwen/Qwen2.5-32B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def stamp(**extra):
    """Return a provenance dict; pass run-specific fields (n_v, n_n, seed, models...)."""
    p = {
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "honest_model": HONEST_MODEL, "syco_model": SYCO_MODEL, "gold_model": GOLD_MODEL,
        "temperature": 0.0, "decoding": "greedy",
    }
    p.update(extra)
    return p
