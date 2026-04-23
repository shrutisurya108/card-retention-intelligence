"""
run_pipeline.py
---------------
Master pipeline orchestrator for ChurnShield.

Runs all phases in order. Each phase is a self-contained module
that can also be run independently. This script is the single
entry point to reproduce the entire pipeline end-to-end.

Usage:
    python run_pipeline.py              # Run all phases
    python run_pipeline.py --phase 1    # Run only Phase 1

Phases:
    1 — Data Ingestion      (src/ingestion.py)
    2 — EDA & Features      (src/eda.py, src/features.py)      [coming soon]
    3 — Model Training      (src/train.py, src/evaluate.py)    [coming soon]
    4 — SHAP Explainability (src/explain.py)                   [coming soon]
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.logger import get_logger

log = get_logger("pipeline")


def run_phase_1():
    """Phase 1: Data ingestion — CSV → SQLite + processed CSV."""
    from src.ingestion import run_ingestion
    run_ingestion()


# ── Registry: add new phases here as the project grows ───────────────────────
PHASES = {
    1: ("Data Ingestion",       run_phase_1),
    # 2: ("EDA & Features",     run_phase_2),   ← Phase 2 added here
    # 3: ("Model Training",     run_phase_3),   ← Phase 3 added here
    # 4: ("SHAP Explainability",run_phase_4),   ← Phase 4 added here
}


def main(target_phase: int = None):
    start_time = time.time()

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║        ChurnShield — ML Pipeline Runner          ║")
    log.info("║        card-retention-intelligence               ║")
    log.info("╚══════════════════════════════════════════════════╝")

    phases_to_run = (
        {target_phase: PHASES[target_phase]}
        if target_phase and target_phase in PHASES
        else PHASES
    )

    if target_phase and target_phase not in PHASES:
        log.error(f"Phase {target_phase} not found. Available: {list(PHASES.keys())}")
        sys.exit(1)

    for phase_num, (phase_name, phase_fn) in phases_to_run.items():
        log.info(f"▶  Running Phase {phase_num}: {phase_name}")
        phase_start = time.time()
        try:
            phase_fn()
            elapsed = time.time() - phase_start
            log.info(f"✓  Phase {phase_num} completed in {elapsed:.2f}s")
        except Exception as e:
            log.error(f"✗  Phase {phase_num} failed: {e}")
            raise

    total = time.time() - start_time
    log.info(f"Pipeline finished in {total:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChurnShield ML Pipeline")
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Run a specific phase only (e.g. --phase 1)"
    )
    args = parser.parse_args()
    main(target_phase=args.phase)
