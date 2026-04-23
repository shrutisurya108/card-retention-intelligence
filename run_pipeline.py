"""
run_pipeline.py
---------------
Master pipeline orchestrator for ChurnShield.

Each phase adds one step here. Running this file executes the
complete pipeline end-to-end in the correct order.

Usage:
    python run_pipeline.py              # run all phases
    python run_pipeline.py --phase 1   # run a specific phase only
    python run_pipeline.py --phase 2   # run Phase 2 only
"""

import sys
import argparse
import traceback
from src.logger import get_logger

log = get_logger("pipeline")


def run_phase_1():
    from src.ingestion import run_ingestion
    run_ingestion()


def run_phase_2():
    from src.eda import run_eda
    from src.features import run_feature_engineering
    run_eda()
    run_feature_engineering()


# ── Phase registry ────────────────────────────────────────────────────────────
PHASES = {
    1: ("Data Ingestion",            run_phase_1),
    2: ("EDA & Feature Engineering", run_phase_2),
    # 3: ("Model Training",           run_phase_3),   # added in Phase 3
    # 4: ("SHAP Explainability",      run_phase_4),   # added in Phase 4
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="ChurnShield ML Pipeline — card-retention-intelligence"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=list(PHASES.keys()),
        default=None,
        help="Run a specific phase only. Omit to run all phases.",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    phases = {args.phase: PHASES[args.phase]} if args.phase else PHASES

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║       ChurnShield — ML Pipeline Starting             ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    for phase_num, (phase_name, phase_fn) in phases.items():
        log.info(f"▶  Running Phase {phase_num}: {phase_name}")
        try:
            phase_fn()
            log.info(f"✓  Phase {phase_num} completed successfully\n")
        except Exception as e:
            log.error(f"✗  Phase {phase_num} failed: {e}")
            log.debug(traceback.format_exc())
            sys.exit(1)

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║       Pipeline finished — all phases complete ✓      ║")
    log.info("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()