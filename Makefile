# Makefile
# --------
# Convenience commands for the card-retention-intelligence project.
# Always routes through `python` (pyenv-managed 3.11.9).
#
# Usage:
#   make test           → run all tests
#   make test-phase1    → Phase 1 tests only
#   make test-phase2    → Phase 2 tests only
#   make test-phase3    → Phase 3 tests only
#   make ingest         → run Phase 1
#   make eda            → run Phase 2
#   make train          → run Phase 3
#   make pipeline       → run all phases
#   make dashboard      → launch Streamlit app
#   make clean          → remove generated artifacts

.PHONY: test test-phase1 test-phase2 test-phase3 ingest eda train pipeline dashboard clean

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

test-phase1:
	python -m pytest tests/test_ingestion.py -v

test-phase2:
	python -m pytest tests/test_features.py -v

test-phase3:
	python -m pytest tests/test_train.py -v

# ── Pipeline ──────────────────────────────────────────────────────────────────
ingest:
	python run_pipeline.py --phase 1

eda:
	python run_pipeline.py --phase 2

train:
	python run_pipeline.py --phase 3

pipeline:
	python run_pipeline.py

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard:
	python -m streamlit run dashboard/app.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +