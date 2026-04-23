# Makefile
# --------
# Convenience commands for the card-retention-intelligence project.
# Always routes through `python` (pyenv-managed 3.11.9) to avoid
# picking up system-level binaries like pytest from Python 3.14.
#
# Usage:
#   make test        → run all tests
#   make ingest      → run Phase 1 ingestion
#   make pipeline    → run full pipeline
#   make dashboard   → launch Streamlit app
#   make lint        → check code style
#   make clean       → remove generated artifacts

.PHONY: test ingest pipeline dashboard lint clean

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

test-phase1:
	python -m pytest tests/test_ingestion.py -v

# ── Pipeline ──────────────────────────────────────────────────────────────────
ingest:
	python run_pipeline.py --phase 1

pipeline:
	python run_pipeline.py

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard:
	python -m streamlit run dashboard/app.py

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	python -m flake8 src/ --max-line-length=100 --ignore=E203,W503

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
