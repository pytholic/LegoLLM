SHELL := bash

.PHONY: setup
setup:
	@echo "Setting up Python environment..."
	uv venv
	@echo "Installing dependencies..."
	uv sync --group dev --group docs
	@echo "Setup complete! Activate the virtual environment with: source .venv/bin/activate"

.PHONY: install
install:
	uv sync

.PHONY: install-dev
install-dev:
	uv sync --group dev

.PHONY: install-dev-docs
install-dev-docs:
	uv sync --group dev --group docs

.PHONY: install-extras
install-extras:
	uv sync --all-extras

.PHONY: run
run:
	python main.py

.PHONY: test
test:
	uv run pytest -x --tb=no -rs

.PHONY: test-cov
test-cov:
	uv run pytest --cov=legollm --cov=scripts --cov-report=term-missing

.PHONY: test-unit
test-unit:
	uv run pytest -x --tb=no -rs tests/unit

.PHONY: test-integration
test-integration:
	uv run pytest -x --tb=no -rs tests/integration

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format .

.PHONY: docs-build
docs-build:
	uv run mkdocs build

.PHONY: docs-serve
docs-serve:
	uv run mkdocs serve

.PHONY: clean
clean:
	@echo "Cleaning cache files..."
	@find . -type f -name ".DS_Store" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@echo "Cleaning complete!"

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  setup       Setup virtual environment and install dependencies"
	@echo "  install     Install dependencies"
	@echo "  run         Run the main application"
	@echo "  test        Run tests"
	@echo "  test-cov    Run tests with coverage"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  type-check  Run type checking"
	@echo "  docs-build  Build documentation"
	@echo "  docs-serve  Serve documentation locally"
	@echo "  clean       Clean cache files"

.PHONY: delete-venv
delete-venv:
	uv venv --clear

# ── Ollama serving ────────────────────────────────────────────────────────────
OLLAMA_BIN       ?= /usr/local/bin/ollama
NUM_INSTANCES    ?= 1
WORKERS_PER_GPU  ?= 1
BASE_PORT        ?= 11434
MODEL            ?= gemma3:27b-it-qat
# GPU assignment: comma-separated GPU IDs, one per instance.
# Examples:
#   GPU_IDS=0,1,2,3  — 4 instances, each on a different GPU (default: 0..NUM_INSTANCES-1)
#   GPU_IDS=0,0,0,0  — 4 instances, all on GPU 0
GPU_IDS          ?= 0

.PHONY: ollama-serve
ollama-serve:
	@mkdir -p logs
	@echo "Starting $(NUM_INSTANCES) Ollama instance(s) from port $(BASE_PORT) ($(WORKERS_PER_GPU) workers each)..."
	@gpu_ids="$(GPU_IDS)"; \
	for i in $$(seq 0 $$(($(NUM_INSTANCES) - 1))); do \
		port=$$(($(BASE_PORT) + i)); \
		if [ -n "$$gpu_ids" ]; then \
			gpu=$$(echo "$$gpu_ids" | cut -d',' -f$$((i + 1))); \
		else \
			gpu=$$i; \
		fi; \
		echo "  Instance $$i → GPU $$gpu, port $$port"; \
		CUDA_VISIBLE_DEVICES=$$gpu OLLAMA_HOST=127.0.0.1:$$port OLLAMA_NUM_PARALLEL=$(WORKERS_PER_GPU) $(OLLAMA_BIN) serve >> logs/ollama_$$port.log 2>&1 & \
	done
	@echo "Done. Logs in logs/ollama_<port>.log"

.PHONY: ollama-pull
ollama-pull:
	@echo "Pulling $(MODEL) on all $(NUM_INSTANCES) instances..."
	@for i in $$(seq 0 $$(($(NUM_INSTANCES) - 1))); do \
		port=$$(($(BASE_PORT) + i)); \
		OLLAMA_HOST=127.0.0.1:$$port $(OLLAMA_BIN) pull $(MODEL) & \
	done
	@wait
	@echo "Done."

.PHONY: ollama-stop
ollama-stop:
	@echo "Stopping all Ollama instances..."
	@pkill -f "$(OLLAMA_BIN) serve" || true
	@echo "Done."
