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

.PHONY: run
run:
	python main.py

.PHONY: test
test:
	uv run pytest -x --tb=no -rs

.PHONY: test-cov
test-cov:
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

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
