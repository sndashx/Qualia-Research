.PHONY: help setup test train-toy eval-toy lint clean

PYTHON ?= python3
VENV ?= .venv

help:
	@echo "Targets:"
	@echo "  setup       Create venv, install package + dev deps, install pre-commit hooks"
	@echo "  test        Run pytest"
	@echo "  train-toy   Run toy training (placeholder until training is wired)"
	@echo "  eval-toy    Run toy eval (placeholder until eval is wired)"
	@echo "  lint        ruff + black --check"
	@echo "  clean       Remove caches"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	$(VENV)/bin/pre-commit install

test:
	$(PYTHON) -m pytest

train-toy:
	$(PYTHON) -m qualia.train.train --config-name=baseline

eval-toy:
	$(PYTHON) -m qualia.eval.run_eval --config-name=baseline

lint:
	ruff check src tests
	black --check src tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -exec rm -rf {} +