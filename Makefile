.PHONY: help setup test smoke train-toy eval-toy lint clean

VENV ?= .venv
PYTHON := $(VENV)/bin/python

help:
	@echo "Targets:"
	@echo "  setup       Create venv, install package + dev deps, install pre-commit hooks"
	@echo "  test        Run full pytest suite (includes smoke)"
	@echo "  smoke       Run smoke tests only (~60 s CPU, for fast polecat feedback)"
	@echo "  train-toy   Run toy training (placeholder until training is wired)"
	@echo "  eval-toy    Run toy eval (placeholder until eval is wired)"
	@echo "  lint        ruff + black --check"
	@echo "  clean       Remove caches"

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	$(VENV)/bin/pre-commit install

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m pytest -m smoke

train-toy:
	$(PYTHON) -m qualia.train.train --config-name=baseline

eval-toy:
	$(PYTHON) -m qualia.eval.run_eval --config-name=baseline

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/black --check src tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -exec rm -rf {} +