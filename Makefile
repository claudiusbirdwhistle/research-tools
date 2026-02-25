# Research Tools — common development targets
#
# Requires the shared venv to be active or use PYTHON to override.
# Example: make test PYTHON=research-engine/.venv/bin/python

PYTHON ?= research-engine/.venv/bin/python
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff

.PHONY: test lint check format install

## Run the pytest test suite
test:
	$(PYTEST) tests/ -v --tb=short

## Lint all library and tool code with ruff
lint:
	$(RUFF) check lib/ tests/

## Run both tests and lint
check: lint test

## Auto-format code with ruff
format:
	$(RUFF) format lib/ tests/

## Install the shared library in editable mode (development)
install:
	$(PYTHON) -m pip install -e ".[dev]"
