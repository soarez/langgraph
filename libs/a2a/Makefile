.PHONY: all format lint type test test_watch tck help

all: help

######################
# TESTING
######################

TEST ?= tests

test:
	uv run pytest $(TEST)

test_watch:
	uv run ptw $(TEST)

# Conformance. Clones the pinned A2A TCK into .tck/ on first run, or set
# A2A_TCK_PATH to a checkout you already have.
tck:
	uv run python -m tests.tck.run

######################
# LINTING AND FORMATTING
######################

PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langgraph
lint_tests: PYTHON_FILES=tests

lint lint_diff lint_package lint_tests:
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run ty check langgraph

type:
	uv run ty check langgraph

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --fix $(PYTHON_FILES)

spell_check:
	uv run codespell --toml pyproject.toml

spell_fix:
	uv run codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '===================='
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'type                         - run type checking'
	@echo '-- TESTS --'
	@echo 'test                         - run the acceptance suite'
	@echo 'test TEST=<path>             - run one file'
	@echo 'tck                          - run the official A2A TCK against the SUT'
