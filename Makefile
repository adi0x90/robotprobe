# RobotProbe developer tasks. Uses uv (https://docs.astral.sh/uv/).
#
# `make setup` installs the core library plus dev tooling only. Simulators and the
# white-box bridge are opt-in so that first contact stays small.
.PHONY: setup setup-simulators setup-nnsight setup-all test lint fmt fmt-check ci demo demo-clean clean

setup:  ## sync .venv from pyproject + uv.lock (core + dev tooling)
	uv sync

setup-simulators:  ## also install simulator extras (gymnasium / mujoco / pybullet)
	uv sync --extra simulators

setup-nnsight:  ## also install the optional white-box bridge (nnsight + torch)
	uv sync --extra nnsight

setup-all:  ## everything: simulators + white-box bridge
	uv sync --extra simulators --extra nnsight

test:  ## run the full test suite
	uv run pytest

lint:  ## static checks
	uv run ruff check robotprobe robotsec_range tests scripts

fmt:  ## autoformat
	uv run ruff format robotprobe robotsec_range tests scripts

fmt-check:  ## report formatting drift (not enforced by `make ci`; run `make fmt` first)
	uv run ruff format --check robotprobe robotsec_range tests scripts

ci:  ## local checks: lint, tests, doctor, demo smoke
	@echo "== lint =="        && uv run ruff check robotprobe robotsec_range tests scripts
	@echo "== tests =="       && uv run pytest
	@echo "== doctor =="      && uv run robotprobe doctor
	@echo "== demo smoke =="  && rm -rf .ci-demo && uv run robotprobe demo --out .ci-demo >/dev/null
	@echo "== artifacts =="   && test -f .ci-demo/report.md && test -f .ci-demo/summary.json
	@rm -rf .ci-demo
	@echo "local ci: OK"

demo:  ## run the quickstart incident bundle into examples/quickstart/out
	uv run python -m robotprobe demo

demo-clean:  ## run the quickstart into a throwaway directory and print the report
	uv run python -m robotprobe demo --out /tmp/robotprobe-demo && cat /tmp/robotprobe-demo/report.md

clean:  ## remove the venv, caches, and demo output
	rm -rf .venv .pytest_cache .ruff_cache .ci-demo examples/quickstart/out
