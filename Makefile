# Development tasks. Each Python package is tested in its own uv environment, because the engine plugins, the
# benchmark tools and the voice worker have disjoint dependencies (vLLM, AIPerf, LiveKit).

RUFF := uvx ruff@0.17.1
VLLM_REF := v0.32.0
VLLM_SRC ?= $(HOME)/.cache/leaseholder/vllm-$(VLLM_REF)
# Every Python package except the engine plugins (tested by test-vllm); each one has a `test` extra.
PY_PACKAGES := $(filter-out python/leaseholder_vllm,$(patsubst %/pyproject.toml,%,$(wildcard python/*/pyproject.toml)))
# E2's committed records (lh-bench e2 export), one directory per point under its campaign, and where
# reproduce-e2 writes their report.
E2_RESULTS := benchmarks/results/2026-10-23-e2
E2_REPORT ?= build/reproduce-e2

.PHONY: check lint lint-rust lint-python fmt test test-rust test-vllm test-python schema reproduce-e2

check: lint test

lint: lint-rust lint-python

lint-rust:
	cargo fmt --all -- --check
	cargo clippy --all-targets --locked -- -D warnings

lint-python:
	$(RUFF) check python
	$(RUFF) format --check python

fmt:
	cargo fmt --all
	$(RUFF) check --fix python
	$(RUFF) format python

test: test-rust test-vllm test-python

test-rust:
	cargo test --locked

# The offload manager is plain Python, so the plugin tests load it from a vLLM source checkout: no GPU or torch.
$(VLLM_SRC):
	git -c advice.detachedHead=false clone -q --depth 1 --branch $(VLLM_REF) --filter=blob:none --sparse https://github.com/vllm-project/vllm $@
	git -C $@ sparse-checkout set vllm

test-vllm: $(VLLM_SRC)
	cd python/leaseholder_vllm && VLLM_SRC=$(VLLM_SRC) uv run -q --extra test pytest -q

# Integration tests need hosts (a LiveKit server, a running engine); CI runs everything else.
test-python:
	@for pkg in $(PY_PACKAGES); do \
	  echo "== $$pkg"; (cd $$pkg && uv run -q --extra test pytest -q -m "not integration") || exit 1; \
	done

schema:
	cargo run -q -p lh-proto --bin lh-proto-schema

# E2's report (frontier.md, frontier.csv, frontier.svg, waterfall.svg) from the committed records, with the bench
# venv; it prints the files it writes.
reproduce-e2:
	@cd python/leaseholder_bench && uv run -q lh-bench e2 report \
	  $(abspath $(dir $(wildcard $(E2_RESULTS)/*/*/run.json))) --out $(abspath $(E2_REPORT))
