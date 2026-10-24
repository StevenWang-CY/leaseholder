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

.PHONY: check lint lint-rust lint-python fmt test test-rust test-vllm test-python schema reproduce-e1 reproduce-e2 \
  e2-figures

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

# E1's tables in crates/lh-sim/README.md from the 32x corpus, which stays outside the repository (`lh-bench
# scale-traces` writes it; python/leaseholder_bench/README.md): the regime map, the realistic regime over 20 seeds,
# and the other GPUs at the 32x replay and at their own scalings. Usage: make reproduce-e1 CORPUS=<its directory>.
E1_REPORT ?= build/reproduce-e1
E1_SIM = target/release/lh-sim
reproduce-e1:
	@test -n "$(CORPUS)" || { echo "usage: make reproduce-e1 CORPUS=<the 32x corpus directory>" >&2; exit 2; }
	cargo build --release --locked -p lh-sim
	@mkdir -p $(E1_REPORT)
	$(E1_SIM) regime $(CORPUS) --csv $(E1_REPORT)/regime.csv --markdown $(E1_REPORT)/regime.md >/dev/null
	$(E1_SIM) regime $(CORPUS) --sessions 8,12,16,24 --cpu-ratio 4 --pause-sources trace \
	  --seeds $$(seq 11 30 | paste -sd, -) --csv $(E1_REPORT)/realistic.csv --markdown $(E1_REPORT)/realistic.md \
	  >/dev/null
	$(E1_SIM) hardware $(CORPUS) --csv $(E1_REPORT)/hardware.csv --markdown $(E1_REPORT)/hardware.md >/dev/null
	$(E1_SIM) hardware $(CORPUS) --fit --csv $(E1_REPORT)/hardware-fit.csv --markdown $(E1_REPORT)/hardware-fit.md \
	  >/dev/null
	@ls $(E1_REPORT)/*.md

# E2's report (frontier.md, frontier.csv, frontier.svg, waterfall.svg) from the committed records, with the bench
# venv; it prints the files it writes.
reproduce-e2:
	@cd python/leaseholder_bench && uv run -q lh-bench e2 report \
	  $(abspath $(dir $(wildcard $(E2_RESULTS)/*/*/run.json))) --out $(abspath $(E2_REPORT))

# E2's figures in the docs, docs/figures/e2-frontier.svg and e2-waterfall.svg, from the records of the configurations an
# operator could deploy (the ablations left out), each with all its runs. The demo GIF beside them is demo.py's.
E2_DEPLOYABLE := a1-voice disc-frozen-s24 disc-a4-b1024-t0-s24 disc-a6-reserve2048-s24 disc-a7b-sglang-hicache-s24 \
  disc-a8c-resubmit-s24 disc-a9-s24 disc-pr-ab48-s24
E2_STAGES := a1-voice disc-frozen-s24 disc-a6-reserve2048-s24 disc-a8c-resubmit-s24 disc-a9-s24 disc-pr-ab48-s24
e2-figures:
	@cd python/leaseholder_bench && uv run -q lh-bench e2 report \
	  $(abspath $(dir $(foreach p,$(E2_DEPLOYABLE),$(wildcard $(E2_RESULTS)/*/$(p)/run.json)))) \
	  $(foreach p,$(E2_STAGES),--waterfall $(p)) --out $(abspath build/e2-figures) >/dev/null
	@mkdir -p docs/figures
	cp build/e2-figures/frontier-weka-x32-long.svg docs/figures/e2-frontier.svg
	cp build/e2-figures/waterfall.svg docs/figures/e2-waterfall.svg
