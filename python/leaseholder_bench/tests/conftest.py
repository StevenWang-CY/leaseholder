from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the caller's LH_BENCH_* settings (the plugin's and the YAML template's) out of every test."""
    for name in [name for name in os.environ if name.startswith("LH_BENCH_")]:
        monkeypatch.delenv(name)


@pytest.fixture
def fake_prompt_generator() -> MagicMock:
    """A tokenizer-free stand-in for AIPerf's PromptGenerator.

    Hash id h decodes to the corpus tokens at offset 64 * h, and a token t decodes to the word
    " w<t>", so whitespace splitting recovers the exact token sequence of a synthesized prompt.
    """
    pg = MagicMock()
    pg._tokenized_corpus = list(range(10_000, 14_096))
    pg._corpus_size = len(pg._tokenized_corpus)
    pg._cache = {}
    pg._bpe_stable_terminator_tokens = []
    state = {"hash_id": 0}
    pg._hash_id_corpus_rng.reseed_for_hash_id.side_effect = lambda h: state.update(hash_id=h)
    pg._hash_id_corpus_rng.randrange.side_effect = lambda n: (state["hash_id"] * 64) % n
    pg.tokenizer.decode.side_effect = lambda tokens: "".join(f" w{t}" for t in tokens)
    return pg
