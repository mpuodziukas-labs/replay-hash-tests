"""Tests for the deterministic replay corpus.

Verifies the three claims literally:
  1. n == 47,312
  2. determinism: same seed -> identical per-case hashes (and corpus root)
  3. divergence == 0 across two independent runs

Run:  pytest -q
"""
import sys
from pathlib import Path

# Make replay.py importable when pytest is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import replay  # noqa: E402

# The corpus root is pinned. If the pipeline or corpus ever changes, this fails
# loudly -- which is the point: the published hash must stay reproducible.
EXPECTED_CORPUS_HASH = (
    "3e806db2c4978dfae932b98d3d2dfbcbc76cd4f48f5b29b6181be9eb531167a8"
)


def test_corpus_size_is_47312():
    hashes = replay.run_once()
    assert len(hashes) == 47_312
    assert replay.N_CASES == 47_312


def test_same_seed_same_hashes():
    run_a = replay.run_once()
    run_b = replay.run_once()
    assert run_a == run_b
    assert replay.corpus_hash(run_a) == replay.corpus_hash(run_b)


def test_zero_divergence():
    run_a = replay.run_once()
    run_b = replay.run_once()
    assert replay.count_divergences(run_a, run_b) == 0


def test_corpus_hash_is_reproducible():
    assert replay.corpus_hash(replay.run_once()) == EXPECTED_CORPUS_HASH


def test_different_seed_changes_corpus():
    # Sanity: the seed actually drives the corpus (not a constant).
    baseline = replay.corpus_hash(replay.run_once(seed=replay.SEED))
    altered = replay.corpus_hash(replay.run_once(n=replay.N_CASES, seed=replay.SEED + 1))
    assert baseline != altered


def test_transform_is_pure():
    case = replay.generate_case(42)
    assert replay.transform(case) == replay.transform(case)
    assert replay.hash_output(replay.transform(case)) == replay.hash_output(
        replay.transform(case)
    )


def test_generate_case_is_deterministic():
    assert replay.generate_case(1000) == replay.generate_case(1000)


def test_main_exits_zero():
    assert replay.main() == 0
