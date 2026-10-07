#!/usr/bin/env python3
"""Deterministic replay corpus harness.

Generates a seeded, reproducible synthetic corpus of N cases, runs each through
a deterministic transformation pipeline, hashes every output with SHA-256, and
asserts bit-for-bit identical results across two independent runs (divergence=0).

This is a SYNTHETIC, reproducible demonstration of deterministic-replay
methodology. It is NOT a production deployment and NOT client data. See README.

Run:  python3 replay.py
Out:  47,312 deterministic replays | 0 divergence | corpus_hash=<hash>
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass

# --- Configuration (the claim is pinned here) -------------------------------
N_CASES = 47_312
SEED = 0xC0FFEE  # fixed seed -> fully reproducible corpus
# The corpus root for (N_CASES, SEED) with the pipeline below. main() fails if
# a run does not reproduce it, so the published hash cannot drift silently.
EXPECTED_CORPUS_HASH = (
    "3e806db2c4978dfae932b98d3d2dfbcbc76cd4f48f5b29b6181be9eb531167a8"
)
_U64 = 1 << 64
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)

# A small fixed vocabulary so synthetic cases look like structured records
# without depending on any external data source.
_CATEGORIES = ("alpha", "bravo", "charlie", "delta", "echo")
_REGIONS = ("us-east", "us-west", "eu-central", "ap-south")


@dataclass(frozen=True)
class Case:
    """A single synthetic input record. Fully determined by its index + seed."""

    index: int
    category: str
    region: str
    amount: int
    flags: int


def _splitmix64(x: int) -> int:
    """Deterministic, dependency-free 64-bit hash/PRNG step (SplitMix64).

    Pure integer arithmetic -> identical on every platform and Python build.
    No use of the global `random` module, so no hidden global state.
    """
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    z = z ^ (z >> 31)
    return z & mask


def generate_case(index: int, seed: int = SEED) -> Case:
    """Generate the i-th synthetic case. Pure function of (index, seed).

    index must be an int >= 0 and seed an int in [0, 2**64): a wider seed would
    alias a narrower one, and a non-int index could expand without bound.
    """
    if not _is_int(index) or index < 0:
        raise ValueError(f"index must be an int >= 0, got {index!r}")
    if not _is_int(seed) or not 0 <= seed < _U64:
        raise ValueError(f"seed must be an int in [0, 2**64), got {seed!r}")
    r = _splitmix64(seed ^ (index * 0x100000001B3))
    return Case(
        index=index,
        category=_CATEGORIES[r % len(_CATEGORIES)],
        region=_REGIONS[(r >> 8) % len(_REGIONS)],
        amount=(r >> 16) % 1_000_000,
        flags=(r >> 40) & 0xFF,
    )


def generate_corpus(n: int = N_CASES, seed: int = SEED):
    """Yield n deterministic synthetic cases (n >= 1)."""
    if not _is_int(n) or n < 1:
        raise ValueError(f"n must be an int >= 1, got {n!r}")
    for i in range(n):
        yield generate_case(i, seed)


def transform(case: Case) -> dict:
    """Deterministic transformation pipeline applied to one case.

    Pure: output is a function of input only. No clocks, no randomness, no I/O.
    Mirrors a real normalize -> enrich -> score pipeline shape, but every step
    is a fixed integer/string operation so replays are bit-for-bit identical.
    """
    # Stage 1: normalize
    norm_amount = case.amount // 100  # truncating integer division (stable)
    # Stage 2: enrich (derive deterministic fields)
    risk_bits = bin(case.flags).count("1")
    bucket = (case.amount + case.index) % 7
    # Stage 3: score (fixed-weight integer model -> no float nondeterminism)
    score = (norm_amount * 3 + risk_bits * 11 + bucket * 5) % 100_000
    return {
        "index": case.index,
        "category": case.category,
        "region": case.region,
        "norm_amount": norm_amount,
        "risk_bits": risk_bits,
        "bucket": bucket,
        "score": score,
    }


def _canon(obj, out: list) -> None:
    """Type-tagged, length-prefixed canonical encoding (appends bytes to out).

    Different types never collide: int 1 vs str "1" vs bool, tuple vs list, and
    int key vs str key all encode differently. Non-finite floats and unknown
    types are refused, never coerced.
    """
    if obj is None:
        out.append(b"N;")
    elif isinstance(obj, bool):
        out.append(b"B1;" if obj else b"B0;")
    elif isinstance(obj, int):
        out.append(b"I%d;" % obj)
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float refused: {obj!r}")
        out.append(b"F" + repr(obj).encode("ascii") + b";")
    elif isinstance(obj, str):
        raw = obj.encode("utf-8")
        out.append(b"S%d:" % len(raw) + raw + b";")
    elif isinstance(obj, (list, tuple)):
        out.append((b"L%d[" if isinstance(obj, list) else b"T%d[") % len(obj))
        for item in obj:
            _canon(item, out)
        out.append(b"]")
    elif isinstance(obj, dict):
        pairs = []
        for k, v in obj.items():
            kb, vb = [], []
            _canon(k, kb)
            _canon(v, vb)
            pairs.append((b"".join(kb), b"".join(vb)))
        pairs.sort()
        out.append(b"D%d{" % len(pairs))
        for kb, vb in pairs:
            out.append(kb)
            out.append(vb)
        out.append(b"}")
    else:
        raise TypeError(f"cannot canonically encode {type(obj).__name__}")


def canonical_bytes(obj) -> bytes:
    """Canonical type-tagged bytes of obj (dict key order does not matter)."""
    out: list = []
    _canon(obj, out)
    return b"".join(out)


def hash_output(output: dict) -> str:
    """SHA-256 of an output record, serialized canonically and type-tagged."""
    return hashlib.sha256(canonical_bytes(output)).hexdigest()


def hash_record(case: Case, output: dict) -> str:
    """SHA-256 over one input record together with its output.

    The transform discards information (popcount, //100, %7), so hashing the
    output alone would let an input edit pass unnoticed. Binding the input in
    means any change to a generated input moves the hash.
    """
    return hashlib.sha256(
        canonical_bytes({"input": asdict(case), "output": output})
    ).hexdigest()


def run_once(n: int = N_CASES, seed: int = SEED) -> list[str]:
    """One full replay: corpus -> transform -> per-record SHA-256 list."""
    return [hash_record(c, transform(c)) for c in generate_corpus(n, seed)]


def corpus_hash(hashes: list[str]) -> str:
    """Single SHA-256 over the ordered list of per-case hashes (the corpus root)."""
    if len(hashes) == 0:
        raise ValueError("corpus_hash of an empty run is refused")
    h = hashlib.sha256()
    for digest in hashes:
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError(f"not a SHA-256 hex digest: {digest!r}")
        h.update(digest.encode("ascii"))
    return h.hexdigest()


_CHILD = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "import replay\n"
    "sys.stdout.write('\\n'.join(replay.run_once(int(sys.argv[2]), int(sys.argv[3]))))\n"
)


def run_once_isolated(n: int = N_CASES, seed: int = SEED) -> list[str]:
    """A replay in a separate process with a different hash seed and time zone.

    Divergence between this and an in-process run exposes any dependence on the
    process (pid, hash randomization, time zone) that one process cannot see.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "4343" if env.get("PYTHONHASHSEED") == "4242" else "4242"
    env["TZ"] = "UTC" if env.get("TZ") == "Pacific/Kiritimati" else "Pacific/Kiritimati"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, here, str(n), str(seed)],
        capture_output=True, text=True, env=env, timeout=600, cwd=here,
    )
    if r.returncode != 0:
        raise RuntimeError(f"isolated replay failed rc={r.returncode}: {r.stderr[-300:]}")
    return r.stdout.split("\n") if r.stdout else []


def count_divergences(run_a: list[str], run_b: list[str]) -> int:
    """Number of positions where two runs differ. 0 == perfect determinism."""
    if len(run_a) == 0 or len(run_b) == 0:
        raise ValueError("count_divergences of an empty run is refused")
    if len(run_a) != len(run_b):
        # Length mismatch -> every extra/missing position counts as divergence.
        return abs(len(run_a) - len(run_b)) + sum(
            1 for a, b in zip(run_a, run_b) if a != b
        )
    return sum(1 for a, b in zip(run_a, run_b) if a != b)


def main(argv=()) -> int:
    if argv:
        print("usage: replay.py (takes no arguments)", file=sys.stderr)
        return 2

    # Replay A in this process; replay B in a separate process with a different
    # PYTHONHASHSEED and TZ, so process-dependent behaviour shows as divergence.
    problems = []
    try:
        run_a = run_once()
        run_b = run_once_isolated(N_CASES, SEED)
        divergence = count_divergences(run_a, run_b)
        root = corpus_hash(run_a)
        if len(run_a) != N_CASES:
            problems.append(f"expected {N_CASES} cases, got {len(run_a)}")
        if divergence != 0:
            problems.append(f"divergence={divergence} (expected 0)")
        if corpus_hash(run_b) != root:
            problems.append("corpus root differed across runs")
        if root != EXPECTED_CORPUS_HASH:
            problems.append(f"corpus_hash={root} is not the pinned {EXPECTED_CORPUS_HASH}")
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        # Explicit checks, not assert: python -O must not be able to skip the claim.
        problems.append(f"{type(e).__name__}: {e}")
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1

    line = (
        f"{N_CASES:,} deterministic replays | "
        f"{divergence} divergence | "
        f"corpus_hash={root}"
    )
    # A closed or failing stdout must not look like success.
    if sys.stdout is None:
        return 1
    try:
        print(line)
        sys.stdout.flush()
    except (OSError, ValueError):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
