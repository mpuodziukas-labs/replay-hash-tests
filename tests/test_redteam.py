"""Red-team regression tests. Each case was a wrong answer, a silent pass,
or a README drift before the fix."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import replay  # noqa: E402

PINNED = "3e806db2c4978dfae932b98d3d2dfbcbc76cd4f48f5b29b6181be9eb531167a8"


def _cli(*args, env_extra=None, flags=()):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, *flags, str(ROOT / "replay.py"), *args],
        capture_output=True, text=True, env=env, timeout=120,
    )


# --- the claim must not depend on assert statements -------------------------

_DIVERGE = (
    "import sys, itertools, replay\n"
    "c = itertools.count()\n"
    "orig = replay.transform\n"
    "def bad(case):\n"
    "    out = orig(case)\n"
    "    if case.index == 7:\n"
    "        out = dict(out, score=next(c))\n"
    "    return out\n"
    "replay.transform = bad\n"
    "sys.exit(replay.main())\n"
)


@pytest.mark.parametrize("flags", [(), ("-O",)])
def test_divergence_exits_nonzero_even_under_O(flags):
    r = subprocess.run([sys.executable, *flags, "-c", _DIVERGE], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0, r.stdout
    assert "deterministic replays" not in r.stdout


def test_wrong_size_exits_nonzero_even_under_O():
    code = ("import sys, replay\n"
            "replay.N_CASES = 10\n"
            "sys.exit(replay.main())\n")
    r = subprocess.run([sys.executable, "-O", "-c", code], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0, r.stdout


def test_deterministic_but_changed_pipeline_fails_pinned_hash(monkeypatch, capsys):
    orig = replay.transform
    monkeypatch.setattr(replay, "transform",
                        lambda c: dict(orig(c), score=(orig(c)["score"] + 1) % 100_000))
    assert replay.main() != 0
    assert "deterministic replays" not in capsys.readouterr().out


def test_pinned_hash_lives_in_code():
    assert replay.EXPECTED_CORPUS_HASH == PINNED


def test_cli_rejects_extra_arguments():
    r = _cli("--n", "10")
    assert r.returncode == 2
    assert "deterministic replays" not in r.stdout


# --- input validation ---------------------------------------------------------

@pytest.mark.parametrize("n", [0, -1, -47312])
def test_empty_or_negative_corpus_is_an_error(n):
    with pytest.raises(ValueError):
        replay.run_once(n)


@pytest.mark.parametrize("n", [True, 1.5, "3", None])
def test_non_int_corpus_size_rejected(n):
    with pytest.raises((ValueError, TypeError)):
        replay.run_once(n)


@pytest.mark.parametrize("seed", [-1, 1 << 64, (1 << 64) + replay.SEED, True, 1.5])
def test_seed_must_be_a_distinct_u64(seed):
    with pytest.raises((ValueError, TypeError)):
        replay.generate_case(0, seed)


def test_seeds_do_not_alias_modulo_2_64():
    with pytest.raises(ValueError):
        replay.generate_case(0, replay.SEED + (1 << 64))


@pytest.mark.parametrize("idx", [-1, True, 1.5, "1"])
def test_bad_index_rejected(idx):
    with pytest.raises((ValueError, TypeError)):
        replay.generate_case(idx)


@pytest.mark.parametrize("digest", ["a", "ab" * 31, "g" * 64, "A" * 64, "é" * 64, 5, None])
def test_corpus_hash_rejects_malformed_digests(digest):
    with pytest.raises((ValueError, TypeError)):
        replay.corpus_hash([digest])


def test_corpus_hash_is_not_concatenation_ambiguous():
    a = "0" * 64
    b = "1" * 64
    assert replay.corpus_hash([a, b]) != replay.corpus_hash([b, a])
    assert replay.corpus_hash([a]) != replay.corpus_hash([a, a])
    with pytest.raises(ValueError):  # R7: empty run refused (was: valid root)
        replay.corpus_hash([])


def test_count_divergences_length_and_position():
    x = ["0" * 64] * 3
    assert replay.count_divergences(x, x) == 0
    assert replay.count_divergences(x, x[:2]) == 1
    assert replay.count_divergences(x, ["1" * 64] + x[1:]) == 1
    with pytest.raises(ValueError):  # R7: empty run refused (was: 3)
        replay.count_divergences([], x)


# --- golden vectors independent of the corpus root ---------------------------

def test_golden_first_and_last_case():
    first = replay.generate_case(0)
    assert (first.category, first.region, first.amount, first.flags) == ("echo", "us-east", 78936, 22)
    assert replay.hash_output(replay.transform(first)) == \
        "4321a2ced0a7e565153bfb714ace41783f7fcf6cfd6c72cc16e3ec9b8628ba58"
    last = replay.generate_case(47311)
    assert (last.category, last.region, last.amount, last.flags) == ("delta", "us-east", 280184, 8)
    assert replay.hash_output(replay.transform(last)) == \
        "39ed68a6174bff646fbffe564ad6e15a663cef50e398be64ecb13b69fd5efbea"


def test_transform_output_is_integer_and_string_only():
    for c in replay.generate_corpus(2000):
        out = replay.transform(c)
        assert all(isinstance(v, (int, str)) and not isinstance(v, bool) for v in out.values())


# --- environment independence ---------------------------------------------------

@pytest.mark.parametrize("env", [
    {"PYTHONHASHSEED": "1"},
    {"PYTHONHASHSEED": "987654"},
    {"PYTHONHASHSEED": "random", "TZ": "Pacific/Kiritimati"},
    {"LC_ALL": "tr_TR.UTF-8", "LANG": "tr_TR.UTF-8"},
    {"PYTHONIOENCODING": "ascii"},
])
def test_output_independent_of_environment(env):
    r = _cli(env_extra=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"47,312 deterministic replays | 0 divergence | corpus_hash={PINNED}"


# --- README must match the code --------------------------------------------------

def test_readme_hash_lines_match_real_output():
    real = _cli().stdout.strip()
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if "deterministic replays |" in ln]
    assert len(lines) == 2
    assert all(ln.strip() == real for ln in lines)
    assert set(re.findall(r"corpus_hash=([0-9a-f]+)", text)) == {PINNED}
    assert text.count("47,312") >= 3
