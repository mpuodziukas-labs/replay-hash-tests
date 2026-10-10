"""Adversarial regression tests. Each test replays an attack on the code or on a temp copy of it."""
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import replay  # noqa: E402


def _copy(tmp_path, with_git=False):
    dst = tmp_path / "repo"
    ign = ["__pycache__", ".pytest_cache"] + ([] if with_git else [".git"])
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(*ign))
    return dst


def _run(cwd, *args, env_extra=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True,
                          text=True, env=env, timeout=300)


def _pin():
    return replay.EXPECTED_CORPUS_HASH


# --- R1: inputs are bound into the corpus hash ----------------------------------

def _rot(f):
    return ((f << 1) + (f >> 7)) & 255


def test_r1_flags_bit_rotation_moves_the_corpus_hash(monkeypatch):
    g = replay.generate_case

    def tampered(i, s=replay.SEED):
        c = g(i, s)
        return c if i in (0, 47311) else replay.Case(c.index, c.category, c.region, c.amount, _rot(c.flags))

    monkeypatch.setattr(replay, "generate_case", tampered)
    assert replay.corpus_hash(replay.run_once()) != _pin()
    assert replay.main() != 0


def test_r1_amount_plus_7_moves_the_corpus_hash(monkeypatch):
    g = replay.generate_case

    def tampered(i, s=replay.SEED):
        c = g(i, s)
        # +7 keeps amount//100 and (amount+index)%7 for amount%100 < 93: output unchanged
        if i in (0, 47311) or c.amount % 100 >= 93:
            return c
        return replay.Case(c.index, c.category, c.region, c.amount + 7, c.flags)

    monkeypatch.setattr(replay, "generate_case", tampered)
    assert replay.corpus_hash(replay.run_once()) != _pin()
    assert replay.main() != 0


def test_r1_record_hash_binds_input_to_output():
    c = replay.generate_case(5)
    c2 = replay.Case(c.index, c.category, c.region, c.amount, _rot(c.flags))
    out = replay.transform(c)
    assert replay.hash_record(c, out) != replay.hash_record(c2, out)
    assert replay.hash_record(c, out) != replay.hash_output(out)


# --- R2: the pin is anchored outside the tree a change edits ----------------------

def _git(cwd, *a):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "tag.gpgsign=false", "-c", "commit.gpgsign=false",
                           *a], capture_output=True, text=True, check=True)


def _repin(repo):
    new = "f" * 64
    old = _pin()
    for f in ("replay.py", "REFERENCE"):
        p = Path(repo) / f
        p.write_text(p.read_text().replace(old, new))


def test_r2_repin_without_reference_change_line_fails(tmp_path):
    repo = _copy(tmp_path, with_git=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-verify", "--allow-empty", "-m", "base")
    _git(repo, "tag", "base")
    _repin(repo)
    _git(repo, "commit", "-qam", "fix: sneaky repin")
    r = _run(repo, "check_reference.py", "--base", "base")
    assert r.returncode == 1, r.stdout + r.stderr


def test_r2_repin_with_reference_change_line_passes(tmp_path):
    repo = _copy(tmp_path, with_git=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-verify", "--allow-empty", "-m", "base")
    _git(repo, "tag", "base")
    _repin(repo)
    _git(repo, "commit", "-qam", "fix: repin\n\nREFERENCE-CHANGE: scoring weight changed")
    r = _run(repo, "check_reference.py", "--base", "base")
    assert r.returncode == 0, r.stdout + r.stderr


def test_r2_code_pin_must_equal_reference(tmp_path):
    repo = _copy(tmp_path)
    p = repo / "replay.py"
    p.write_text(p.read_text().replace(_pin(), "e" * 64))
    assert _run(repo, "check_reference.py").returncode == 1


def test_r2_reference_equals_every_pin_in_the_tree():
    ref = (ROOT / "REFERENCE").read_text().strip()
    assert ref == f"corpus_hash={_pin()}"
    assert _run(ROOT, "check_reference.py").returncode == 0
    for f in ("tests/test_replay.py", "tests/test_redteam.py"):
        assert set(re.findall(r'"([0-9a-f]{64})"', (ROOT / f).read_text())) >= {_pin()}


def test_r2_ci_compares_against_the_base_branch():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "check_reference.py" in ci and "--base" in ci and "fetch-depth: 0" in ci


# --- R3: the second replay is another process with another env ---------------------

_FIXTURES = {
    "pid": ("score = (norm_amount", "score = (os.getpid() + norm_amount", {}),
    "hashseed": ("score = (norm_amount", "score = (hash(case.category) % 1000 + norm_amount", {}),
    "tz": ("score = (norm_amount", "score = (__import__('time').timezone + norm_amount", {"TZ": "UTC"}),
}


@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_r3_process_dependent_pipeline_fails_determinism(tmp_path, name):
    old, new, env = _FIXTURES[name]
    repo = _copy(tmp_path)
    p = repo / "replay.py"
    src = p.read_text()
    assert old in src
    p.write_text(src.replace(old, new, 1).replace("import re\n", "import re, os\n", 1))
    r = _run(repo, "replay.py", env_extra=env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "divergence=" in r.stderr, r.stderr
    assert "deterministic replays" not in r.stdout


def test_r3_isolated_replay_matches_in_process_replay():
    a = replay.run_once(500)
    assert replay.run_once_isolated(500) == a


# --- R4: type-tagged canonical encoding -----------------------------------------------

def test_r4_int_key_and_str_key_differ():
    assert replay.hash_output({1: "a"}) != replay.hash_output({"1": "a"})


def test_r4_tuple_and_list_differ():
    assert replay.hash_output({"x": (1, 2)}) != replay.hash_output({"x": [1, 2]})


def test_r4_bool_int_str_float_none_all_differ():
    hs = [replay.hash_output({"x": v}) for v in [True, 1, "1", 1.0, None, 0, False]]
    assert len(set(hs)) == len(hs)


def test_r4_key_order_still_irrelevant_and_unknown_types_refused():
    assert replay.hash_output({"a": 1, "b": 2}) == replay.hash_output({"b": 2, "a": 1})
    with pytest.raises(TypeError):
        replay.hash_output({"x": {1, 2}})


# --- R5: non-finite floats refused ----------------------------------------------------

@pytest.mark.parametrize("v", [float("nan"), float("inf"), float("-inf"), 1e400])
def test_r5_non_finite_float_refused(v):
    with pytest.raises(ValueError):
        replay.hash_output({"x": v})


def test_r5_nan_output_makes_main_exit_1(monkeypatch, capsys):
    orig = replay.transform
    monkeypatch.setattr(replay, "transform", lambda c: dict(orig(c), score=float("nan")))
    assert replay.main() == 1
    assert "deterministic replays" not in capsys.readouterr().out


# --- R6: a closed stdout is a failure -----------------------------------------------------

def test_r6_closed_stdout_exits_nonzero():
    cmd = f"{shlex.quote(sys.executable)} replay.py >&-"
    r = subprocess.run(["sh", "-c", cmd], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0


@pytest.mark.skipif(not os.path.exists("/dev/full"), reason="no /dev/full on this OS")
def test_r6_full_stdout_exits_nonzero():
    cmd = f"{shlex.quote(sys.executable)} replay.py > /dev/full"
    r = subprocess.run(["sh", "-c", cmd], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0


# --- R7: empty runs refused ------------------------------------------------------------------

def test_r7_empty_run_helpers_raise():
    with pytest.raises(ValueError):
        replay.count_divergences([], [])
    with pytest.raises(ValueError):
        replay.corpus_hash([])
    with pytest.raises(ValueError):
        replay.count_divergences([], ["0" * 64])


# --- R8: the 60 second claim is a timed assertion -----------------------------------------------

def test_cli_wall_time_under_60s():
    t0 = time.monotonic()
    r = _run(ROOT, "replay.py")
    elapsed = time.monotonic() - t0
    assert r.returncode == 0, r.stderr
    assert elapsed < 60, f"{elapsed:.1f}s"


def test_r8_readme_names_the_timed_test_and_drops_unmeasured_words():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "test_cli_wall_time_under_60s" in text
    assert "runs in seconds" not in text


# --- R9: pinned toolchain ---------------------------------------------------------------------

def test_r9_pytest_is_pinned_exactly():
    lines = [ln for ln in (ROOT / "requirements.txt").read_text().splitlines()
             if ln.strip().startswith("pytest")]
    assert lines and all(re.match(r"pytest==\d+\.\d+\.\d+(\s*;.*)?$", ln.strip()) for ln in lines), lines


def test_r9_pytest_pin_has_the_tmpdir_fix():
    # GHSA-6w46-j5rx-g56g: pytest below 9.0.3 has vulnerable tmpdir handling.
    pins = re.findall(r"^pytest==(\d+)\.(\d+)\.(\d+)", (ROOT / "requirements.txt").read_text(), re.M)
    assert pins and all(tuple(map(int, p)) >= (9, 0, 3) for p in pins), pins


def test_r9_actions_pinned_to_commit_shas():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    uses = re.findall(r"uses:\s*(\S+)", ci)
    assert uses
    assert all(re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u) for u in uses), uses


# --- R10: no dead skip entries -------------------------------------------------------------------

def test_r10_no_dead_skip_for_missing_corpus_files():
    text = (ROOT / "tests/test_stranger.py").read_text()
    assert "build_corpus" not in text
    assert 'startswith("corpus/")' not in text
