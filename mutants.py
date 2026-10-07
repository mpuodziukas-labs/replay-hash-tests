#!/usr/bin/env python3
"""Mutation check: break replay.py in known ways and require the test suite to fail.

  python3 mutants.py

Exit 0 = the unmodified code passes AND every mutant is killed (tests fail).
Exit 1 = a mutant survived or the control run failed.
Works on a temporary copy (including .git, which the tests need); the real files are never edited.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))

# (name, exact text in replay.py, replacement). Each old text must occur once.
MUTANTS = [
    ("score weight changed", "risk_bits * 11", "risk_bits * 12"),
    ("input dropped from record hash", '{"input": asdict(case), "output": output}', '{"output": output}'),
    ("seed ignored by generator", "_splitmix64(seed ^ (index * 0x100000001B3))", "_splitmix64(index * 0x100000001B3)"),
    ("dict key order not canonical", "pairs.sort()", "pass"),
    ("empty run accepted by divergence check", "if len(run_a) == 0 or len(run_b) == 0:", "if False:"),
    ("bool accepted as an int", "isinstance(x, int) and not isinstance(x, bool)", "isinstance(x, int)"),
]


def run_tests(cwd):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
        cwd=cwd, capture_output=True, text=True, timeout=900,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    return r.returncode


def fresh_copy():
    d = os.path.join(tempfile.mkdtemp(prefix="mut-"), "repo")
    shutil.copytree(ROOT, d, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return d


def main():
    d = fresh_copy()
    try:
        if run_tests(d) != 0:
            print("control run FAILED: unmodified code must pass the tests")
            return 1
        survived = 0
        for name, old, new in MUTANTS:
            src = open(os.path.join(d, "replay.py"), encoding="utf-8").read()
            if src.count(old) != 1:
                print(f"BAD MUTANT {name}: pattern occurs {src.count(old)} times")
                return 1
            with open(os.path.join(d, "replay.py"), "w", encoding="utf-8") as fh:
                fh.write(src.replace(old, new))
            killed = run_tests(d) != 0
            print(f"{'killed  ' if killed else 'SURVIVED'} {name}")
            survived += 0 if killed else 1
            with open(os.path.join(d, "replay.py"), "w", encoding="utf-8") as fh:
                fh.write(src)
        print(f"mutants killed: {len(MUTANTS) - survived}/{len(MUTANTS)}")
        return 1 if survived else 0
    finally:
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
