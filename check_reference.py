#!/usr/bin/env python3
"""Fail if the pinned corpus hash moved without an explicit REFERENCE-CHANGE.

REFERENCE holds the pinned hash outside replay.py and the tests. A change set
that moves it must carry a line starting `REFERENCE-CHANGE:` in a commit
message, so a re-pin is always a visible, deliberate act.

  python3 check_reference.py                 in-tree consistency only
  python3 check_reference.py --base <ref>    also compare with REFERENCE at <ref>

Exit 0 = ok, 1 = violation, 2 = usage. Needs git for --base.
Limitation: anyone who can write the base branch can still move the pin.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LINE = re.compile(r"corpus_hash=([0-9a-f]{64})\n?")
CODE_PIN = re.compile(r'EXPECTED_CORPUS_HASH\s*=\s*\(?\s*"([0-9a-f]{64})"\s*\)?')


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def check(base=None) -> int:
    ref = ROOT / "REFERENCE"
    if not ref.is_file():
        print("FAIL: REFERENCE file missing", file=sys.stderr)
        return 1
    m = LINE.fullmatch(ref.read_text(encoding="utf-8"))
    if not m:
        print("FAIL: REFERENCE must be exactly one line corpus_hash=<64 hex>", file=sys.stderr)
        return 1
    pinned = m.group(1)
    code = CODE_PIN.search((ROOT / "replay.py").read_text(encoding="utf-8"))
    if not code or code.group(1) != pinned:
        print("FAIL: replay.py EXPECTED_CORPUS_HASH does not equal REFERENCE", file=sys.stderr)
        return 1
    if base is None:
        print("OK in-tree: REFERENCE equals replay.py pin (no base compared)")
        return 0
    if _git("rev-parse", "--verify", "--quiet", base + "^{commit}").returncode != 0:
        print(f"FAIL: base ref {base!r} not found", file=sys.stderr)
        return 1
    old = _git("show", f"{base}:REFERENCE")
    if old.returncode != 0:
        print(f"OK: {base} has no REFERENCE yet (first adoption)")
        return 0
    om = LINE.fullmatch(old.stdout)
    if om and om.group(1) == pinned:
        print(f"OK: pin unchanged from {base}")
        return 0
    log = _git("log", f"{base}..HEAD", "--format=%B")
    if log.returncode == 0 and re.search(r"^REFERENCE-CHANGE:", log.stdout, re.M):
        print(f"OK: pin moved from {base} with a REFERENCE-CHANGE line")
        return 0
    print(f"FAIL: pin moved from {base} with no REFERENCE-CHANGE line in any commit message",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == []:
        raise SystemExit(check())
    if len(a) == 2 and a[0] == "--base":
        raise SystemExit(check(a[1]))
    print("usage: check_reference.py [--base <ref>]", file=sys.stderr)
    raise SystemExit(2)
