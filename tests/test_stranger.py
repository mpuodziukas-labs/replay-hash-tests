import subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = {"LICENSE", "tests/test_stranger.py"}
DASHES = ("\u2014", "\u2013")


def test_no_em_or_en_dash_in_tracked_text():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    bad = []
    for f in files:
        if f in SKIP:
            continue
        text = (ROOT / f).read_text(encoding="utf-8", errors="ignore")
        if any(d in text for d in DASHES):
            bad.append(f)
    assert not bad, bad


def test_readme_uses_python3_everywhere():
    import re
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"^python (?!3)", text, re.M), "bare `python` is missing on many machines"
