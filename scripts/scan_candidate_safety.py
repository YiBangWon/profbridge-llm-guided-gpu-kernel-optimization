from __future__ import annotations

import argparse
import re
from pathlib import Path

SECRET_PATTERNS = [
    re.compile("OPENAI" + "_API_KEY"),
    re.compile("s" + "k-" + r"[A-Za-z0-9_-]{20,}"),
    re.compile("Authori" + "zation" + r"\s*:", re.I),
    re.compile("Bear" + "er" + r"\s+[A-Za-z0-9._-]{20,}", re.I),
    re.compile("BEGIN " + r"(?:RSA |OPENSSH |PRIVATE )?KEY"),
    re.compile("ssh" + "pass"),
]

SUSPICIOUS_PATTERNS = [
    re.compile(r"from\s+.*\s+import\s+Model\b"),
    re.compile(r"\btry\s*:", re.M),
    re.compile(r"\bexcept\b", re.M),
    re.compile(r"\.cpu\s*\("),
    re.compile(r"torch\.zeros|torch\.ones|torch\.empty"),
    re.compile(r"open\s*\(|socket\.|requests\.|urllib\."),
]

def scan_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    issues = []
    for pat in SECRET_PATTERNS:
        if pat.search(text):
            issues.append("secret-like pattern")
    for pat in SUSPICIOUS_PATTERNS:
        if pat.search(text):
            issues.append(f"suspicious pattern: {pat.pattern}")
    return {"path": str(path), "passed": not issues, "issues": issues}

def main() -> int:
    parser = argparse.ArgumentParser(description="Scan generated GPU candidate code for safety and benchmark loopholes.")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    ok = True
    for raw in args.paths:
        result = scan_file(Path(raw))
        ok = ok and result["passed"]
        print(result)
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
