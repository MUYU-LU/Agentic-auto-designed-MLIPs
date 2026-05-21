#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_RE = re.compile(r"https?://(?:www\.)?(?:github|gitlab)\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GitHub/GitLab repository links from text.")
    parser.add_argument("path", nargs="?", help="Text file to scan. Reads stdin when omitted.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args()

    if args.path:
        text = Path(args.path).expanduser().read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    links = sorted(set(REPO_RE.findall(text)))
    print(json.dumps({"repo_links": links}, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
