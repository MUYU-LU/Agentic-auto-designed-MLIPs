#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


REPO_RE = re.compile(r"https?://(?:www\.)?(?:github|gitlab)\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?")


def extract_with_pdftotext(path: Path, pages: int, max_chars: int) -> dict | None:
    if not shutil.which("pdftotext"):
        return None
    cmd = ["pdftotext", "-f", "1", "-l", str(max(pages, 1)), str(path), "-"]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        return {
            "pdf": str(path),
            "readable": False,
            "method": "pdftotext",
            "error": result.stderr.strip() or f"pdftotext exited {result.returncode}",
            "text": "",
            "repo_links": [],
        }
    return {
        "pdf": str(path),
        "readable": True,
        "method": "pdftotext",
        "pages_attempted": pages,
        "text": text[:max_chars],
        "repo_links": sorted(set(REPO_RE.findall(text))),
    }


def extract_with_pypdf(path: Path, pages: int, max_chars: int) -> dict:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - depends on local env
        return {
            "pdf": str(path),
            "readable": False,
            "method": "pypdf",
            "error": f"pypdf import failed: {exc}",
            "text": "",
            "repo_links": [],
        }

    try:
        reader = PdfReader(str(path))
        chunks = []
        for page in reader.pages[:pages]:
            chunks.append(page.extract_text() or "")
        text = "\n".join(chunks).strip()
        return {
            "pdf": str(path),
            "readable": bool(text),
            "method": "pypdf",
            "pages_attempted": min(pages, len(reader.pages)),
            "text": text[:max_chars],
            "repo_links": sorted(set(REPO_RE.findall(text))),
        }
    except Exception as exc:  # pragma: no cover - depends on PDF details
        return {
            "pdf": str(path),
            "readable": False,
            "method": "pypdf",
            "error": str(exc),
            "text": "",
            "repo_links": [],
        }


def extract_pdf(path: Path, pages: int, max_chars: int) -> dict:
    pdftotext_result = extract_with_pdftotext(path, pages, max_chars)
    if pdftotext_result and pdftotext_result.get("readable"):
        return pdftotext_result
    fallback = extract_with_pypdf(path, pages, max_chars)
    if pdftotext_result and not fallback.get("readable"):
        fallback["pdftotext_error"] = pdftotext_result.get("error")
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract first pages and repo links from a local PDF.")
    parser.add_argument("pdf", help="PDF path.")
    parser.add_argument("--pages", type=int, default=6, help="Number of pages to read.")
    parser.add_argument("--max-chars", type=int, default=12000, help="Maximum text chars to emit.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args()

    result = extract_pdf(Path(args.pdf).expanduser().resolve(), args.pages, args.max_chars)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
