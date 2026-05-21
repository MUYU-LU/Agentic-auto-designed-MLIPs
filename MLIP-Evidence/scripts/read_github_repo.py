#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse


KEY_FILE_HINTS = (
    "train.py",
    "model.py",
    "main.py",
    "requirements.txt",
    "pyproject.toml",
)

DEFAULT_CACHE_ROOT = Path(
    os.environ.get(
        "MLIP_EVIDENCE_REPO_CACHE",
        str(Path.cwd() / ".cache" / "MLIP-Evidence" / "repos"),
    )
)


def parse_repo(repo: str) -> tuple[str, str]:
    if repo.startswith("http"):
        parsed = urlparse(repo)
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"Cannot parse repo URL: {repo}")
        return parts[0], parts[1].removesuffix(".git")
    owner, name = repo.split("/", 1)
    return owner, name.removesuffix(".git")


def git_env(*, no_proxy: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    if no_proxy:
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            env.pop(key, None)
    return env


def run_git(args: list[str], cwd: Path | None = None, *, no_proxy: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        env=git_env(no_proxy=no_proxy),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )


def ensure_repo(owner: str, name: str, cache_root: Path, *, no_proxy: bool = False) -> tuple[Path, str, str | None]:
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{owner}__{name}"
    url = f"https://github.com/{owner}/{name}.git"
    warning = None
    if (target / ".git").exists():
        result = run_git(["pull", "--ff-only"], cwd=target, no_proxy=no_proxy)
        if result.returncode == 0:
            return target, "pull", None
        warning = result.stderr.strip() or result.stdout.strip() or "git pull failed"
        head = run_git(["rev-parse", "--verify", "HEAD"], cwd=target, no_proxy=no_proxy)
        files = run_git(["ls-files"], cwd=target, no_proxy=no_proxy)
        if head.returncode == 0 and files.returncode == 0 and files.stdout.strip():
            return target, "cache_reuse_after_pull_failed", warning
        shutil.rmtree(target)
        warning = f"removed invalid cached repo after pull failed: {warning}"
    if target.exists():
        shutil.rmtree(target)
    result = run_git(["clone", "--depth", "1", url, str(target)], no_proxy=no_proxy)
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip() or result.stdout.strip()}")
    return target, "clone", warning


def should_read(path: str) -> bool:
    lower = path.lower()
    name = Path(path).name.lower()
    if lower.endswith(KEY_FILE_HINTS):
        return True
    if name.endswith((".yaml", ".yml")) and ("config" in name or "/config" in lower):
        return True
    if lower.endswith(".py") and any(
        marker in lower
        for marker in (
            "/model/",
            "/models/",
            "/layers/",
            "/modules/",
            "/nn/",
            "/representation/",
            "/representations/",
            "/training/",
            "/examples/",
        )
    ):
        return True
    if lower.endswith(".py") and any(
        token in name
        for token in (
            "train",
            "trainer",
            "interaction",
            "message",
            "basis",
            "radial",
            "spherical",
            "envelope",
            "readout",
            "output",
            "energy",
            "force",
        )
    ):
        return True
    return "/configs/" in lower or "/config/" in lower or "/examples/" in lower


def read_file(path: Path, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def git_scalar(args: list[str], cwd: Path, *, no_proxy: bool = False) -> str:
    result = run_git(args, cwd=cwd, no_proxy=no_proxy)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def read_repo(repo: str, max_files: int, max_file_chars: int, cache_root: Path, *, no_proxy: bool = False) -> dict:
    owner, name = parse_repo(repo)
    repo_root, git_action, git_warning = ensure_repo(owner, name, cache_root, no_proxy=no_proxy)
    files_result = run_git(["ls-files"], cwd=repo_root, no_proxy=no_proxy)
    if files_result.returncode != 0:
        raise RuntimeError(files_result.stderr.strip() or "git ls-files failed")
    tree_paths = [line for line in files_result.stdout.splitlines() if line]
    selected = [path for path in tree_paths if should_read(path)][:max_files]
    key_files = {
        path: text
        for path in selected
        if (text := read_file(repo_root / path, max_file_chars))
    }
    readme = ""
    for name_candidate in ("README.md", "README.rst", "README.txt", "README"):
        candidate = repo_root / name_candidate
        if candidate.exists():
            readme = read_file(candidate, 8000)
            break
    return {
        "repo": f"{owner}/{name}",
        "html_url": f"https://github.com/{owner}/{name}",
        "method": "git",
        "git_action": git_action,
        "git_warning": git_warning,
        "cache_root": str(cache_root),
        "cache_path": str(repo_root),
        "commit": git_scalar(["rev-parse", "HEAD"], cwd=repo_root, no_proxy=no_proxy),
        "branch": git_scalar(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, no_proxy=no_proxy),
        "readme_excerpt": readme,
        "tree_paths": tree_paths[:500],
        "key_files_content": key_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read GitHub repo through local git clone/pull, README, tree, and key files.")
    parser.add_argument("--repo", required=True, help="owner/name or GitHub URL.")
    parser.add_argument("--max-files", type=int, default=20, help="Maximum key files to read.")
    parser.add_argument("--max-file-chars", type=int, default=12000, help="Maximum chars per key file.")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT), help="Git clone/pull cache. Keep this outside research_runtime; package artifacts record cache_path and commit.")
    parser.add_argument("--no-proxy", action="store_true", help="Unset proxy variables for git. Default keeps the current shell proxy environment.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args()

    result = read_repo(args.repo, args.max_files, args.max_file_chars, Path(args.cache_root).expanduser(), no_proxy=args.no_proxy)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
