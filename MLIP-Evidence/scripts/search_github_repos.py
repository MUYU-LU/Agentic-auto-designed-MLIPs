#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import requests


def search_github(query: str, topk: int) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "MLIP-Evidence",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(
        "https://api.github.com/search/repositories",
        headers=headers,
        params={"q": query, "sort": "stars", "order": "desc", "per_page": topk},
        timeout=30,
    )
    response.raise_for_status()
    items = []
    for repo in response.json().get("items", []):
        items.append(
            {
                "full_name": repo.get("full_name"),
                "html_url": repo.get("html_url"),
                "description": repo.get("description"),
                "stars": repo.get("stargazers_count"),
                "updated_at": repo.get("updated_at"),
                "language": repo.get("language"),
            }
        )
    return {"query": query, "results": items}


def main() -> None:
    parser = argparse.ArgumentParser(description="Search GitHub repositories for MLIP-Evidence.")
    parser.add_argument("--query", required=True, help="GitHub search query.")
    parser.add_argument("--topk", type=int, default=5, help="Maximum repositories to return.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args()

    result = search_github(args.query, args.topk)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
