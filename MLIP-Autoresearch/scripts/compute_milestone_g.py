from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime_common import LEDGER, STAGING_RUNTIME_ROOT, q_fields_for_unit, resolve_unit

MILESTONES = LEDGER / "milestones"


def q_breakdown(unit_root: Path) -> dict:
    fields = q_fields_for_unit(unit_root)
    if fields.get("Q_total") is None:
        raise SystemExit("missing Q values in benchmark metrics")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    child = q_breakdown(resolve_unit(args.current, STAGING_RUNTIME_ROOT))
    parent = q_breakdown(resolve_unit(args.reference, STAGING_RUNTIME_ROOT))
    g_delta = child["Q_total"] - parent["Q_total"]
    out = {
        "current": args.current,
        "reference": args.reference,
        "Q_total_child": child["Q_total"],
        "Q_total_parent": parent["Q_total"],
        "G_delta": g_delta,
    }
    out_path = MILESTONES / args.current.replace("/", "__") / "total.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
