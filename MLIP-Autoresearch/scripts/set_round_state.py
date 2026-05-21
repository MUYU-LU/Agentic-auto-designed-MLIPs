from __future__ import annotations

import argparse

from runtime_common import STAGING_RUNTIME_ROOT, load_config, sync_round_state_to_remote, update_round_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-state", default=None)
    parser.add_argument("--current-generation", default=None)
    parser.add_argument("--active-writer-unit", default=None)
    parser.add_argument("--active-evidence-task", default=None)
    parser.add_argument("--blocked-reason", default=None)
    parser.add_argument("--last-completed-generation", default=None)
    parser.add_argument("--continuation-source-unit", default=None)
    parser.add_argument("--active-proposal-directory", default=None)
    parser.add_argument("--active-selection-file", default=None)
    parser.add_argument("--materialized-units-root", default=None)
    parser.add_argument("--next-recommended-step", default=None)
    parser.add_argument("--sync-remote", action="store_true")
    args = parser.parse_args()

    updates = {}
    for key, value in {
        "workflow_state": args.workflow_state,
        "current_generation": args.current_generation,
        "active_writer_unit": args.active_writer_unit,
        "active_evidence_task": args.active_evidence_task,
        "blocked_reason": args.blocked_reason,
        "last_completed_generation": args.last_completed_generation,
        "continuation_source_unit": args.continuation_source_unit,
        "active_proposal_directory": args.active_proposal_directory,
        "active_selection_file": args.active_selection_file,
        "materialized_units_root": args.materialized_units_root,
        "next_recommended_step": args.next_recommended_step,
        "source_of_truth": "remote",
    }.items():
        if value is not None:
            updates[key] = value

    state = update_round_state(STAGING_RUNTIME_ROOT, **updates)
    if args.sync_remote:
        sync_round_state_to_remote(config=load_config())
    print(state)


if __name__ == "__main__":
    main()
