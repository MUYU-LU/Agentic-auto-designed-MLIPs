from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from runtime_common import (
    STAGING_RUNTIME_ROOT,
    load_config,
    remote_runtime_root,
    remote_unit_path,
    resolve_unit,
    rsync_from_remote,
    rsync_to_remote,
    run_remote_bash,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--profile-name", default="mad10k_adaptation")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-frames", type=int, default=None)
    parser.add_argument("--max-val-frames", type=int, default=None)
    parser.add_argument("--max-test-frames", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    if not config.get("remote_conda_env"):
        raise SystemExit("Missing remote_conda_env in config.json.")

    local_unit = resolve_unit(args.unit, STAGING_RUNTIME_ROOT)
    runtime_remote = remote_runtime_root(config)
    remote_tools = f"{runtime_remote}/tools"
    remote_script = f"{remote_tools}/run_adaptation_profile.py"
    local_script = Path(__file__).resolve().parent / "run_adaptation_profile.py"

    sync_script = rsync_to_remote(local_script.parent, remote_tools, config=config)
    if sync_script.returncode != 0:
        raise SystemExit(sync_script.stderr or sync_script.stdout or sync_script.returncode)

    remote_python = f"{config['remote_conda_env'].rstrip('/')}/bin/python"
    remote_profile = f"{runtime_remote}/benchmark/profiles/mad10k"
    remote_unit = remote_unit_path(args.unit, config)
    extra = []
    for flag, value in [
        ("--max-train-frames", args.max_train_frames),
        ("--max-val-frames", args.max_val_frames),
        ("--max-test-frames", args.max_test_frames),
    ]:
        if value is not None:
            extra.extend([flag, str(value)])

    command = " ".join(
        [
            shlex.quote(remote_python),
            shlex.quote(remote_script),
            "--runtime-root",
            shlex.quote(runtime_remote),
            "--unit",
            shlex.quote(args.unit),
            "--profile-root",
            shlex.quote(remote_profile),
            "--profile-name",
            shlex.quote(args.profile_name),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            *[shlex.quote(item) for item in extra],
        ]
    )

    script = f"""
set -euo pipefail
test -d {shlex.quote(remote_profile)}
test -d {shlex.quote(remote_unit)}
{command}
"""
    result = run_remote_bash(script, config=config, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr or "")
        sys.stdout.write(result.stdout or "")
        # Still fetch the failure summary if the script wrote one.
        rsync_from_remote(f"{remote_unit}/outputs/profiles/{args.profile_name}/", local_unit / "outputs" / "profiles" / args.profile_name, config=config)
        raise SystemExit(result.returncode)

    fetch = rsync_from_remote(f"{remote_unit}/outputs/profiles/{args.profile_name}/", local_unit / "outputs" / "profiles" / args.profile_name, config=config)
    if fetch.returncode != 0:
        raise SystemExit(fetch.stderr or fetch.stdout or fetch.returncode)
    print(local_unit / "outputs" / "profiles" / args.profile_name / "summary.json")


if __name__ == "__main__":
    main()
