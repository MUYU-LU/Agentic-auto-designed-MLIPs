from __future__ import annotations

import argparse
import re
import shlex

from runtime_common import (
    STAGING_RUNTIME_ROOT,
    load_config,
    load_implementation_status,
    remote_unit_path,
    resolve_unit,
    rsync_from_remote,
    run_remote_bash,
    sync_status_files_to_remote,
    update_implementation_status,
)


def mad10k_force_grad_sanity_command(*, remote_log: str, gpu_prefix: str, samples: int) -> str:
    return (
        f"{gpu_prefix}$PY - <<'PY' >> {shlex.quote(remote_log)} 2>&1\n"
        "from __future__ import annotations\n"
        "\n"
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        "\n"
        "import torch\n"
        "\n"
        "root = pathlib.Path.cwd()\n"
        "sys.path.insert(0, str(root))\n"
        "\n"
        "from model.dataloader import make_dataset\n"
        "from model.model import EvolutionMLIP\n"
        "import model.train as train_mod\n"
        "\n"
        "cfg = json.loads((root / 'config.json').read_text())\n"
        "benchmark_root = (root / cfg['benchmark_root']).resolve()\n"
        "profile_root = benchmark_root / 'profiles' / 'mad10k'\n"
        "if not profile_root.exists():\n"
        "    raise SystemExit(f'mad10k profile root missing: {profile_root}')\n"
        "\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('CUDA is required for MAD10K force-gradient sanity')\n"
        "device = torch.device('cuda')\n"
        "\n"
        "model = EvolutionMLIP(\n"
        "    hidden_dim=getattr(train_mod, 'MODEL_HIDDEN_DIM', 96),\n"
        "    num_rbf=getattr(train_mod, 'MODEL_NUM_RBF', 32),\n"
        "    cutoff=getattr(train_mod, 'MODEL_CUTOFF', 5.0),\n"
        ").to(device)\n"
        "train_loader = train_mod.make_dataloader('mad10k', profile_root / 'train', batch_size=1, shuffle=False)\n"
        "atomref_fitted = train_mod.initialize_atomref_lstsq(\n"
        "    model,\n"
        "    train_loader,\n"
        "    ridge=float(cfg.get('atomref_lstsq_ridge', 1e-3)),\n"
        "    device=device,\n"
        ")\n"
        "if atomref_fitted:\n"
        "    model.atomref.weight.requires_grad_(False)\n"
        "model.train()\n"
        "\n"
        "target = max(1, int(" + str(int(samples)) + "))\n"
        "checked = 0\n"
        "periodic_seen = 0\n"
        "failures = []\n"
        "\n"
        "def move(sample: dict) -> dict:\n"
        "    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}\n"
        "\n"
        "def check_batch(label: str, batch: dict) -> None:\n"
        "    with torch.enable_grad():\n"
        "        energy, forces = model(batch)\n"
        "    if not torch.is_tensor(energy):\n"
        "        raise RuntimeError(f'{label}: energy is not a tensor')\n"
        "    if tuple(forces.shape) != tuple(batch['forces'].shape):\n"
        "        raise RuntimeError(f'{label}: force shape mismatch: predicted {tuple(forces.shape)} target {tuple(batch[\"forces\"].shape)}')\n"
        "    if not bool(torch.isfinite(energy.detach()).all().item()):\n"
        "        raise RuntimeError(f'{label}: non-finite energy')\n"
        "    if not bool(torch.isfinite(forces.detach()).all().item()):\n"
        "        raise RuntimeError(f'{label}: non-finite forces')\n"
        "\n"
        "synthetic_isolated = {\n"
        "    'numbers': torch.tensor([1], dtype=torch.long, device=device),\n"
        "    'positions': torch.zeros(1, 3, dtype=torch.float32, device=device),\n"
        "    'energy': torch.tensor(0.0, dtype=torch.float32, device=device),\n"
        "    'forces': torch.zeros(1, 3, dtype=torch.float32, device=device),\n"
        "    'cell': torch.eye(3, dtype=torch.float32, device=device) * 20.0,\n"
        "    'pbc': torch.tensor([False, False, False], dtype=torch.bool, device=device),\n"
        "}\n"
        "synthetic_periodic_isolated = dict(synthetic_isolated)\n"
        "synthetic_periodic_isolated['pbc'] = torch.tensor([True, True, True], dtype=torch.bool, device=device)\n"
        "check_batch('synthetic_isolated_no_neighbor', synthetic_isolated)\n"
        "check_batch('synthetic_periodic_isolated_no_neighbor', synthetic_periodic_isolated)\n"
        "\n"
        "def spaced_indices(n: int, count: int) -> list[int]:\n"
        "    if n <= 0 or count <= 0:\n"
        "        return []\n"
        "    if n <= count:\n"
        "        return list(range(n))\n"
        "    if count == 1:\n"
        "        return [0]\n"
        "    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})\n"
        "\n"
        "for split in ('train', 'val', 'test'):\n"
        "    ds = make_dataset('mad10k', profile_root / split)\n"
        "    first_indices = list(range(min(len(ds), max(target, 16))))\n"
        "    spaced = spaced_indices(len(ds), min(len(ds), max(target * 4, target)))\n"
        "    candidate_indices = sorted(set(first_indices + spaced))\n"
        "    for idx in candidate_indices:\n"
        "        sample = ds[idx]\n"
        "        pbc = sample.get('pbc')\n"
        "        if pbc is None or not bool(torch.as_tensor(pbc).to(torch.bool).any().item()):\n"
        "            continue\n"
        "        periodic_seen += 1\n"
        "        batch = move(sample)\n"
        "        try:\n"
        "            check_batch(f'{split}[{idx}]', batch)\n"
        "        except Exception as exc:\n"
        "            failures.append(f'{split}[{idx}]: {type(exc).__name__}: {exc}')\n"
        "            break\n"
        "        checked += 1\n"
        "        if checked >= target:\n"
        "            break\n"
        "    if failures or checked >= target:\n"
        "        break\n"
        "\n"
        "print(f'MAD10K force-gradient sanity: atomref_fitted={atomref_fitted} periodic_seen={periodic_seen} checked={checked} target={target}')\n"
        "if periodic_seen == 0:\n"
        "    raise SystemExit('MAD10K force-gradient sanity found no periodic samples')\n"
        "if failures:\n"
        "    raise SystemExit('MAD10K force-gradient sanity failed: ' + failures[0])\n"
        "if checked < min(target, periodic_seen):\n"
        "    raise SystemExit(f'MAD10K force-gradient sanity checked too few samples: {checked}/{min(target, periodic_seen)}')\n"
        "PY"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--allow-materialized-control", action="store_true")
    args = parser.parse_args()

    config = load_config()
    policies = config.get("policies", {})
    if args.dataset:
        datasets = [args.dataset]
    else:
        configured = policies.get("remote_smoke_datasets")
        datasets = list(configured) if isinstance(configured, list) and configured else [policies.get("remote_smoke_dataset", "rmd17")]
    epochs = int(args.epochs if args.epochs is not None else policies.get("remote_smoke_epochs", 1))
    max_samples = int(args.max_samples if args.max_samples is not None else policies.get("remote_smoke_max_samples", 1))
    mad10k_force_grad_samples = int(policies.get("mad10k_force_grad_smoke_samples", 16))
    gpu = args.gpu
    if gpu is not None and not re.fullmatch(r"\d+(,\d+)*", str(gpu).strip()):
        raise SystemExit(f"Invalid --gpu value: {gpu!r}. Expected numeric GPU id or comma-separated numeric ids.")

    unit_root = resolve_unit(args.unit, STAGING_RUNTIME_ROOT)
    status = load_implementation_status(unit_root)

    if args.unit != "base_unit":
        if status.get("implementation_state") == "implementation_needed" and not args.allow_materialized_control:
            raise SystemExit("remote smoke requires implementation review first; pass --allow-materialized-control only for the exact control replicate.")
        if not status.get("remote_synced"):
            raise SystemExit("remote smoke requires remote_synced == true")

    remote_python = f"{config.get('remote_conda_env', '').rstrip('/')}/bin/python"
    if not config.get("remote_conda_env"):
        raise SystemExit("Missing remote_conda_env in config.json.")

    remote_path = remote_unit_path(args.unit, config)
    smoke_log_remotes = {dataset: f"{remote_path}/outputs/smoke_{dataset}.log" for dataset in datasets}
    smoke_log_locals = {dataset: unit_root / "outputs" / f"smoke_{dataset}.log" for dataset in datasets}

    update_implementation_status(
        unit_root,
        implementation_state="remote_smoke_pending",
        remote_path=remote_path,
        last_actor="remote_smoke_unit.py",
    )
    sync_status_files_to_remote(unit_root, args.unit, config=config, include_meta=True)

    gpu_prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu).strip())} " if gpu is not None else ""
    smoke_commands = []
    for dataset in datasets:
        smoke_log = smoke_log_remotes[str(dataset)]
        command = (
            f"TEST_DATASET={shlex.quote(str(dataset))}\n"
            f"{gpu_prefix}$PY \"$UNIT/main.py\" --dataset \"$TEST_DATASET\" --epochs {epochs} --batch-size 1 --max-samples {max_samples} "
            f"> {shlex.quote(smoke_log)} 2>&1"
        )
        if str(dataset) == "mad10k":
            command += "\n" + mad10k_force_grad_sanity_command(
                remote_log=smoke_log,
                gpu_prefix=gpu_prefix,
                samples=mad10k_force_grad_samples,
            )
        smoke_commands.append(command)
    smoke_command_text = "\n".join(smoke_commands)
    script = f"""
set -euo pipefail
UNIT={shlex.quote(remote_path)}
PY={shlex.quote(remote_python)}
test -d "$UNIT"
test -f "$UNIT/main.py"
mkdir -p "$UNIT/outputs"
cd "$UNIT"
python3 - <<'PYCHK'
import json
import pathlib
root = pathlib.Path.cwd()
cfg = json.loads((root / 'config.json').read_text())
bench = (root / cfg['benchmark_root']).resolve()
if not bench.exists():
    raise SystemExit(f'benchmark_root missing: {{bench}}')
print(bench)
PYCHK
{smoke_command_text}
"""
    result = run_remote_bash(script, config=config)
    for dataset in datasets:
        rsync_from_remote(smoke_log_remotes[str(dataset)], smoke_log_locals[str(dataset)], config=config)

    if result.returncode == 0:
        update_implementation_status(
            unit_root,
            implementation_state="launch_ready",
            remote_smoke_passed=True,
            remote_path=remote_path,
            smoke_datasets=datasets,
            smoke_logs_local={dataset: str(path) for dataset, path in smoke_log_locals.items()},
            smoke_logs_remote=smoke_log_remotes,
            last_failure_class=None,
            last_actor="remote_smoke_unit.py",
        )
    else:
        update_implementation_status(
            unit_root,
            implementation_state="implemented",
            remote_smoke_passed=False,
            remote_path=remote_path,
            smoke_datasets=datasets,
            smoke_logs_local={dataset: str(path) for dataset, path in smoke_log_locals.items()},
            smoke_logs_remote=smoke_log_remotes,
            last_failure_class=f"remote_smoke_exit_{result.returncode}",
            last_actor="remote_smoke_unit.py",
        )
    sync_status_files_to_remote(unit_root, args.unit, config=config, include_meta=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
