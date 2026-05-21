from __future__ import annotations

import argparse
import subprocess

from runtime_common import RUNTIME_ROOT, load_config, local_python_path, resolve_unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--dataset", default="rmd17")
    args = parser.parse_args()

    config = load_config()
    local_python = local_python_path(config)
    unit_root = resolve_unit(args.unit, RUNTIME_ROOT)

    cmd = [
        local_python,
        "-c",
        (
            "import sys, pathlib; "
            "print('which python=', sys.executable); "
            "print('python -V=', sys.version.split()[0]); "
            "print('sys.executable=', sys.executable); "
            "import torch; print('torch=', torch.__version__); "
            "root = pathlib.Path(r'%s'); "
            "print('unit_root=', root)"
        ) % str(unit_root),
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
