#!/usr/bin/env python3
"""Launch netcoevolve simulations from an interactive menu or the CLI.

The launcher keeps all generated files below the repository's ``output/``
directory. It does not require pandas or matplotlib.
"""
from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"

DEFAULTS: dict[str, Any] = {
    "n": 1000,
    "rho": 1.0,
    "eta": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
    "sd0": 0.05,
    "sd1": 0.9,
    "sc0": 0.9,
    "sc1": 0.05,
    "p1": 0.05,
    "p00": 0.5,
    "p01": 0.5,
    "p11": 0.5,
    "sample_delta": 0.01,
    "t_max": 1.0,
    "seed": "42",
    "diagnostic_fraction": 1.0,
    "burn_in": 10.0,
}

NUMERIC_FLAGS = (
    "n", "rho", "eta", "beta", "gamma", "sd0", "sd1", "sc0", "sc1",
    "p1", "p00", "p01", "p11", "sample_delta", "t_max",
    "diagnostic_fraction", "burn_in",
)
BOOL_FLAGS = ("dump_adj", "stop_at_polarisation", "densities_only")


# Edit this block for repeatable experiments, then run:
#     python scripts/menu.py --manual
# Available modes: "single", "closure", and "sweep".
MANUAL_MODE = "single"
# MANUAL_VALUES: dict[str, Any] = {
#     **DEFAULTS,
#     "n": 500,
#     "eta": 1.0,
#     "rho": 2.0,
#     "t_max": 5.0,
#     "seed": "42",
# }

MANUAL_VALUES: dict[str, Any] = {
    **DEFAULTS,
    "n": 1000,
    "rho": 1.0,
    "eta": 1.0,
    "beta": 3.0,
    "gamma": 1.0,
    "sd0": 0.05,
    "sd1": 1.0,
    "sc0": 0.7,
    "sc1": 0.1,
    "p1": 0.05,
    "p00": 0.5,
    "p01": 0.5,
    "p11": 0.5,
    "sample_delta": 0.05,
    "t_max": 100.0,
    "burn_in": 2.0,
    "diagnostic_fraction": 1.0,
    "seed": "42",
}

name = "slice5"
#S1 = slice 1

mode_string = f"{MANUAL_MODE}"
if MANUAL_MODE == "closure":
    mode_string += f"_diag_frac{MANUAL_VALUES['diagnostic_fraction']}"


MANUAL_REPEATS = 6
s_string = f"sd0{MANUAL_VALUES['sd0']}_d1{MANUAL_VALUES['sd1']}_c0{MANUAL_VALUES['sc0']}_c1{MANUAL_VALUES['sc1']}"
MANUAL_FOLDER = f"{name}_{mode_string}_n{MANUAL_VALUES['n']}_T{MANUAL_VALUES['t_max']}-runs_rho{MANUAL_VALUES['rho']}"
MANUAL_NAME = f"params_beta{MANUAL_VALUES['beta']}_{s_string}_p1{MANUAL_VALUES['p1']}_nsim{MANUAL_REPEATS}"
MANUAL_SWEEP = {
    "parameter": "beta",
    "values": [1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.75, 3.0, 4.0]#[500, 1000, 2000, 3000],#[1.12, 1.38],#[0.5, 1.0, 1.25, 1.5, 1.65, 1.85, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0],
}

#slice1
    # "sd0": 0.05,
    # "sd1": 0.95,
    # "sc0": 0.95,
    # "sc1": 0.05,
# slice2: all 0.5
# slice3
    # "sd0": 0.0,
    # "sd1": 1.0,
    # "sc0": 1.0,
    # "sc1": 0.0,
# slice 4
    # "sd0": 0.0,
    # "sd1": 1.0,
    # "sc0": 0.5,
    # "sc1": 0.0,
# slice 5
    # "sd0": 0.05,
    # "sd1": 1.0,
    # "sc0": 0.7,
    # "sc1": 0.1,
# slice 6
    # "sd0": 0.291,
    # "sd1": 1.424,
    # "sc0": 0.056,
    # "sc1": 0.412,
# slice 7
    # "sd0": 1.424,
    # "sd1": 0.291,
    # "sc0": 0.056,
    # "sc1": 0.412,
#s_c0 = 0.056, s_c1 = 0.412, s_d0 = 1.424, s_d1 = 0.291

def find_binary(requested: str | None, build: bool) -> Path:
    candidates = [
        Path(requested).expanduser() if requested else None,
        PROJECT_ROOT / "target" / "release" / "netcoevolve",
        PROJECT_ROOT / "target" / "debug" / "netcoevolve",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    if build:
        subprocess.run(["cargo", "build", "--release"], cwd=PROJECT_ROOT, check=True)
        release_binary = PROJECT_ROOT / "target" / "release" / "netcoevolve"
        if release_binary.is_file() and os.access(release_binary, os.X_OK):
            return release_binary
    raise FileNotFoundError(
        "No executable found. Run `cargo build --release` or pass --build."
    )


def output_directory(folder: str) -> Path:
    candidate = (OUTPUT_ROOT / folder).resolve()
    try:
        candidate.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("--folder must name a directory inside the repository output/ directory") from error
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def output_path(folder: str, name: str | None) -> Path:
    directory = output_directory(folder)
    stem = name or f"simulation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not stem.endswith(".csv"):
        stem += ".csv"
    path = directory / stem
    counter = 1
    while path.exists():
        path = directory / f"{Path(stem).stem}-{counter:04d}.csv"
        counter += 1
    return path


def build_command(binary: Path, values: dict[str, Any], csv_path: Path) -> list[str]:
    command = [str(binary)]
    for key in NUMERIC_FLAGS:
        command.extend([f"--{key}", str(values[key])])
    command.extend(["--seed", str(values["seed"]), "--output", str(csv_path)])
    for key in BOOL_FLAGS:
        if values.get(key, False):
            command.append(f"--{key}")
    return command


def run(values: dict[str, Any], binary_arg: str | None, folder: str,
        name: str | None, build: bool, dry_run: bool) -> int:
    binary = find_binary(binary_arg, build)
    csv_path = output_path(folder, name)
    command = build_command(binary, values, csv_path)
    print("Launching:")
    print("  " + shlex.join(command))
    if dry_run:
        return 0
    environment = os.environ.copy()
    environment.setdefault("NO_COLOR", "1")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
    if completed.returncode == 0:
        print(f"Output: {csv_path}")
    return completed.returncode


def run_manual(binary_arg: str | None, build: bool, dry_run: bool) -> int:
    """Run the experiment selected in the editable MANUAL_* section."""
    if MANUAL_MODE not in {"single", "closure", "sweep"}:
        raise ValueError("MANUAL_MODE must be single, closure, or sweep")

    values = dict(DEFAULTS)
    values.update(MANUAL_VALUES)
    repeats = int(MANUAL_REPEATS)
    if repeats < 1:
        raise ValueError("MANUAL_REPEATS must be at least 1")
    if MANUAL_MODE == "closure":
        # Full statistics contain drift_closed and drift_error. These are not
        # written when densities_only is enabled.
        values["densities_only"] = False

    if MANUAL_MODE != "sweep":
        for repeat in range(repeats):
            task_values = dict(values)
            task_values["seed"] = next_seed(values["seed"], repeat)
            name = f"{MANUAL_NAME}-r{repeat + 1}" if repeats > 1 else MANUAL_NAME
            result = run(task_values, binary_arg, MANUAL_FOLDER, name,
                         build, dry_run)
            if result != 0:
                return result
        return 0

    parameter = MANUAL_SWEEP["parameter"]
    sweep_values = MANUAL_SWEEP["values"]
    if parameter not in NUMERIC_FLAGS:
        raise ValueError("MANUAL_SWEEP['parameter'] must be a numeric simulation flag")

    for value, repeat in itertools.product(sweep_values, range(repeats)):
        task_values = dict(values)
        task_values[parameter] = value
        task_values["seed"] = next_seed(values["seed"], repeat)
        name = f"{MANUAL_NAME}__{parameter}{value:g}-r{repeat + 1}"
        result = run(task_values, binary_arg, MANUAL_FOLDER, name,
                     build, dry_run)
        if result != 0:
            return result
    return 0


def next_seed(seed: Any, offset: int) -> str:
    """Return deterministic seeds for numeric bases; preserve ``random``."""
    seed_text = str(seed)
    if seed_text.isdigit():
        return str(int(seed_text) + offset)
    return seed_text


def prompt(label: str, default: Any, converter: type = str) -> Any:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return converter(raw)
        except ValueError:
            print(f"Please enter a valid {converter.__name__}.")


def interactive(binary_arg: str | None, build: bool, dry_run: bool) -> int:
    print("netcoevolve simulation menu")
    print("1. Run one simulation")
    print("2. Show Rust help")
    print("3. Exit")
    choice = input("Select an option [1]: ").strip() or "1"
    if choice == "3":
        return 0
    binary = find_binary(binary_arg, build)
    if choice == "2":
        return subprocess.run([str(binary), "--help"], cwd=PROJECT_ROOT).returncode
    if choice != "1":
        print("Unknown option.", file=sys.stderr)
        return 2

    values = dict(DEFAULTS)
    converters = {"n": int, "seed": str}
    for key in DEFAULTS:
        values[key] = prompt(key, DEFAULTS[key], converters.get(key, float))
    values["dump_adj"] = prompt("dump_adj (y/n)", "n") .lower() in {"y", "yes"}
    values["stop_at_polarisation"] = prompt("stop_at_polarisation (y/n)", "n").lower() in {"y", "yes"}
    values["densities_only"] = prompt("densities_only (y/n)", "n").lower() in {"y", "yes"}
    folder = prompt("output folder inside output/", "runs")
    name = prompt("file name (blank for timestamp)", "") or None
    return run(values, binary_arg, folder, name, build, dry_run)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", action="store_true", help="run non-interactively instead of opening the menu")
    result.add_argument("--manual", action="store_true", help="run the editable MANUAL_* experiment configuration")
    result.add_argument("--binary", help="path to the compiled netcoevolve executable")
    result.add_argument("--build", action="store_true", help="build the release executable if needed")
    result.add_argument("--dry-run", action="store_true", help="print the command without running it")
    result.add_argument("--folder", default="runs", help="subfolder below output/ (default: runs)")
    result.add_argument("--name", help="CSV filename (default: timestamped name)")
    for key in NUMERIC_FLAGS:
        result.add_argument(f"--{key}", type=int if key == "n" else float, default=DEFAULTS[key])
    result.add_argument("--seed", default=DEFAULTS["seed"])
    for key in BOOL_FLAGS:
        result.add_argument(f"--{key}", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.manual:
        try:
            return run_manual(args.binary, args.build, args.dry_run)
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
    if not args.run:
        try:
            return interactive(args.binary, args.build, args.dry_run)
        except (FileNotFoundError, ValueError, KeyboardInterrupt) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
    values = {key: getattr(args, key) for key in NUMERIC_FLAGS + ("seed",) + BOOL_FLAGS}
    try:
        return run(values, args.binary, args.folder, args.name, args.build, args.dry_run)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())