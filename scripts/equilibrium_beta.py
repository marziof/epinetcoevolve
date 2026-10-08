#!/usr/bin/env python3
"""Plot the equilibrium infected fraction against beta.

Each input CSV contributes one estimate of I*: the mean of ``col1`` over the
last fraction of its trajectory. Files are grouped by initial infected probability (p1) and
transmission parameter (beta); repeated runs are shown as mean +/- standard
deviation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PARAMETER_PATTERN = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def parse_parameters(path: Path) -> dict[str, str]:
    """Read key=value parameters from the first comment line."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                return {key: value for key, value in PARAMETER_PATTERN.findall(line)}
            if line.strip():
                break
    return {}


def parse_fixed_parameters(values: list[str]) -> dict[str, float]:
    fixed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --fixed value '{value}'; expected PARAMETER=VALUE")
        key, raw_value = value.split("=", 1)
        try:
            fixed[key] = float(raw_value)
        except ValueError as error:
            raise ValueError(f"Invalid value in --fixed '{value}'") from error
    return fixed


def matches_fixed_parameters(params: dict[str, str], fixed: dict[str, float]) -> bool:
    for key, expected in fixed.items():
        try:
            actual = float(params[key])
        except (KeyError, ValueError):
            return False
        if actual != expected:
            return False
    return True


def estimate_file(path: Path, final_window: float | None, final_fraction: float,
                  min_t_max: float,
                  fixed: dict[str, float]) -> dict[str, object] | None:
    params = parse_parameters(path)
    if not params or "beta" not in params or "p1" not in params:
        return None
    if not matches_fixed_parameters(params, fixed):
        return None

    try:
        beta = float(params["beta"])
        p1 = float(params["p1"])
        t_max = float(params.get("t_max", "nan"))
    except ValueError:
        return None
    if pd.notna(t_max) and t_max < min_t_max:
        return None

    try:
        data = pd.read_csv(path, comment="#")
    except (OSError, pd.errors.ParserError):
        return None
    if "time" not in data.columns or "col1" not in data.columns:
        return None

    times = pd.to_numeric(data["time"], errors="coerce")
    infected = pd.to_numeric(data["col1"], errors="coerce")
    end_time = t_max if pd.notna(t_max) else times.max()
    window_start = (end_time - final_window if final_window is not None
                    else end_time * (1.0 - final_fraction))
    in_window = (times >= window_start - 1e-12) & (times <= end_time + 1e-12)
    infected = infected[in_window].dropna()
    if infected.empty:
        return None
    return {
        "file": path.name,
        "beta": beta,
        "p1": p1,
        "t_max": t_max,
        "window_start": window_start,
        "window_end": end_time,
        "n_samples": len(infected),
        "I_star": infected.mean(),
        "I_window_sd": infected.std(ddof=1) if len(infected) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot final-time average infected fraction I* versus beta."
    )
    parser.add_argument("--dir", type=Path, default=Path("output"),
                        help="Directory containing simulation CSV files (default: output)")
    parser.add_argument("--pattern", default="*.csv",
                        help="Input filename pattern (default: *.csv)")
    parser.add_argument("--final-window", type=float,
                        help="Fixed length of final simulation-time window (overrides --final-fraction)")
    parser.add_argument("--final-fraction", type=float, default=0.2,
                        help="Fraction of trajectory at the end to average (default: 0.2)")
    parser.add_argument("--min-t-max", type=float, default=10.0,
                        help="Ignore runs with t_max below this value (default: 10)")
    parser.add_argument("--fixed", action="append", default=[], metavar="PARAMETER=VALUE",
                        help="Keep only runs with this fixed parameter; repeat as needed")
    parser.add_argument("--out", type=Path,
                        help="Plot filename (default: DIR/equilibrium-beta.png)")
    parser.add_argument("--summary", type=Path,
                        help="Summary CSV filename (default: next to the plot)")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot after saving it")
    args = parser.parse_args()

    if args.final_window is not None and args.final_window <= 0:
        parser.error("--final-window must be positive")
    if not 0 < args.final_fraction <= 1:
        parser.error("--final-fraction must be in (0, 1]")
    try:
        fixed = parse_fixed_parameters(args.fixed)
    except ValueError as error:
        parser.error(str(error))
    files = sorted(args.dir.glob(args.pattern))
    estimates = [estimate_file(path, args.final_window, args.final_fraction,
                                args.min_t_max, fixed) for path in files]
    estimates = [row for row in estimates if row is not None]
    if not estimates:
        raise SystemExit(f"No usable simulation CSVs found in {args.dir}")

    runs = pd.DataFrame(estimates).sort_values(["p1", "beta", "file"])
    grouped = (runs.groupby(["p1", "beta"], as_index=False)
               .agg(I_star=("I_star", "mean"),
                    I_star_sd=("I_star", lambda values: values.std(ddof=1) if len(values) > 1 else 0.0),
                    replicates=("I_star", "size")))
    grouped = grouped.sort_values(["p1", "beta"])

    plot_path = args.out or args.dir / "equilibrium-beta.png"
    summary_path = args.summary or plot_path.with_name(f"{plot_path.stem}-summary.csv")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(summary_path, index=False)

    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    for p1, curve in grouped.groupby("p1", sort=True):
        curve = curve.sort_values("beta")
        axis.errorbar(curve["beta"], curve["I_star"], yerr=curve["I_star_sd"],
                      marker="o", capsize=3, linewidth=1.5, label=f"p1 = {p1:g}")
    axis.set_xlabel("Transmission parameter beta")
    axis.set_ylabel("Equilibrium infected fraction I*")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend(title="Initial condition")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"Used {len(runs)} runs across {len(grouped)} beta/initial-condition points")
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()