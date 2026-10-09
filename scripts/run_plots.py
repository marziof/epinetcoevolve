#!/usr/bin/env python3
"""Create plots from simulation CSV files produced by ``menu.py``.

Edit the PLOT_* block for repeatable plotting, then run
``python scripts/run_plots.py --manual``. The script also supports an
interactive menu and command-line options.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"

# Edit this block for repeatable plots.
PLOT_MODE = "errors"  # single, multiple, errors, or bifurcation
PLOT_FOLDER = "S2_closure_diag_frac1.0_n1000_T100.0-runs_rho0.1"  # "manual-runs"
PLOT_PATTERN = "*.csv" #"*_nsim10-r*.csv"
PLOT_COMPONENT = "both"  # both, vertex, or edge (single/multiple modes)
PLOT_OUTPUT = f"plots/{PLOT_MODE}_{PLOT_FOLDER}_{PLOT_COMPONENT}.png"
PLOT_PARAMETER = "beta"  # parameter used by bifurcation mode
PLOT_MULTIPLE_PARAMETER_VALUE: float | None = 2.2  # e.g. 1.15; None keeps all matching values
PLOT_GROUP_BY = "p1"  # separate bifurcation curves by this fixed parameter
PLOT_PARAMETER_RANGE: tuple[float | None, float | None] = (1.4, 1.6)#(None, None) #(, 2)#(None, None)  # e.g. (0.5, 1.5)
PLOT_P1_VALUES: list[float] | None = [0.05, 0.95]  # e.g. [0.05], [0.05, 0.95], or None for all
PLOT_FINAL_WINDOW = 1.0
PLOT_SHOW = False
PLOT_ODE = True
PLOT_ODE_METHOD = "DOP853"
PLOT_TITLE_SIZE = 16
PLOT_LABEL_SIZE = 14
PLOT_TICK_SIZE = 12
PLOT_LEGEND_SIZE = 11
# if values stop at a time less than 100, pad with the last value to the end of the longest run for mean/spread calculations
PAD_TO_END = False
def pad_to_end_func(data: pd.DataFrame, column: str, end_time: float) -> pd.DataFrame:
    if column not in data or data.empty:
        return data
    last_time = float(data["time"].max())
    if last_time >= end_time:
        return data
    last_value = float(data[column].iloc[-1])
    padding = pd.DataFrame({"time": [last_time, end_time], column: [last_value, last_value]})
    return pd.concat([data, padding], ignore_index=True)


PARAMETER_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
ERROR_COLUMNS = (
    "drift_error",
    "identity_si_error",
    "identity_ss_error",
    "n_var_error",
    "n_cov_error",
)
RELATIVE_ERROR_COLUMNS = (
    "n_var_relative_error",
    "drift_relative_error",
)


def resolve_folder(folder: str) -> Path:
    candidate = (OUTPUT_ROOT / folder).resolve()
    try:
        candidate.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("--folder must name a directory inside output/") from error
    if not candidate.is_dir():
        raise FileNotFoundError(f"Output folder not found: {candidate}")
    return candidate


def parse_parameters(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                return dict(PARAMETER_RE.findall(line))
            if line.strip():
                break
    return {}


def load_csv(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    data = pd.read_csv(path, comment="#")
    if "time" not in data.columns:
        raise ValueError(f"Missing time column in {path.name}")
    for column in data.columns:
        if column != "time":
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if {"n_var_s", "n_var_reference"}.issubset(data.columns):
        data["n_var_error"] = data["n_var_s"] - data["n_var_reference"]
    if {"n_cov_s", "n_cov_reference"}.issubset(data.columns):
        data["n_cov_error"] = data["n_cov_s"] - data["n_cov_reference"]
    if {"n_var_error", "n_var_reference"}.issubset(data.columns):
        reference = data["n_var_reference"].abs()
        data["n_var_relative_error"] = np.where(
            reference > 1e-12,
            data["n_var_error"] / reference,
            np.nan,
        )
    if {"drift_error", "drift_closed"}.issubset(data.columns):
        reference = data["drift_closed"].abs()
        data["drift_relative_error"] = np.where(
            reference > 1e-12,
            data["drift_error"] / reference,
            np.nan,
        )
    if {"p_d", "p_c0", "p_c1"}.issubset(data.columns):
        data["edge_density"] = data["p_d"] + data["p_c0"] + data["p_c1"]
    elif {"e00", "e01", "e11"}.issubset(data.columns):
        data["edge_density"] = data["e00"] + data["e01"] + data["e11"]
    return data, parse_parameters(path)


def output_path(folder: Path, requested: str | None, default_stem: str) -> Path:
    path = (OUTPUT_ROOT / requested).resolve() if requested else folder / f"{default_stem}.png"
    try:
        path.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("--out must point inside output/") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def vertex_columns(data: pd.DataFrame) -> list[tuple[str, str]]:
    columns = []
    if "q" in data:
        columns.append(("q", "q: vertex-1 density"))
    elif "col1" in data:
        columns.append(("col1", "vertex-1 density"))
    if "col0" in data:
        columns.append(("col0", "vertex-0 density"))
    return columns


def edge_columns(data: pd.DataFrame) -> list[tuple[str, str]]:
    candidates = [
        ("edge_density", "total edge density"),
        ("p_d", "discordant edges"),
        ("concordant_present", "concordant edges"),
        ("p_c0", "0-0 edges"),
        ("p_c1", "1-1 edges"),
    ]
    return [(column, label) for column, label in candidates if column in data]


def style_axis(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", labelsize=PLOT_TICK_SIZE)


def ode_parameters(params: dict[str, str]) -> dict[str, float]:
    names = ("eta", "rho", "beta", "gamma", "sd0", "sd1", "sc0", "sc1")
    try:
        return {name: float(params[name]) for name in names}
    except (KeyError, ValueError) as error:
        raise ValueError(f"Missing ODE parameter in CSV header: {error}") from error


def ode_columns(data: pd.DataFrame) -> dict[str, str]:
    candidates = {
        "q": ("q", "col1"),
        "pd": ("p_d", "e01"),
        "pc0": ("p_c0", "e00"),
        "pc1": ("p_c1", "e11"),
    }
    result = {}
    for name, choices in candidates.items():
        column = next((choice for choice in choices if choice in data), None)
        if column is None:
            raise ValueError(f"CSV lacks the ODE density column for {name}")
        result[name] = column
    return result


def solve_ode(data: pd.DataFrame, params: dict[str, str], method: str = "DOP853") -> tuple[np.ndarray, np.ndarray]:
    """Solve the four-density ODE at the CSV's sample times."""
    columns = ode_columns(data)
    times = data["time"].dropna().to_numpy(dtype=float)
    times = np.unique(times)
    if len(times) < 2:
        raise ValueError("Need at least two sample times for ODE comparison")
    values = data.iloc[0]
    initial = np.array([values[columns[name]] for name in ("q", "pd", "pc0", "pc1")], dtype=float)
    if not np.all(np.isfinite(initial)):
        raise ValueError("Initial density row contains non-finite values")
    rates = ode_parameters(params)

    def rhs(_time: float, state: np.ndarray) -> list[float]:
        q, pd, pc0, pc1 = state
        denominator = max(1.0 - q, 1e-12)
        eta = rates["eta"]
        rho = rates["rho"]
        beta = rates["beta"]
        gamma = rates["gamma"]
        sd0, sd1 = rates["sd0"], rates["sd1"]
        sc0, sc1 = rates["sc0"], rates["sc1"]
        return [
            eta * (beta * pd / 2.0 - gamma * q),
            eta * (beta * (pd * pc0 / denominator - pd * pd / (2.0 * denominator))
                   + gamma * (2.0 * pc1 - pd))
            + rho * (2.0 * q * (1.0 - q) * sd0 - (sd0 + sd1) * pd),
            eta * (gamma * pd - beta * pd * pc0 / denominator)
            + rho * (sc0 * (1.0 - q) ** 2 - (sc0 + sc1) * pc0),
            eta * (beta * pd * pd / (2.0 * denominator) - 2.0 * gamma * pc1)
            + rho * (sc0 * q ** 2 - (sc0 + sc1) * pc1),
        ]

    solution = solve_ivp(rhs, (times[0], times[-1]), initial, t_eval=times,
                         method=method, rtol=1e-8, atol=1e-10)
    if not solution.success:
        raise ValueError(f"ODE solver failed: {solution.message}")
    return solution.t, solution.y


def plot_ode_sections(axes: np.ndarray, data: pd.DataFrame, params: dict[str, str],
                      sections: list[str], label_suffix: str = "") -> None:
    time, solution = solve_ode(data, params, PLOT_ODE_METHOD)
    density_columns = {
        "vertex": [(0, "q ODE")],
        "edge": [(1, "p_d ODE"), (2, "p_c0 ODE"), (3, "p_c1 ODE")],
    }
    for index, section in enumerate(sections):
        axis = axes[index, 0]
        for state_index, label in density_columns[section]:
            axis.plot(time, solution[state_index], linestyle="--", linewidth=1.8,
                      label=f"{label}{label_suffix}")


def plot_trajectory(data: pd.DataFrame, title: str, path: Path, show: bool,
                    component: str = "both", params: dict[str, str] | None = None) -> None:
    if component not in {"both", "vertex", "edge"}:
        raise ValueError("component must be both, vertex, or edge")
    if not vertex_columns(data) and not edge_columns(data):
        raise ValueError("CSV contains no recognised vertex or edge density columns")
    sections = [] if component == "both" else [component]
    if component == "both":
        sections = ["vertex", "edge"]
    figure, axes = plt.subplots(len(sections), 1, figsize=(9, 4 * len(sections)),
                                sharex=True, squeeze=False)
    for index, section in enumerate(sections):
        axis = axes[index, 0]
        columns = vertex_columns(data) if section == "vertex" else edge_columns(data)
        for column, label in columns:
            axis.plot(data["time"], data[column], label=label)
        axis.set_title("Vertex densities" if section == "vertex" else "Edge densities",
                       fontsize=PLOT_TITLE_SIZE)
        axis.legend(fontsize=PLOT_LEGEND_SIZE)
    if PLOT_ODE:
        if params is None:
            raise ValueError("ODE overlay requires CSV parameters")
        plot_ode_sections(axes, data, params, sections)
        for axis in axes[:, 0]:
            axis.legend(fontsize=PLOT_LEGEND_SIZE)
    axes[-1, 0].set_xlabel("time", fontsize=PLOT_LABEL_SIZE)
    for axis in axes[:, 0]:
        style_axis(axis)
        axis.grid(alpha=0.3)
    figure.suptitle(title, fontsize=PLOT_TITLE_SIZE)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close(figure)


def aggregate_runs(
    runs: list[tuple[Path, pd.DataFrame, dict[str, str]]],
    column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Align trajectories, padding early-stopped runs with their last value."""
    usable = []
    for _, data, _ in runs:
        if column not in data:
            continue
        values = data[["time", column]].dropna().sort_values("time")
        if len(values) >= 2:
            usable.append(values.drop_duplicates("time"))
    if not usable:
        return None

    start = max(float(values["time"].min()) for values in usable)
    longest = max(usable, key=lambda values: float(values["time"].max()))
    grid = longest["time"].to_numpy(dtype=float)
    grid = grid[grid >= start]
    if len(grid) == 0:
        return None

    aligned = np.vstack([
        np.interp(grid, values["time"].to_numpy(dtype=float), values[column].to_numpy(dtype=float))
        for values in usable
    ])
    mean = aligned.mean(axis=0)
    spread = aligned.std(axis=0, ddof=1) if len(aligned) > 1 else np.zeros_like(mean)
    return grid, mean, spread


def plot_mean_with_spread(axis: plt.Axes, runs: list[tuple[Path, pd.DataFrame, dict[str, str]]],
                          columns: list[tuple[str, str]], label_suffix: str = "") -> None:
    for column, label in columns:
        aggregate = aggregate_runs(runs, column)
        if aggregate is None:
            continue
        time, mean, spread = aggregate
        line = axis.plot(time, mean, label=f"{label}{label_suffix}")[0]
        axis.fill_between(time, mean - spread, mean + spread,
                          color=line.get_color(), alpha=0.18)


def plot_multiple(
    runs: Iterable[tuple[Path, pd.DataFrame, dict[str, str]]],
    path: Path,
    show: bool,
    p1_values: list[float] | None = None,
    pad_to_end: bool = True,
    component: str = "both",
) -> None:
    # If pad_to_end is True, pad each run's data to the end of the longest run
    if pad_to_end:
        longest_run = max(runs, key=lambda item: float(item[1]["time"].max()))
        end_time = float(longest_run[1]["time"].max())
        padded_runs = []
        for source, data, params in runs:
            padded_data = data.copy()
            for column in padded_data.columns:
                if column != "time":
                    padded_data = pad_to_end_func(padded_data, column, end_time)
            padded_runs.append((source, padded_data, params))
        runs = padded_runs

    if component not in {"both", "vertex", "edge"}:
        raise ValueError("component must be both, vertex, or edge")
    runs = list(runs)
    selected: list[tuple[float, list[tuple[Path, pd.DataFrame, dict[str, str]]]]] = []
    for p1 in sorted({float(params["p1"]) for _, _, params in runs if "p1" in params}):
        if p1_values is not None and not any(abs(p1 - value) < 1e-9 for value in p1_values):
            continue
        group = [(source, data, params) for source, data, params in runs
                 if "p1" in params and abs(float(params["p1"]) - p1) < 1e-9]
        if group:
            selected.append((p1, group))
    if not selected:
        raise ValueError("No runs matched the requested p1 values")
    if sum(len(group) for _, group in selected) < 2:
        raise ValueError("Multiple-run mode requires at least two CSV files")

    sections = ["vertex", "edge"] if component == "both" else [component]
    figure, axes = plt.subplots(len(sections), 1, figsize=(9, 4 * len(sections)),
                                sharex=True, squeeze=False)
    for p1, group in selected:
        suffix = f" (p1={p1:g}, n={len(group)})"
        for index, section in enumerate(sections):
            columns = vertex_columns(group[0][1]) if section == "vertex" else edge_columns(group[0][1])
            plot_mean_with_spread(axes[index, 0], group, columns, suffix)
        if PLOT_ODE:
            longest_run = max(group, key=lambda item: float(item[1]["time"].max()))
            mean_initial = longest_run[1].copy()
            for column in mean_initial.columns:
                if column != "time":
                    mean_initial[column] = pd.to_numeric(mean_initial[column], errors="coerce")
            for column in mean_initial.columns:
                if column != "time":
                    values = [run_data.iloc[0][column] for _, run_data, _ in group if column in run_data]
                    if values:
                        mean_initial.loc[mean_initial.index[0], column] = np.mean(values)
            plot_ode_sections(axes, mean_initial, group[0][2], sections, f" (p1={p1:g})")
    for index, section in enumerate(sections):
        axes[index, 0].set_title(
            ("Vertex" if section == "vertex" else "Edge") + " densities across runs",
            fontsize=PLOT_TITLE_SIZE,
        )
    axes[-1, 0].set_xlabel("time", fontsize=PLOT_LABEL_SIZE)
    for axis in axes[:, 0]:
        style_axis(axis)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=PLOT_LEGEND_SIZE)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close(figure)


def plot_errors(data: pd.DataFrame, title: str, path: Path, show: bool) -> None:
    # plot aboslute error - need to add np.abs() to the error columns to avoid negative values
    available = [(column, column.replace("_", " ")) for column in ERROR_COLUMNS if column in data]
    relative = [(column, column.replace("_", " ")) for column in RELATIVE_ERROR_COLUMNS if column in data]
    if not available and not relative:
        raise ValueError("No closure-error columns found in this CSV")
    rows = 2 if relative else 1
    figure, axes = plt.subplots(rows, 1, figsize=(9, 4.5 * rows), sharex=True,
                                squeeze=False)
    absolute_axis = axes[0, 0]
    for column, label in available:
        #absolute_axis.plot(data["time"], data[column], label=label)
        absolute_axis.plot(data["time"], np.abs(data[column]), label=label)
    absolute_axis.axhline(0.0, color="black", linewidth=0.7)
    absolute_axis.set_title(f"{title}: absolute errors", fontsize=PLOT_TITLE_SIZE)
    absolute_axis.set_ylabel("absolute error", fontsize=PLOT_LABEL_SIZE)
    style_axis(absolute_axis)
    absolute_axis.grid(alpha=0.3)
    if available:
        absolute_axis.legend(fontsize=PLOT_LEGEND_SIZE)
    if relative:
        relative_axis = axes[1, 0]
        for column, label in relative:
            relative_axis.plot(data["time"], data[column], label=label)
        relative_axis.axhline(0.0, color="black", linewidth=0.7)
        relative_axis.set_title("Relative errors", fontsize=PLOT_TITLE_SIZE)
        relative_axis.set_ylabel("error / |reference|", fontsize=PLOT_LABEL_SIZE)
        style_axis(relative_axis)
        relative_axis.legend(fontsize=PLOT_LEGEND_SIZE)
        relative_axis.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time", fontsize=PLOT_LABEL_SIZE)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close(figure)


def final_value(data: pd.DataFrame, column: str, window: float) -> float:
    valid = data[["time", column]].dropna()
    if valid.empty:
        return float("nan")
    end = float(valid["time"].max())
    selected = valid[valid["time"] >= end - window]
    return float(selected[column].mean())


def plot_bifurcation(
    runs: Iterable[tuple[Path, pd.DataFrame, dict[str, str]]],
    parameter: str,
    group_by: str,
    window: float,
    path: Path,
    show: bool,
    parameter_range: tuple[float | None, float | None] = (None, None),
) -> None:
    rows = []
    for source, data, params in runs:
        try:
            parameter_value = float(params[parameter])
            group_value = float(params[group_by])
        except (KeyError, ValueError):
            print(f"Skipping {source.name}: no numeric {parameter} or {group_by} in header", file=sys.stderr)
            continue
        if ((parameter_range[0] is not None and parameter_value < parameter_range[0])
                or (parameter_range[1] is not None and parameter_value > parameter_range[1])):
            continue
        vertex_column = "q" if "q" in data else "col1"
        observables = [(vertex_column, "vertex-1 density"),
                       ("edge_density", "total edge density"),
                       ("p_d", "discordant edges (p_d)"),
                       ("p_c0", "0-0 edges (p_c0)"),
                       ("p_c1", "1-1 edges (p_c1)")]
        for column, label in observables:
            if column in data:
                rows.append({"parameter": parameter_value, "observable": label, "column": column,
                             group_by: group_value, "value": final_value(data, column, window),
                             "file": source.name})
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError(f"No usable runs found for bifurcation parameter '{parameter}'")
    grouped = (summary.groupby(["parameter", group_by, "observable"], as_index=False)
               .agg(value=("value", "mean"), spread=("value", "std"), runs=("value", "count")))
    grouped["spread"] = grouped["spread"].fillna(0.0)
    grouped.to_csv(path.with_suffix(".csv"), index=False)
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for (observable, group_value), curve in grouped.groupby(["observable", group_by]):
        axes[0 if observable == "vertex-1 density" else 1].errorbar(
            curve["parameter"], curve["value"], yerr=curve["spread"], marker="o", capsize=3,
            label=f"{observable}, {group_by}={group_value:g}",
        )
    axes[0].set_title(f"Final vertex density vs {parameter}", fontsize=PLOT_TITLE_SIZE)
    axes[1].set_title(f"Final edge densities vs {parameter}", fontsize=PLOT_TITLE_SIZE)
    axes[1].set_xlabel(parameter, fontsize=PLOT_LABEL_SIZE)
    for axis in axes:
        style_axis(axis)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=PLOT_LEGEND_SIZE)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close(figure)


def load_runs(folder: Path, pattern: str) -> list[tuple[Path, pd.DataFrame, dict[str, str]]]:
    runs = []
    for path in sorted(folder.glob(pattern)):
        try:
            data, params = load_csv(path)
            runs.append((path, data, params))
        except (OSError, ValueError, pd.errors.ParserError) as error:
            print(f"Skipping {path.name}: {error}", file=sys.stderr)
    if not runs:
        raise FileNotFoundError(f"No readable CSV files matching {pattern!r} in {folder}")
    return runs


def filter_runs_by_parameter(
    runs: list[tuple[Path, pd.DataFrame, dict[str, str]]],
    parameter: str,
    value: float | None,
) -> list[tuple[Path, pd.DataFrame, dict[str, str]]]:
    if value is None:
        return runs
    selected = []
    for source, data, params in runs:
        try:
            actual = float(params[parameter])
        except (KeyError, ValueError):
            continue
        if abs(actual - value) <= 1e-9:
            selected.append((source, data, params))
    if not selected:
        raise ValueError(f"No runs found with {parameter}={value:g}")
    return selected


def run(mode: str, folder_name: str, pattern: str, output: str | None,
    parameter: str, group_by: str, window: float, show: bool,
    p1_values: list[float] | None = None,
    pad_to_end: bool = True,
    parameter_range: tuple[float | None, float | None] = (None, None),
    component: str = "both",
    multiple_parameter_value: float | None = None) -> Path:
    if mode not in {"single", "multiple", "errors", "bifurcation"}:
        raise ValueError("mode must be single, multiple, errors, or bifurcation")
    if parameter_range[0] is not None and parameter_range[1] is not None and parameter_range[0] > parameter_range[1]:
        raise ValueError("parameter minimum cannot be greater than parameter maximum")
    folder = resolve_folder(folder_name)
    runs = load_runs(folder, pattern)
    if mode == "single":
        source, data, _ = runs[0]
        path = output_path(folder, output, f"{source.stem}-plot")
        plot_trajectory(data, source.stem, path, show, component, runs[0][2])
    elif mode == "multiple":
        runs = filter_runs_by_parameter(runs, parameter, multiple_parameter_value)
        path = output_path(folder, output, "multiple-runs")
        plot_multiple(runs, path, show, p1_values, pad_to_end, component)
    elif mode == "errors":
        source, data, _ = runs[0]
        path = output_path(folder, output, f"{source.stem}-errors")
        plot_errors(data, source.stem, path, show)
    else:
        path = output_path(folder, output, f"bifurcation-{parameter}")
        plot_bifurcation(runs, parameter, group_by, window, path, show, parameter_range)
    print(f"Wrote {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", action="store_true", help="use the editable PLOT_* configuration")
    parser.add_argument("--mode", choices=["single", "multiple", "errors", "bifurcation"])
    parser.add_argument("--folder", default=PLOT_FOLDER)
    parser.add_argument("--pattern", default=PLOT_PATTERN)
    parser.add_argument("--out", help="PNG path inside output/")
    parser.add_argument("--parameter", default=PLOT_PARAMETER)
    parser.add_argument("--parameter-value", type=float, default=None,
                        help="select one parameter value in multiple mode, e.g. beta=1.15")
    parser.add_argument("--parameter-min", type=float, default=None,
                        help="minimum bifurcation parameter value")
    parser.add_argument("--parameter-max", type=float, default=None,
                        help="maximum bifurcation parameter value")
    parser.add_argument("--group-by", default=PLOT_GROUP_BY,
                        help="fixed header parameter used for separate bifurcation curves (default: p1)")
    parser.add_argument("--p1", type=float, nargs="+", default=None,
                        help="select one or more p1 groups in multiple mode")
    parser.add_argument("--component", choices=["both", "vertex", "edge"],
                        default=PLOT_COMPONENT,
                        help="plot vertex densities, edge densities, or both")
    parser.add_argument("--ode", action="store_true",
                        help="overlay the four-density ODE on single/multiple plots")
    parser.add_argument("--final-window", type=float, default=PLOT_FINAL_WINDOW)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def interactive() -> int:
    mode = input("Plot type [single/multiple/errors/bifurcation] (single): ").strip() or "single"
    folder = input(f"Folder inside output/ [{PLOT_FOLDER}]: ").strip() or PLOT_FOLDER
    parameter = input(f"Bifurcation parameter [{PLOT_PARAMETER}]: ").strip() or PLOT_PARAMETER
    group_by = input(f"Separate curves by [{PLOT_GROUP_BY}]: ").strip() or PLOT_GROUP_BY
    run(mode, folder, PLOT_PATTERN, None, parameter, group_by, PLOT_FINAL_WINDOW, PLOT_SHOW,
        PLOT_P1_VALUES, PAD_TO_END)
    return 0


def main() -> int:
    global PLOT_ODE
    args = parse_args()
    PLOT_ODE = PLOT_ODE or args.ode
    try:
        if args.manual:
            run(PLOT_MODE, PLOT_FOLDER, PLOT_PATTERN, PLOT_OUTPUT,
                PLOT_PARAMETER, PLOT_GROUP_BY, PLOT_FINAL_WINDOW, PLOT_SHOW,
                PLOT_P1_VALUES, PAD_TO_END, PLOT_PARAMETER_RANGE, PLOT_COMPONENT,
                PLOT_MULTIPLE_PARAMETER_VALUE)
        elif args.mode:
            run(args.mode, args.folder, args.pattern, args.out,
                args.parameter, args.group_by, args.final_window, args.show,
                args.p1, (args.parameter_min, args.parameter_max), args.component,
                args.parameter_value)
        else:
            return interactive()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())