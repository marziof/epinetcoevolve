# netcoevolve

`netcoevolve` is a fast Rust implementation of the stochastic co-evolving network model introduced in ["Co-evolving vertex and edge dynamics in dense graphs"](https://arxiv.org/abs/2504.06493) by S. Athreya, F. den Hollander, and A. Röllin. It simulates a dense undirected graph whose vertices have one of two colours and writes colour, edge, and coloured-motif densities to CSV.

## Requirements

- Rust stable with Cargo (edition 2021).
- Python 3.10 or newer for the optional analysis tools.
- The Python packages in `scripts/requirements.txt` (`pandas` and `matplotlib`; these install NumPy as a dependency).
- Tkinter for `scripts/dispatch.py`; FFmpeg is optional for MP4 output from `scripts/analyse.py`.

Install the Python dependencies with:

```bash
python -m pip install -r scripts/requirements.txt
```

## Build

Build the optimized binary with:

```bash
cargo build --release
```

The binary is written to `target/release/netcoevolve`. A development build can be produced with `cargo build`, but it is much slower for simulations.

## Simulation model

Every unordered vertex pair belongs to one of four buckets:

| Bucket | Vertex colours | Edge state |
|---|---|---|
| `C0` | concordant (equal) | absent |
| `C1` | concordant (equal) | present |
| `D0` | discordant (different) | absent |
| `D1` | discordant (different) | present |

The simulator uses the Gillespie direct method. Edge transitions have the following per-pair rates:

| Transition | Per-pair rate |
|---|---|
| `C0 -> C1` | `rho * sc0` |
| `C1 -> C0` | `rho * sc1` |
| `D0 -> D1` | `rho * sd0` |
| `D1 -> D0` | `rho * sd1` |

Each `D1` edge triggers colour flips at total rate `2 * eta`; one of its endpoints is selected uniformly, so each endpoint is selected at rate `eta` through that edge. A colour flip reclassifies all pairs incident to that vertex.

## Run

Run with all defaults:

```bash
./target/release/netcoevolve
```

Example with custom parameters:

```bash
./target/release/netcoevolve \
  --n 500 --eta 1.0 --rho 2.0 \
  --sd0 0.0 --sd1 1.0 --sc0 0.0 --sc1 1.0 \
  --p1 0.5 --p00 0.5 --p01 0.5 --p11 0.5 \
  --sample_delta 0.01 --t_max 5.0 --seed 42
```

Use `./target/release/netcoevolve --help` to see the CLI help.

### Dynamics parameters

| Flag | Meaning | Default |
|---|---|---|
| `--n` | Number of vertices; must be at least 2 | `1000` |
| `--rho` | Common edge-event rate multiplier | `1.0` |
| `--eta` | Colour-flip rate per endpoint of each discordant present edge | `1.0` |
| `--beta` | Scaling shortcut: require `beta > 0`, set `rho = n`, and replace `eta` with `beta`; cannot be combined with `--rho` | not set |
| `--sd0` | Discordant absent-to-present multiplier | `0.7` |
| `--sd1` | Discordant present-to-absent multiplier | `2.0` |
| `--sc0` | Concordant absent-to-present multiplier | `1.5` |
| `--sc1` | Concordant present-to-absent multiplier | `0.3` |

### Initial state

Vertex colours are initialized independently. Conditional on those colours, edges are also initialized independently.

| Flag | Meaning | Default |
|---|---|---|
| `--p1` | Probability that a vertex initially has colour 1 | `0.5` |
| `--p00` | Initial edge probability between two colour-0 vertices | `0.5` |
| `--p01` | Initial edge probability between differently coloured vertices | `0.5` |
| `--p11` | Initial edge probability between two colour-1 vertices | `0.5` |

### Sampling and output

| Flag | Meaning | Default |
|---|---|---|
| `--sample_delta` | Interval between simulation-time sampling thresholds | `0.01` |
| `--t_max` | Simulation end time | `1.0` |
| `--seed` | Unsigned integer RNG seed, or `random` for a time-derived seed in `0..65535` | `42` |
| `--output` | Output CSV path | `output/simulation-<timestamp>.csv` |
| `--dump_adj` | Write an adjacency snapshot for every recorded sample | disabled |
| `--stop_at_polarisation` | Stop as soon as there are no discordant present edges (`D1` is empty) | disabled |

The CSV starts with comment lines containing the package version and all resolved parameters, followed by columns for:

- time, colour fractions, and present/absent edge densities by colour pattern;
- coloured triangle and 2-path homomorphism densities;
- coloured 3-path and 3-star homomorphism densities.

There is an initial sample at time 0. During a non-absorbing run, sampling is triggered as simulation time passes successive `sample_delta` thresholds; because events occur at random times, recorded times need not lie exactly on that grid. A final row labelled `t_max` is also written unless `--stop_at_polarisation` ends the run earlier.

With `--dump_adj`, an output such as `output/run.csv` gets a sibling directory named `output/run-adj/`. Each text snapshot contains the parameter header, the vertex-colour bit string, and the adjacency matrix in original vertex order.

If the total event rate becomes zero, the state is absorbing. The simulator writes the unchanged state at all remaining sampling times through `t_max` and marks the final progress message with `(absorbing)`. With `--stop_at_polarisation`, the stopping state is recorded and the final message is marked `(polarised)`.

## Python tools

The maintained package tools directly under `scripts/` are the following. Experiment-specific scans and one-off plotting scripts live in `scripts/adhoc/` and are not part of the package interface.

### `scripts/visualise.py`

Plot colour fractions, edge densities, and optional motif densities from a simulation CSV. If the CSV argument is omitted, the newest `simulation-*.csv` in the current directory or `output/` is used.

```bash
python scripts/visualise.py [csv_file] [options]
```

Useful options include:

- `--out FILE`, `--show`, `--split-panels`, `--dpi N`, and `--ratio W:H` for output control;
- `--triangles`, `--2paths`, `--3paths`, `--3stars`, or `--all` for motif panels;
- `--projections` for composition-based motif projections;
- `--conc-disc` for the concordant-minus-discordant edge panel;
- `--hide-non-edges`, `--no-partitions`, and `--show-parameters` for display control;
- `--pair PATTERN1 PATTERN2` to compare two specific motif patterns.

### `scripts/analyse.py`

Analyse adjacency snapshots produced by `--dump_adj`.

```bash
python scripts/analyse.py <subcommand> <snapshot-directory> [options]
```

Implemented subcommands are:

- `animate`: render an MP4 or GIF of the adjacency matrices;
- `diagnostics`: compute spectral, rank-1, and optional extended block metrics;
- `info`: summarize the snapshot set;
- `correlations`: compute vertex and edge correlations relative to a reference time.

Run `python scripts/analyse.py <subcommand> --help` for frame selection, vertex ordering, smoothing, and export options. The `frames`, `permutation`, and `verify` subcommands currently exist only as stubs.

### `scripts/polarisation.py`

Summarize polarisation across the simulation CSV files directly inside a directory:

```bash
python scripts/polarisation.py --dir output --k 10
```

`--k` is the number of final rows used for the stability test. The default output is `<dir>/polarisation.csv`; use `--out FILE` to change it.

### `scripts/dispatch.py`

Launch the Tkinter GUI for concurrent parameter sweeps:

```bash
python scripts/dispatch.py
```

The GUI configures the binary, output directory, worker count, repetitions, `n` values, an `eta` range, and fixed simulation parameters. It uses Unix PTYs and is intended for macOS or Linux.

## Article examples

A run similar to Figure 1 of the article can be produced with:

```bash
./target/release/netcoevolve \
  --n 1000 --eta 1.0 --rho 1.1 \
  --sc0 1.5 --sd0 0.7 --sc1 0.5 --sd1 2.0 \
  --sample_delta 0.005 --t_max 3.0 --seed 61 \
  --output output/figure-1.csv
python scripts/visualise.py output/figure-1.csv --out plot-1.png --split-panels
```

A polarising run similar to Figure 2 can be produced with:

```bash
./target/release/netcoevolve \
  --n 1000 --eta 1.0 --rho 2.0 \
  --sc0 0.0 --sd0 0.0 --sc1 1.0 --sd1 1.0 \
  --sample_delta 0.005 --t_max 3.0 --seed 17 \
  --output output/figure-2.csv
python scripts/visualise.py output/figure-2.csv --out plot-2.png --split-panels
```

## Performance and reproducibility

- Use the release build for meaningful performance; `Cargo.toml` enables optimization and link-time optimization.
- Motif statistics are relatively expensive. Increasing `--sample_delta` reduces sampling overhead at the cost of time resolution.
- The simulator uses `Xoshiro256++`. It prints the resolved seed and parameters and stores them in the CSV header.
- A run started with `--seed=random` is reproducible by reusing the resolved integer seed printed at startup.
