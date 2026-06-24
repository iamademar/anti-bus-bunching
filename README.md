# Anti-Bunching Copilot

A streaming machine-learning experiment that predicts, online, when a bus is about to **bunch** with
the bus ahead and would nudge the driver to ease off so the gap recovers. It is trained and evaluated
prequentially (test-then-train) on the **Salvador Urban Network Transportation (SUNT)** Origin–
Destination data for a few scoped Salvador routes.

The repository reproduces the full pipeline behind the accompanying report, in four stages:

| Stage | Folder | Entry point | Reads | Writes |
|-------|--------|-------------|-------|--------|
| 1. Preprocessing | `Code/Preprocessing/` | `prepare_data.py` | raw SUNT OD parquet | `Code/data/processed/features.csv` |
| 2. Experiment | `Code/Experiment/` | `run_experiment.py` | `features.csv` | `Code/outputs/predictions.csv`, `metrics.json`, `nudges.json`, `drift_events.json` |
| 3. Simulation | `Code/Simulation/` | `run_simulation.py`, `make_beforeafter.py` | `Code/outputs/predictions.csv` | `Code/outputs/simulation_results.json`, `beforeafter_*` |
| 4. Report | `Report/` | (LaTeX) | (none) | `main.pdf` |

All code lives under `Code/`; the shared package is `Code/src/`, and every stage script imports from
it. Run the stage commands from inside `Code/`.

## Repository layout

```
anti-bus-bunching/
├── README.md                  # this file
├── LICENSE                    # MIT (code). SUNT data keeps its own licence (see "Data").
├── Report/                    # paper: main.pdf, main.tex, references.bib, figures/
└── Code/                      # everything code-related (run commands from here)
    ├── requirements.txt       # pinned deps (Python 3.12; CapyMOA needs Java 11+)
    ├── config.yaml            # scope (routes/dates), label thresholds, model/sim params, seed
    ├── src/                   # shared package: config, headway, features, experiment,
    │                          #   baselines, simulate, evidence, plots
    ├── data/processed/
    │   └── features.csv       # committed (~18 MB); lets you skip Stage 1 / the big download
    ├── outputs/               # committed model + simulation artifacts (small JSON/CSV)
    ├── figures/               # committed figures (PNG)
    ├── eda.ipynb              # exploratory data analysis (run from Code/)
    ├── pipeline_walkthrough.ipynb  # narrated, runnable notebook covering the whole pipeline (run from Code/)
    ├── Preprocessing/         # Stage 1 entry script + README
    ├── Experiment/            # Stage 2 entry scripts + README
    ├── Simulation/            # Stage 3 entry scripts + README
    └── Verification/          # standalone checks that re-prove key data claims
```

## Data

This repository does **not** contain the raw SUNT OD data (about 1.4 GB, 183 daily Parquet files).
It is a published, third-party dataset and is not redistributed here.

- **Source:** Ferreira, M. V. et al. (2025), *Salvador Urban Network Transportation (SUNT): A
  Landmark Spatiotemporal Dataset for Public Transportation*, *Scientific Data* 12, 1320.
  DOI: [10.1038/s41597-025-05674-6](https://doi.org/10.1038/s41597-025-05674-6).

### Where to put the data after downloading

The raw data is **not** on GitHub (it is too large and is third-party), so you download it yourself and
place it where the code expects. The scripts read the folder named by `paths.od_dir` in
`Code/config.yaml`, resolved relative to `Code/`. The default is `Dataset/SUNT/data/od`, i.e. the code
looks in `Code/Dataset/SUNT/data/od/`.

1. Download the OD Parquet files from the SUNT release (the DOI above).
2. From the repository root, create the folder and move the files into it so the layout is:

   ```
   anti-bus-bunching/
   └── Code/
       └── Dataset/SUNT/data/od/
           ├── od-2024-03-01.parquet
           ├── od-2024-03-02.parquet
           └── od-2024-03-03.parquet      # ... one od-YYYY-MM-DD.parquet per day
   ```

   The default scope (`config.yaml: scope`) uses 1-3 March 2024, so those three files are the minimum;
   add more days if you widen the date range. The filenames must follow the `od-YYYY-MM-DD.parquet`
   pattern (this is how the loader finds each day).
3. If you keep the data somewhere else, edit `Code/config.yaml: paths.od_dir` to point at your folder
   instead (an absolute path works too).

This `Dataset/` folder is git-ignored (see `.gitignore`), so the large files are never committed or
pushed back to GitHub.

You only need the raw data to **re-run Stage 1** (and for the travel-time context inside
`make_beforeafter.py`, explained below). Because the generated `Code/data/processed/features.csv` and
`Code/outputs/` are committed, you can run **Stages 2-4 immediately** without downloading anything; the
only thing that needs the raw data is the optional peak-vs-off-peak travel-time table, which the code
now skips cleanly when the data is absent.

## Setup

Requires **Python 3.12** and, for the streaming learners (CapyMOA runs on a JVM), **Java 11+** on the
`PATH` (or `JAVA_HOME` set).

```bash
cd anti-bus-bunching/Code
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running the pipeline

Run every command **from `Code/`** (`anti-bus-bunching/Code/`) so `from src ...` resolves. Each
stage takes an optional `--config config.yaml`.

```bash
cd anti-bus-bunching/Code

# Stage 1: Preprocessing (needs the SUNT OD data; otherwise use the committed features.csv)
python Preprocessing/prepare_data.py

# Stage 2: Experiment (CapyMOA; needs Java 11+)
python Experiment/run_experiment.py
python Experiment/make_evidence.py        # descriptive "before" stats + figures

# Stage 3: Simulation (counterfactual "after the nudge")
python Simulation/run_simulation.py
python Simulation/make_beforeafter.py
python Simulation/make_figures.py         # model figures (string diagram, rolling acc, lead time)
```

### Notebooks

Two notebooks accompany the scripts. Both call the same `src/` functions, so launch Jupyter from
`Code/` for the imports to resolve.

- **`eda.ipynb`** (exploratory data analysis): a read-only pass that characterises the data and
  justifies the design choices. Section A looks at the raw OD data (trajectories, a string diagram,
  forward headway, crowding, data-quality checks); Section B looks at the processed `features.csv`
  (class balance, feature distributions, which features separate bunching, class separability, and
  drift hints). It writes no files and recommends no model; it only reports neutral findings.
  Section B needs `data/processed/features.csv`, so run `prepare_data.py` first if it is missing.
- **`pipeline_walkthrough.ipynb`** (full pipeline): the narrated, runnable version of the whole
  pipeline (preprocess, train and score the streaming models, metrics and bake-off, figures, and an
  illustrative simulation). Use it as an interactive alternative to running the six scripts in order.

## Reproducing the report

The report is built from artifacts the pipeline produces. Run the six scripts from `Code/` in the
order below. Each stage feeds the next: Stage 1 builds the dataset, Stage 2 trains the model and emits
the descriptive evidence, and Stage 3 runs the counterfactual and draws the model figures. Because
`data/processed/features.csv` and `outputs/` are committed, **steps 2-6 reproduce the report's numbers
and figures without the raw SUNT download**. Only **step 1** (`prepare_data.py`) truly needs it.

There is one small, clearly-bounded exception. Step 5 (`make_beforeafter.py`) also computes a
peak-vs-off-peak **travel-time context** table, and that single table is the one quantity that can only
be derived from the raw OD data (it comes from per-trip start/end times that are not in the committed
`predictions.csv`). If the raw data is absent, `make_beforeafter.py` **skips just that table** (it
prints a clear `[skip] travel-time context needs the raw SUNT OD data` note, leaves the committed
`outputs/beforeafter_travel_time.csv` in place, and marks the travel-time fields in
`beforeafter_summary.json` as unavailable). Everything else step 5 produces, the before/after panel
(`beforeafter_panel.csv`) and the two "after" figures, is regenerated and matches the report exactly.
So without SUNT you reproduce every reported result; the only thing you cannot regenerate is the
congestion-context travel-time numbers, for which the committed file is kept.

Run each from `Code/`. The table shows what every script writes and where that output appears in the
report. Prerequisites: step 1 needs the raw SUNT download and step 2 needs Java 11+; the rest run off
the committed `features.csv` and `outputs/` (step 5 additionally uses the raw data only for the
optional travel-time table noted above).

| Run (from `Code/`) | Produces | Used in the report |
|--------------------|----------|--------------------|
| `python Preprocessing/prepare_data.py` | `data/processed/features.csv` | §3 Methodology (Data Preprocessing): the dataset and the engineered-feature tables (engineered columns and the worked example) |
| `python Experiment/run_experiment.py` | `outputs/predictions.csv`, `metrics.json`, `nudges.json`, `drift_events.json` | §4 Experiments / §5 Results: the model bake-off metrics (Table 4); also the input the simulation and figure steps consume |
| `python Experiment/make_evidence.py` | `outputs/evidence_*`, `headway_cv_by_route.png`, `headway_ratio_hist.png`, `excess_wait_by_route.png`, `severity_timeseries.png` | §5 Results (Before): headway CV, headway ratio, avoidable wait, and severity over the day |
| `python Simulation/run_simulation.py` | `outputs/simulation_results.json`, `counterfactual_comparison.png`, `counterfactual_sensitivity.png` | §5 Results (After): the counterfactual comparison and ease-off sensitivity figures, and the counterfactual numbers |
| `python Simulation/make_beforeafter.py` | `outputs/beforeafter_panel.csv`, `beforeafter_summary.json`, `severity_timeseries_after.png`, `string_diagram_1007_I_after.png` (plus `beforeafter_travel_time.csv` only when the raw SUNT data is present) | §5 Results (After): severity after the nudge and the after string diagram; the travel-time context numbers need the raw SUNT data, otherwise this step skips that one table and keeps the committed file |
| `python Simulation/make_figures.py` | `rolling_accuracy.png`, `string_diagram_1007_I.png`, `lead_time_hist_ARF.png` | §4 Experiments (Bake-off): rolling accuracy and nudge lead time; §5 Results (Before): the string diagram |

Order: `prepare_data` feeds `run_experiment` and `make_evidence`; `run_experiment` feeds
`run_simulation` and `make_figures`; `run_simulation` feeds `make_beforeafter`.
`pipeline_walkthrough.ipynb` (run from `Code/`) walks the same pipeline interactively if you prefer a
notebook. The before/after
numeric panel comes from `outputs/beforeafter_panel.csv` and `simulation_results.json`, and Figure 1
in the report is an external adapted image, not produced by the pipeline.

### Verification checks (`Code/Verification/`)

These are standalone scripts that re-prove specific data claims the report relies on. They are not
part of the pipeline run order; each is self-contained and exits `0` if its claim holds and `1` if it
fails, so they double as assertions. Run each from `Code/`.

| Run (from `Code/`) | Proves | Needs raw OD? |
|--------------------|--------|---------------|
| `python Verification/check_trip_id_is_one_bus.py` | A `trip_id` is exactly one physical bus on one run, so the headway must be keyed on the stop, not on `trip_id`. | No (uses the committed `features.csv`) |
| `python Verification/check_headway_key_needs_all_three.py` | The headway key needs all three of `(route, direction, stop_id)`; dropping any one corrupts the gap. | No |
| `python Verification/reproduce_preprocessing_chain.py` | Re-runs Stage 1 from the raw OD and reproduces the committed `features.csv` **exactly** (185,197 → 116,905 → 109,029 rows; 2,507 → 1,944 → 1,898 trips), including the 22.5% non-monotonic-trip drop and the 21,123 bunched arrivals behind the 19.4% prevalence figure. | Yes (reads the OD Parquet) |

`reproduce_preprocessing_chain.py` resolves the OD folder from `paths.od_dir` in `config.yaml` (as the
pipeline does), or you can pass the folder as the first argument; it exits `2` with a clear message when
the raw data is absent. The other two checks run off the committed `features.csv` and need no download.

### Where the report figures come from

Every figure in the report is generated by this pipeline and saved to `Code/figures/`; the copies in
`Report/figures/` are the same images under the names the LaTeX expects. The mapping is below. Eight
names are identical. Three are written with a route or model suffix (so the generated file is
self-describing), and appear in the report under a shorter name; they are the same image.

| `Code/figures/` (generated) | `Report/figures/` (used in the report) | Report figure |
|------------------------------|-----------------------------------------|---------------|
| `headway_cv_by_route.png` | `headway_cv_by_route.png` | Before: headway CV by route |
| `headway_ratio_hist.png` | `headway_ratio_hist.png` | Before: headway-ratio histogram |
| `excess_wait_by_route.png` | `excess_wait_by_route.png` | Before: avoidable wait by route |
| `severity_timeseries.png` | `severity_timeseries.png` | Before: severity over the day |
| `string_diagram_1007_I.png` | `string_diagram.png` | Before: string diagram (route 1007, dir I) |
| `rolling_accuracy.png` | `rolling_accuracy.png` | Bake-off: rolling accuracy |
| `lead_time_hist_ARF.png` | `lead_time_hist.png` | Bake-off: nudge lead-time histogram (ARF) |
| `counterfactual_comparison.png` | `counterfactual_comparison.png` | After: counterfactual comparison |
| `counterfactual_sensitivity.png` | `counterfactual_sensitivity.png` | After: ease-off sensitivity |
| `severity_timeseries_after.png` | `severity_timeseries_after.png` | After: severity over the day |
| `string_diagram_1007_I_after.png` | `string_diagram_after.png` | After: string diagram (route 1007, dir I) |

The only report image not produced by the pipeline is Figure 1 (`12469_2019_203_Fig1_HTML.png`), an
external adapted illustration.

## Configuration (`Code/config.yaml`)

Key knobs:

- `scope.routes`, `scope.directions`, `scope.start_date`, `scope.end_date`: which routes/days to use.
- `label.bunch_frac` (0.40), `label.warn_frac` (0.60), `label.horizon_stops` (3),
  `label.headway_window` (8): the bunching label definition.
- `preprocess.min_headway_seconds` (30), `preprocess.drop_nonmonotonic`: data-cleaning thresholds.
- `model.*`: ARF ensemble size, grace period, ADWIN delta, random seed.
- `simulate.*`: ease-off magnitude and cap, closing threshold, sensitivity sweep.

## Reproducibility note

The committed `Code/data/processed/features.csv` and `Code/outputs/` reproduce the numbers and
figures in the report. Full regeneration from the raw OD data requires the SUNT download and
re-running Stage 1.

## Licence

Code is released under the MIT Licence (see `LICENSE`). The SUNT dataset is **not** covered by this
licence and retains the licence of its original release; cite and obtain it from the source above.
