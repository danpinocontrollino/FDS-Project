*** Begin Patch
# FDS Project – Deep Learning for Daily Mental Wellness

This `presentation_bundle` contains a focused, runnable subset of the full FDS project used for demos and presentations. The goal of the project is to predict per-day mental-wellness indicators from behavioural and time-series data and to provide HTML profile reports and an interactive demo.

## Summary

- Goal: given daily features (sleep, working hours, breaks, exercise, screen time, app usage, etc.) predict wellness indicators (stress, mood, job satisfaction) for the same day and short-term future using tabular and sequence models (ML baselines, LSTM/GRU/Transformer).
- Deliverables in this bundle: an interactive Streamlit demo (`demo_app.py`), profile/report generation scripts, example reports in `reports/`, and minimal configs to reproduce the demo.

## What we will need

- Python 3.10 or newer.
- A virtual environment (`venv` or `conda`) is recommended.
- Install Python dependencies from `requirements.txt` (present in this bundle). Core packages used:
   - `numpy`, `pandas`, `scikit-learn` – data processing and baselines
   - `torch` (or `tensorflow`) – for LSTM/GRU/training (GPU optional but recommended for speed)
   - `streamlit` – interactive demo UI
   - `jinja2` / `beautifulsoup4` or similar – HTML report generation
- Optional: CUDA-capable GPU + matching `torch` build to speed model training and evaluation.

## Data

- Source dataset: "Work-Life Balance Synthetic Daily Wellness Dataset" (Kaggle). This repository does not include the CSVs.
- Place CSVs under `data/raw/` (create this directory). Expected pattern: `data/raw/*.csv` or `data/raw/<user>_7day.csv` for demo profiles.
- A small sample or single-user CSV (7-day window) is sufficient to run the demo and generate example reports.

## Configuration files

- `config/job_categories.json` — mapping used by the report generator.
- `config/thresholds.json` — numeric thresholds used to turn scores into flags/actions.

Keep these files next to the bundle or in the same `presentation_bundle/config/` folder. The demo reads these configs at runtime.

## How to run

1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Prepare data (example)

```bash
mkdir -p data/raw
# copy or download one CSV to data/raw/, e.g. data/raw/martina_7day.csv
```

3) Run interactive demo (Streamlit)

```bash
streamlit run demo_app.py --server.port 8501
```

4) Generate a single HTML profile from CSV (non-interactive)

```bash
python scripts/generate_profile.py --csv data/raw/martina_7day.csv --html --output-dir reports/profiles
```

## Reproducibility & training

- Training scripts use deterministic seeds in the top of each script; check `scripts/` for exact commands.
- To retrain models end-to-end you will need the full dataset and sufficient compute; use `scripts/train_*` entry points and monitor GPU memory.

## Files of interest in this bundle

- `demo_app.py` — Streamlit demo and entry point for interactive exploration.
- `scripts/generate_profile.py` — single-user HTML profile generator.
- `reports/` — example generated HTML reports and assets.
- `requirements.txt` — Python dependencies.

## Notes & next steps for collaborators

- Add a small sample CSV to `data/raw/` for quick demo runs.
- If you want reproducible exact experiments, add a `run_experiment.sh` wrapper that pins seeds and logs config files.
- For presentation, prefer running the demo on a machine with a stable Python environment or build a lightweight Docker image.
