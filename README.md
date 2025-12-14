# FDS Project – Deep Learning for Daily Mental Wellness

This repository contains the code for our **Foundations of Data Science** final project.

We use the **Work-Life Balance Synthetic Daily Wellness Dataset** (Kaggle) to predict
daily mental wellness indicators of workers from their work–life patterns and
recent history, using deep learning models (LSTM and Transformer).

**🎯 Key Achievement:** 98.5% accuracy on job satisfaction prediction (vs 89.3% baseline)

> 📊 **Research Context:** Our results align with state-of-the-art synthetic data studies (94-99% accuracy) and represent a theoretical upper bound compared to real-world wearable studies (70-82%). See [External Benchmarks & Related Work](docs/EXTERNAL_BENCHMARKS.md) for detailed comparison with clinical and ML research.

---

## 🚀 Quick Start

### Interactive Demo (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive Streamlit demo
streamlit run demo_app.py
```

## 🧪 Reproduction
1. **Setup:** `pip install -r requirements.txt`
2. **Demo:** `streamlit run demo_app.py`
3. **Research:** See `reports/two_stage_analysis/` for pre-generated uncertainty plots.

## ⚠️ Disclaimer
This software provides **statistical inferences** based on synthetic and aggregate patterns. It is **not a diagnostic tool**. The "Sedentary Safety Layer" manually adjusts scores based on clinical heuristics (see `config/thresholds.json`).


### Generate Profile from CSV
```bash
python scripts/generate_profile.py \
  --csv data/martina_7day.csv \
  --all-users \
  --html \
  --output-dir reports/profiles
```

### View Example Reports
Open any file in `reports/examples/`:
- `teacher_profile.html` - Education sector example
- `software_engineer_profile.html` - Knowledge worker example  
- `creative_professional_profile.html` - Film producer example

---

## 1. Project Overview

**Goal.**

Given daily information about workers (sleep, working hours, breaks, exercise,
screen time, etc.), we want to:

1. **Predict daily mental wellness scores** (e.g., stress, mood, burnout risk)
   from the features of a single day (tabular ML).
2. **Use time series models (LSTM/GRU)** to predict the wellness of tomorrow
   from the last *N* days of behaviour.
3. Optionally, learn a compact representation of behaviour patterns with an
   **autoencoder** and explore clusters of worker profiles.

We treat this as a **deep learning project on tabular + time-series data**, with
classical ML baselines for comparison.

---

## 2. Dataset

We use the Kaggle dataset:

> **Work-Life Balance Synthetic Daily Wellness Dataset**  
> by Wafaa El-Husseini  
> https://www.kaggle.com/datasets/wafaaelhusseini/worklife-balance-synthetic-daily-wellness-dataset

The dataset is **synthetic**, which avoids privacy issues but preserves realistic
relationships between work habits and wellness indicators.

We do **not** store the CSV files in this repository.  
Instead, each collaborator downloads the data locally using the Kaggle API
(see below).

---

## 3. Repository Structure

```text
FDS-Project/
  ├── data/
  │   └── raw/                  # (ignored by git) Kaggle CSV files go here
  ├── notebooks/
  │   ├── 01_eda.ipynb          # exploratory data analysis
  │   ├── 02_baseline_models.ipynb
  │   ├── 03_deep_mlp.ipynb
  │   └── 04_lstm_timeseries.ipynb
  ├── scripts/
  │   └── download_data.py      # Kaggle download script
  ├── models/                   # (optional) saved model definitions / weights
  ├── docs/
  │   ├── EXTERNAL_BENCHMARKS.md  # Research comparison & validation
  │   └── COMPLETE_PROJECT_EXPLANATION.md
  ├── .gitignore
  ├── README.md
  └── requirements.txt          # (to be added)
