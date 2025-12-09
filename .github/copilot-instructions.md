# Mental Health Profiling System - AI Agent Instructions

## Project Overview
Deep learning system for predicting mental health outcomes from behavioral data. Built for FDS university project, uses PyTorch LSTM/Transformer models on 1.5M+ synthetic Kaggle records. Main system: **generate_profile.py** (3,665 lines) - parses Google Form CSVs, predicts 8 mental health outcomes, generates HTML reports with interventions.

## Architecture & Data Flow

### Core Pipeline
```
Google Form CSV → parse_google_form_csv() → LSTM/Transformer → 8 predictions → HTML/JSON reports
```

**Key Components:**
- **Input:** 17 behavioral features × 7 days (sleep, work hours, exercise, caffeine, screen time, etc.)
- **Models:** `models/saved/mental_health_lstm.pt` - multi-task LSTM with 8 prediction heads
- **Targets:** 4 daily (stress, mood, energy, focus) + 4 weekly (PSS, anxiety, depression, job_satisfaction)
- **Config:** `config/job_categories.json`, `config/thresholds.json` - job mappings & clinical thresholds

### Critical Files
- **Model Definition:** `scripts/model_definitions.py` - `MentalHealthPredictor` class (LSTM/GRU/Transformer encoders)
- **Profiling Engine:** `scripts/generate_profile.py` - end-to-end profile generation (parsing → prediction → HTML)
- **Explanation Engine:** `scripts/explain_predictions.py` - SHAP-like feature contributions, recommendations
- **History Manager:** `scripts/history_manager.py` - file-based daily tracking (`data/user_history/{user_id}/YYYY-MM-DD.json`)

## Essential Workflows

### Training a Model
```bash
# Train multi-target LSTM (8 predictions)
python scripts/train_mental_health.py --model lstm --epochs 30

# Train MLP with entity embeddings (baseline)
python scripts/train_mlp.py --epochs 80 --batch-size 256

# Preprocess raw data first (creates tabular_ml_ready.parquet)
python scripts/preprocess.py
```

**Training expects:** `data/processed/daily_with_burnout.parquet` (from `preprocess.py`)  
**Output:** `models/saved/mental_health_lstm.pt` with checkpoint dict containing `model_state`, `feature_cols`, `targets`, `scaler_mean`, `scaler_std`

### Generating Profiles
```bash
# From Google Form CSV (handles fuzzy column matching)
python scripts/generate_profile.py --csv data/martina_7day.csv --all-users --html

# From user history (last 7 days)
python scripts/generate_profile.py --from-history --user-id sarah@example.com --html

# Interactive data collection
python scripts/collect_daily_data.py --user demo@example.com
```

**Outputs:** `reports/profile_{user}_{timestamp}.json` + `report_{user}_{timestamp}.html`

### Running Demo
```bash
streamlit run demo_app.py  # Interactive web app with sliders
```

## Project-Specific Conventions

### Inverted Targets
**CRITICAL:** `mood_score`, `energy_level`, `focus_score`, `job_satisfaction` - **higher is better**. Check `INVERTED_TARGETS` set before interpreting predictions. Affects risk thresholds and trend analysis.

### Google Form Column Mapping
`generate_profile.py` uses **fuzzy matching** for 113 Google Form columns → 17 features. Pattern: lowercase, strip spaces, check substrings. See `GOOGLE_FORM_COLUMN_MAPPING` (line ~96-200) and `map_columns_fuzzy()` (line ~625-750).

Example: `"How many hours did you sleep? (Monday)"` → `sleep_hours_monday` → feature index 0

### Multi-Target Prediction Heads
Model architecture uses **shared encoder** (LSTM last hidden state) + **8 independent heads** (`PredictionHead` class). Each head returns `(regression_value, binary_classification)`. Load with:

```python
from model_definitions import MentalHealthPredictor
model = MentalHealthPredictor(input_dim=17, hidden_dim=128, encoder_type="lstm", targets=ALL_TARGETS)
checkpoint = torch.load("models/saved/mental_health_lstm.pt", weights_only=False)
model.load_state_dict(checkpoint["model_state"])
```

### Feature Importance System
`explain_predictions.py` loads pre-computed weights from `data/processed/feature_importance_weights.json` (Random Forest + Permutation + Correlation). Combines with user z-scores for SHAP-like contributions. **Controllable features** only in recommendations (see `CONTROLLABLE_FEATURES` dict).

### Job-Specific Advice
8 job categories with keyword-based classification (`classify_job_category()` in `generate_profile.py` line ~1900-1950). Each category has tailored advice generation (~75 lines each, total 600 lines). Categories: `knowledge_worker`, `healthcare`, `education`, `creative`, `management`, `service`, `manual_labor`, `sales_marketing`, `other`.

### Contradiction Detection
20+ patterns across 4 categories (physiological, work, mental health, behavioral). Example: `sleep_hours < 6 AND energy_level > 7` = HIGH severity. See `detect_contradictions()` line ~1300-1500.

### Behavioral Interventions
Evidence-based from 332 intervention cases. 6 categories: sleep hours/quality, exercise, work-life balance, caffeine, social connection, stress management. Each has: `current → target`, `change_needed`, `evidence`, `specific_actions[]`, `priority` (HIGH/MEDIUM/LOW). See `generate_behavioral_interventions()` line ~1686+.

### History Tracking
File-based storage: `data/user_history/{user_id}/{YYYY-MM-DD}.json`. `UserHistoryManager` saves/retrieves daily entries. Trend analysis in `analyze_mental_health_trends()` calculates slopes, classifies as improving/declining/stable/mixed, generates alerts.

## Testing & Validation

### Clinical Comparison System
```bash
python scripts/generate_clinical_comparison.py \
  --model-report reports/profile_test_demo@example.com_20251209_142131.json \
  --output-html reports/clinical_comparison.html
```

Compares predictions against 5 literature benchmarks (Mohr PSS, Maslach MBI, WHO-5, Mayo Clinic, Sleep Society). Generates HTML "confrontation report" with verdict analysis.

### Test Data
- `data/martina_7day.csv` - 4 real users with 7 days
- `data/user_history/test_demo@example.com/` - pre-generated history
- Run workflow test: `python scripts/test_history_workflow.py`

## Common Issues & Gotchas

### Model Loading
Use `weights_only=False` when loading checkpoints - they contain metadata dicts. Always check for `feature_cols`, `targets`, `scaler_mean`, `scaler_std` keys.

### Missing Data Handling
Imputation uses training set mean from checkpoint. If column missing in Google Form CSV, `parse_google_form_csv()` fills with `np.nan`, later replaced by scaler mean.

### Threshold Configuration
Clinical thresholds in `config/thresholds.json`:
- PSS ≥20 = High Stress (clinical cutoff)
- Anxiety ≥6 = At-Risk (custom, dataset-based)
- Depression ≥7 = At-Risk (custom, dataset-based)
- Job Satisfaction <6 = Low (custom)

### Chart.js in HTML Reports
Reports use **Chart.js CDN** (not Python package). HTML template embeds JavaScript for interactive visualizations. Self-contained 28-40KB files.

### Entity Embeddings in MLP
`train_mlp.py` uses entity embeddings for `job_type_encoded`, `work_arrangement_encoded`, `chronotype_encoded`. Embedding dims: 4-8. Load embedding metadata from `data/processed/feature_columns.json`.

## Dependencies & Environment
- **PyTorch:** 2.1.0 (core deep learning)
- **Streamlit:** 1.29.0 (demo app)
- **Pandas/NumPy:** data processing
- **scikit-learn:** preprocessing, baselines
- **Kaggle API:** dataset download (`python scripts/download_data.py`)

Run in dev container (Debian Trixie) with Python 3.11+.

## Key Documentation
- `docs/COMPLETE_PROJECT_EXPLANATION.md` - Full development journey
- `docs/EXTERNAL_BENCHMARKS.md` - Research validation & state-of-the-art comparison
- `BEHAVIORAL_INTERVENTIONS.md` - Intervention system design
- `QUICK_START_PRESENTATION.md` - Command cheat sheet
- `LONGITUDINAL_TRACKING.md` - History system architecture
- `CLINICAL_VALIDATION_COMPLETE.md` - Validation methodology

## When Modifying Code

### Adding New Features
1. Update `FEATURE_COLS` in `train_mental_health.py` and `generate_profile.py`
2. Add to `GOOGLE_FORM_COLUMN_MAPPING` if from form
3. Retrain model with `preprocess.py` → `train_mental_health.py`
4. Update scaler in checkpoint

### Adding New Targets
1. Add to `ALL_TARGETS` list
2. Create prediction head in `MentalHealthPredictor`
3. Define threshold in `config/thresholds.json`
4. Update HTML template in `generate_html_report()`

### Changing Job Categories
Edit `config/job_categories.json` keywords, then update advice generation in `generate_job_specific_advice()` (~line 2000-2600).

## Performance Expectations
- **Job Satisfaction:** 98.5% accuracy (R² = 0.98)
- **Depression/Anxiety:** 97-98% accuracy
- **PSS:** 97% accuracy
- **Daily metrics:** 52-85% (more variable, harder to predict)
- **Inference time:** <100ms per user (CPU)
