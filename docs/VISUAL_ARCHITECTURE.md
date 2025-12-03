# Visual Project Architecture

## 🏗️ System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MENTAL HEALTH PROFILING SYSTEM                     │
│                         Complete End-to-End Architecture                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────── DATA LAYER ────────────────────────┐
│                                                             │
│  📊 Kaggle Dataset (1.5M+ records)                          │
│  ├── daily_all.csv         (500K daily observations)        │
│  ├── interventions.csv     (332 intervention cases)         │
│  ├── users.csv             (2,100 synthetic workers)        │
│  └── weekly_summaries.csv  (aggregated metrics)             │
│                                                             │
│  📥 User Input                                              │
│  └── Google Form CSV (4 test users, 113 columns)           │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌──────────────────── PREPROCESSING ──────────────────────────┐
│                                                             │
│  🔧 scripts/preprocess.py                                   │
│  • Clean raw data                                           │
│  • Handle missing values                                    │
│  • Feature engineering (7-day sequences)                    │
│  • StandardScaler normalization                             │
│                                                             │
│  Output: data/processed/                                    │
│  ├── tabular_ml_ready.parquet (9.2 MB)                     │
│  ├── feature_scaler.joblib    (1.7 KB)                     │
│  └── feature_columns.json     (3.1 KB)                     │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌──────────────────── MODEL TRAINING ─────────────────────────┐
│                                                             │
│  🧠 scripts/train_mental_health.py                          │
│                                                             │
│  Architecture: Multi-Task LSTM                              │
│  ┌────────────────────────────────────────┐                │
│  │  Input: [batch, 7 days, 17 features]  │                │
│  │             ↓                          │                │
│  │  LSTM (2 layers, 128 hidden, dropout) │                │
│  │             ↓                          │                │
│  │  Last Hidden State [batch, 128]       │                │
│  │             ↓                          │                │
│  │  ┌──────────┴──────────┐              │                │
│  │  │  8 Prediction Heads │              │                │
│  │  └─────────┬───────────┘              │                │
│  │      ┌─────┴─────┐                    │                │
│  │      ↓           ↓                    │                │
│  │  Regression  Binary Class              │                │
│  │  (value)     (at-risk)                │                │
│  └────────────────────────────────────────┘                │
│                                                             │
│  Targets (8):                                               │
│  Daily:  stress_level, mood, energy, focus                 │
│  Weekly: PSS, anxiety, depression, job_sat                 │
│                                                             │
│  Training:                                                  │
│  • Loss: MSE (regression) + BCE (classification)           │
│  • Optimizer: AdamW (lr=1e-3, weight_decay=1e-5)           │
│  • Epochs: 20-30 with early stopping                       │
│  • Split: 70% train, 15% val, 15% test                     │
│                                                             │
│  Output: models/saved/mental_health_lstm.pt (1.2 MB)       │
│                                                             │
│  Performance:                                               │
│  • Job Satisfaction: 98.5% accuracy                        │
│  • Depression: 98% accuracy                                │
│  • Anxiety: 97% accuracy                                   │
│  • PSS: 97% accuracy                                       │
│  • Daily metrics: 52-85% (more variable)                   │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌──────────────── PROFILING SYSTEM ───────────────────────────┐
│                                                             │
│  🎯 scripts/generate_profile.py (3,089 lines)               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Phase 1: INPUT PARSING                             │   │
│  │  ───────────────────────────────────────            │   │
│  │  parse_google_form_csv()                            │   │
│  │  • Map 113 columns → 17 features × 7 days          │   │
│  │  • Fuzzy column matching (10+ patterns)            │   │
│  │  • Handle: timestamps, emails, job titles          │   │
│  │  • Missing data: flag for imputation               │   │
│  │                                                     │   │
│  │  Output: List[Dict] with user profiles             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Phase 2: PREDICTION                                │   │
│  │  ───────────────────────────────────────            │   │
│  │  predict_user()                                     │   │
│  │  • Load model + scaler                              │   │
│  │  • Impute missing (use training mean)               │   │
│  │  • Normalize with StandardScaler                    │   │
│  │  • LSTM inference                                   │   │
│  │  • Extract 8 predictions + confidence               │   │
│  │  • At-risk classification (thresholds)              │   │
│  │                                                     │   │
│  │  Output: predictions dict                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Phase 3: ANALYSIS                                  │   │
│  │  ───────────────────────────────────────            │   │
│  │  1. classify_job_category()                         │   │
│  │     • Keyword matching (8 categories)               │   │
│  │     • Example: "Film Producer" → Creative          │   │
│  │                                                     │   │
│  │  2. detect_contradictions()                         │   │
│  │     • 20+ contradiction patterns                    │   │
│  │     • Categories: physiological, work,              │   │
│  │       mental health, behavioral                     │   │
│  │     • Severity: HIGH/MEDIUM/LOW                     │   │
│  │                                                     │   │
│  │  3. generate_behavioral_interventions()             │   │
│  │     • Evidence-based from 332 cases                 │   │
│  │     • 6 categories: sleep (hours/quality),          │   │
│  │       exercise, work-life, caffeine, social         │   │
│  │     • Each: current→target, evidence, actions       │   │
│  │     • Priority scoring (HIGH/MEDIUM/LOW)            │   │
│  │                                                     │   │
│  │  4. generate_job_specific_advice()                  │   │
│  │     • 8 job categories × 75 lines = 600 lines      │   │
│  │     • Tailored recommendations                      │   │
│  │     • Priorities: HIGH/MEDIUM/LOW                   │   │
│  │                                                     │   │
│  │  5. analyze_mental_health_trends()                  │   │
│  │     • Load past JSON assessments                    │   │
│  │     • Calculate trend slopes                        │   │
│  │     • Classify: improving/declining/stable/mixed    │   │
│  │     • Generate alerts (deterioration warnings)      │   │
│  │     • Handle INVERTED_TARGETS correctly             │   │
│  │                                                     │   │
│  │  Output: UserProfile dataclass                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Phase 4: OUTPUT GENERATION                         │   │
│  │  ───────────────────────────────────────────        │   │
│  │  1. print_profile_summary()                         │   │
│  │     • Console output (formatted text)               │   │
│  │     • Emoji indicators (✓/⚠️/🔴)                     │   │
│  │     • Risk factors & positive factors               │   │
│  │                                                     │   │
│  │  2. save_profile_json()                             │   │
│  │     • Machine-readable format                       │   │
│  │     • All fields: predictions, risks,               │   │
│  │       recommendations, interventions, history       │   │
│  │     • Used for history tracking                     │   │
│  │                                                     │   │
│  │  3. generate_html_report()                          │   │
│  │     • 2000+ lines HTML template                     │   │
│  │     • Embedded Chart.js (CDN)                       │   │
│  │     • Sections:                                     │   │
│  │       - Header (user, job, quality)                 │   │
│  │       - Dashboard cards (8 predictions)             │   │
│  │       - Interactive charts (line + bar)             │   │
│  │       - Behavioral patterns (7-day trends)          │   │
│  │       - Risk assessment (color-coded)               │   │
│  │       - Interventions (current→target cards)        │   │
│  │       - Recommendations (expandable)                │   │
│  │       - Contradictions (warning boxes)              │   │
│  │       - History (timeline with arrows)              │   │
│  │       - Scale guide (depression interpretation)     │   │
│  │     • Responsive design (mobile-friendly)           │   │
│  │     • Print-friendly CSS                            │   │
│  │     • Self-contained (28-40 KB)                     │   │
│  │                                                     │   │
│  │  Outputs:                                           │   │
│  │  • reports/profile_USER_TIMESTAMP.json              │   │
│  │  • reports/report_USER_TIMESTAMP.html               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌──────────────────── OUTPUT FILES ───────────────────────────┐
│                                                             │
│  📄 reports/final_interventions/                            │
│  ├── profile_Marti_20251203_221204.json                    │
│  │   {                                                     │
│  │     "user_id": "Marti",                                 │
│  │     "job_category": "education",                        │
│  │     "predictions": {...},                               │
│  │     "risk_factors": [...],                              │
│  │     "positive_factors": [...],                          │
│  │     "contradictions": [...],                            │
│  │     "recommendations": [...],                           │
│  │     "behavioral_interventions": [                       │
│  │       {                                                 │
│  │         "category": "Sleep Hours",                      │
│  │         "current": "4.1h/night",                        │
│  │         "target": "7-9h/night",                         │
│  │         "change_needed": "+3.4h",                       │
│  │         "evidence": "31% stress reduction...",          │
│  │         "specific_actions": [...],                      │
│  │         "priority": "HIGH"                              │
│  │       }                                                 │
│  │     ],                                                  │
│  │     "history_analysis": null,                           │
│  │     "data_quality_score": 0.888                         │
│  │   }                                                     │
│  │                                                         │
│  └── report_Marti_20251203_221204.html                     │
│      • Beautiful visual report                             │
│      • Interactive charts                                  │
│      • Actionable interventions                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘


┌────────────────────── KEY COMPONENTS ───────────────────────┐
│                                                             │
│  📦 DATA STRUCTURES                                         │
│  ─────────────────────────────────────────                 │
│  @dataclass UserProfile:                                    │
│    • user_id: str                                           │
│    • timestamp: datetime                                    │
│    • job_category: str                                      │
│    • behavioral_data: DataFrame (7 × 17)                    │
│    • predictions: Dict (8 targets)                          │
│    • contradictions: List[Dict]                             │
│    • risk_factors: List[str]                                │
│    • positive_factors: List[str]                            │
│    • recommendations: List[Dict]                            │
│    • behavioral_interventions: List[Dict]  ⭐ NEW           │
│    • history_analysis: Optional[Dict]                       │
│    • data_quality_score: float                              │
│    • missing_features: List[str]                            │
│                                                             │
│  🎯 THRESHOLDS                                              │
│  ─────────────────────────────────────────                 │
│  AT_RISK_THRESHOLDS = {                                     │
│    'perceived_stress_scale': 20,  # PSS ≥20                │
│    'anxiety_score': 6,            # GAD-7 ≥6               │
│    'depression_score': 15,        # Top 5%                 │
│    'job_satisfaction': 5,         # <5 (inverted)          │
│    'stress_level': 7,             # ≥7                     │
│    'mood_score': 4,               # <4 (inverted)          │
│    'energy_level': 4,             # <4 (inverted)          │
│    'focus_score': 4,              # <4 (inverted)          │
│  }                                                          │
│                                                             │
│  INVERTED_TARGETS = {                                       │
│    "mood_score",                                            │
│    "energy_level",                                          │
│    "focus_score",                                           │
│    "job_satisfaction"                                       │
│  }                                                          │
│  # Higher = better for these targets                       │
│                                                             │
│  📊 JOB CATEGORIES (8)                                      │
│  ─────────────────────────────────────────                 │
│  1. Knowledge Worker (software, data science)               │
│  2. Healthcare (doctors, nurses, therapists)                │
│  3. Education (teachers, professors)                        │
│  4. Creative (designers, artists, film producers)           │
│  5. Management (managers, executives)                       │
│  6. Service (customer service, hospitality)                 │
│  7. Manual Labor (construction, warehouse)                  │
│  8. Sales/Marketing (sales, business dev)                   │
│  + Other (fallback)                                         │
│                                                             │
│  🔍 CONTRADICTIONS (20+ patterns)                           │
│  ─────────────────────────────────────────                 │
│  1. High caffeine + good sleep quality                      │
│  2. Severe sleep deficit + high energy                      │
│  3. High work hours + low productivity                      │
│  4. At-risk mental health + poor self-care                  │
│  5. Anxiety + social isolation                              │
│  6. High outdoor + high screen time                         │
│  ... (14+ more patterns)                                    │
│                                                             │
│  🎯 INTERVENTIONS (6 categories + 1 conditional)            │
│  ─────────────────────────────────────────                 │
│  1. Sleep Hours (target: 7-9h)                              │
│  2. Sleep Quality (target: 7-9/10)                          │
│  3. Physical Activity (target: 30+ min)                     │
│  4. Work-Life Balance (target: ≤8-9h)                       │
│  5. Caffeine (target: 200-400mg)                            │
│  6. Social Connection (target: 4-6 interactions)            │
│  7. Stress Management (if PSS ≥20 or Anxiety ≥6)            │
│                                                             │
│  Evidence source: interventions.csv (332 cases)             │
│  • therapy (56), diet_coaching (51),                        │
│  • exercise_plan (51), meditation (48),                     │
│  • sick_leave (47), vacation (44),                          │
│  • workload_cap (35)                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘


┌────────────────── DEVELOPMENT TIMELINE ─────────────────────┐
│                                                             │
│  Week 1-2: Foundation                                       │
│  • Dataset download & EDA                                   │
│  • Preprocessing pipeline                                   │
│  • Baseline ML models                                       │
│  ✅ Output: tabular_ml_ready.parquet                        │
│                                                             │
│  Week 3-4: Deep Learning                                    │
│  • MLP, LSTM, GRU, Transformer experiments                  │
│  • Burnout classification (3-class)                         │
│  • CVAE for interventions                                   │
│  ✅ Output: lstm_sequence.pt (98% accuracy)                 │
│                                                             │
│  Week 5: Multi-Target Pivot                                 │
│  • Redesign for 8 mental health targets                     │
│  • Multi-task learning architecture                         │
│  • Train mental_health_lstm.pt                              │
│  ✅ Output: 98%+ on stable metrics                          │
│                                                             │
│  Week 6: Google Form Integration                            │
│  • Flexible CSV parser (113 columns)                        │
│  • Column mapping with fuzzy matching                       │
│  • Missing data imputation                                  │
│  ✅ Output: Working profile generator                       │
│                                                             │
│  Week 7: Feature Expansion                                  │
│  • Risk assessment system                                   │
│  • Job-specific advice (8 × 75 lines)                       │
│  • Contradiction detection (20+ patterns)                   │
│  • HTML report generation                                   │
│  ✅ Output: Comprehensive profiling                         │
│                                                             │
│  Week 8: Advanced Features                                  │
│  • Mental health history tracking                           │
│  • Trend analysis with alerts                               │
│  • Fixed inverted targets bug                               │
│  • Depression scale interpretation                          │
│  ✅ Output: Longitudinal capabilities                       │
│                                                             │
│  Week 9: Behavioral Interventions ⭐ LATEST                  │
│  • Evaluated CVAE (incompatible)                            │
│  • Data-driven intervention system                          │
│  • Evidence from 332 cases                                  │
│  • Integrated into all outputs                              │
│  ✅ Output: Complete actionable system                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘


┌──────────────── PERFORMANCE SUMMARY ────────────────────────┐
│                                                             │
│  Target                  | Accuracy | MSE  | Confidence    │
│  ─────────────────────────────────────────────────────────  │
│  Job Satisfaction        | 98.5%    | 0.15 | Very High     │
│  Depression              | 98%      | 2.3  | Very High     │
│  Anxiety                 | 97%      | 1.8  | Very High     │
│  PSS                     | 97%      | 3.5  | Very High     │
│  Mood Score              | 85%      | 0.8  | High          │
│  Energy Level            | 62%      | 1.2  | Moderate      │
│  Focus Score             | 52%      | 1.5  | Moderate      │
│  Stress Level            | 52%      | 1.4  | Moderate      │
│                                                             │
│  Insight: Weekly/stable metrics are highly predictable,     │
│           daily/volatile metrics harder to forecast         │
│                                                             │
└─────────────────────────────────────────────────────────────┘


┌────────────────── FILE SIZE METRICS ────────────────────────┐
│                                                             │
│  Component                        | Size                    │
│  ─────────────────────────────────────────────────────────  │
│  Dataset (raw CSV total)          | 1.5M+ records           │
│  Processed data (parquet)         | 9.2 MB                  │
│  LSTM model weights               | 1.2 MB                  │
│  Transformer model weights        | 2.0 MB                  │
│  Feature scaler                   | 1.7 KB                  │
│  Main script (generate_profile.py)| 3,089 lines             │
│  HTML report (per user)           | 28-40 KB                │
│  JSON profile (per user)          | 2-3 KB                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Quick Reference: Command Examples

```bash
# Single user profile
python scripts/generate_profile.py \
  --csv data/form_responses.csv \
  --user-id "Marti" \
  --html \
  --output-dir reports/profiles

# All users in CSV
python scripts/generate_profile.py \
  --csv data/survey_batch.csv \
  --all-users \
  --html \
  --output-dir reports/batch

# With 90-day history tracking
python scripts/generate_profile.py \
  --csv data/new_assessment.csv \
  --user-id "Marti" \
  --html \
  --lookback-days 90 \
  --output-dir reports/longitudinal

# Custom model
python scripts/generate_profile.py \
  --csv data/responses.csv \
  --user-id "John" \
  --model-path models/saved/mental_health_transformer.pt \
  --html
```

---

## 📊 Repository Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | 15 |
| Total Lines of Code | ~15,000+ |
| Models Trained | 13 |
| Dataset Records | 1,568,337 |
| Features (Input) | 17 × 7 days = 119 |
| Targets (Output) | 8 |
| Job Categories | 8 + generic |
| Contradiction Patterns | 20+ |
| Intervention Categories | 6 + conditional |
| Test Users Profiled | 4 |
| HTML Report Size | 28-40 KB |
| LSTM Model Size | 1.2 MB |
| Best Accuracy | 98.5% (job_sat) |

---

*Visual architecture document created: December 3, 2025*
