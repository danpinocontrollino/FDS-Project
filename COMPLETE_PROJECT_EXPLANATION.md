# Complete Project Explanation: Mental Health Profiling System

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset & Problem](#dataset--problem)
3. [Repository Structure](#repository-structure)
4. [Development Journey](#development-journey)
5. [Technical Architecture](#technical-architecture)
6. [Key Components](#key-components)
7. [The Build Process](#the-build-process)
8. [How Everything Connects](#how-everything-connects)
9. [Usage & Examples](#usage--examples)

---

## 🎯 Project Overview

**Goal**: Build an intelligent mental health profiling system that predicts worker wellness and provides personalized, data-driven recommendations.

**What It Does**:
- Takes 7 days of behavioral data (sleep, work hours, exercise, etc.) from Google Forms
- Predicts 8 mental health outcomes using deep learning (LSTM/Transformer)
- Generates comprehensive HTML reports with:
  - Risk assessments
  - Job-specific advice
  - Contradiction detection
  - Evidence-based behavioral interventions
  - Historical trend analysis

**Real-World Application**: Mental health monitoring for workplaces, self-assessment tools, early intervention systems.

---

## 📊 Dataset & Problem

### Dataset: Work-Life Balance Synthetic Daily Wellness
- **Source**: Kaggle (Wafaa El-Husseini)
- **Size**: 1.5M+ records across multiple CSV files
- **Records**: ~500,000 unique daily observations
- **Users**: ~2,100 synthetic workers tracked over time

### Data Files
```
data/raw/
├── daily_all.csv          # Main dataset: daily behavioral + mental health data
├── daily_logs.csv         # Daily activity logs
├── interventions.csv      # 332 intervention cases (therapy, exercise, etc.)
├── users.csv              # User demographics and baselines
└── weekly_summaries.csv   # Aggregated weekly metrics
```

### Key Features (17 Behavioral Variables)
**Sleep**: `sleep_hours`, `sleep_quality`  
**Work**: `work_hours`, `meetings_count`, `tasks_completed`, `emails_received`, `commute_minutes`  
**Physical Health**: `exercise_minutes`, `steps_count`, `caffeine_mg`, `alcohol_units`  
**Lifestyle**: `screen_time_hours`, `social_interactions`, `outdoor_time_minutes`, `diet_quality`  
**Context**: `work_pressure`, `weather_mood_impact`

### Target Variables (8 Mental Health Outcomes)

**Daily Predictions** (next day):
- `stress_level` (1-10)
- `mood_score` (1-10, higher = better)
- `energy_level` (1-10, higher = better)
- `focus_score` (1-10, higher = better)

**Weekly Predictions** (end of week):
- `perceived_stress_scale` (PSS, 0-40, clinical scale)
- `anxiety_score` (GAD-7, 0-21, clinical scale)
- `depression_score` (PHQ-9, 0-27, custom 0-22 scale in our data)
- `job_satisfaction` (1-9, higher = better)

---

## 📁 Repository Structure

```
FDS-Project/
│
├── data/
│   ├── raw/                          # Original CSV files (1.5M+ records)
│   ├── processed/                    # Preprocessed data
│   │   ├── tabular_ml_ready.parquet # Preprocessed for ML
│   │   ├── feature_scaler.joblib    # StandardScaler for normalization
│   │   └── feature_columns.json     # Feature metadata
│   └── martina_7day.csv             # Test data (4 real users)
│
├── models/saved/                     # Trained models (13 models total)
│   ├── mental_health_lstm.pt        # Main LSTM model (8 targets)
│   ├── mental_health_transformer.pt # Transformer alternative
│   ├── mlp_classifier.pt            # Baseline MLP
│   ├── lstm_sequence.pt             # Old burnout LSTM
│   ├── cvae_advisor.pt              # CVAE (legacy, incompatible)
│   └── ...                          # Other experimental models
│
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_baseline_models.ipynb     # Classical ML baselines
│   ├── 03_deep_learning_mlp.ipynb   # MLP experiments
│   ├── 04_lstm_sequence_model.ipynb # LSTM development
│   └── 05_what_if_engine.ipynb      # Counterfactual analysis
│
├── scripts/                          # Production Python scripts
│   ├── generate_profile.py          # 🌟 MAIN PROFILING SYSTEM (3089 lines)
│   ├── train_mental_health.py       # Train multi-target LSTM/Transformer
│   ├── preprocess.py                # Data preprocessing pipeline
│   ├── download_data.py             # Kaggle API download
│   └── ...                          # Other training/prediction scripts
│
├── reports/                          # Generated outputs
│   ├── final_interventions/         # Latest HTML + JSON profiles
│   └── ...                          # Previous test runs
│
├── docs/                             # Documentation
│   └── COMPLETE_PROJECT_EXPLANATION.md  # This file
│
├── BEHAVIORAL_INTERVENTIONS.md       # Intervention system docs
├── README.md                         # Project overview
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Development Journey

### Phase 1: Data Exploration & Preprocessing
**Notebooks**: `01_eda.ipynb`, `02_baseline_models.ipynb`

1. **Downloaded Dataset** from Kaggle (1.5M+ records)
2. **Exploratory Analysis**:
   - Distributions of behavioral features
   - Correlations with mental health outcomes
   - Missing data patterns
   - User demographics
3. **Feature Engineering**:
   - Created 7-day sequences for time-series models
   - Normalized features with StandardScaler
   - Handled missing values with imputation
4. **Baseline Models** (Classical ML):
   - Logistic Regression
   - Random Forest (98% accuracy on burnout)
   - Gradient Boosting

**Output**: `data/processed/tabular_ml_ready.parquet`, `feature_scaler.joblib`

---

### Phase 2: Deep Learning Models
**Notebooks**: `03_deep_learning_mlp.ipynb`, `04_lstm_sequence_model.ipynb`

#### Initial Approach: Burnout Classification (3-class)
- **Problem**: Predict burnout level (0=none, 1=moderate, 2=severe)
- **Models**:
  - MLP (Multi-Layer Perceptron)
  - LSTM (for sequences)
  - GRU (variant of LSTM)
  - Transformer (attention-based)
  - CVAE (Conditional Variational Autoencoder for interventions)

**Best Result**: 98.5% accuracy on job satisfaction, 85%+ on burnout

#### Pivot: Multi-Target Mental Health Prediction
**Why?** Burnout is oversimplified; mental health is multidimensional.

**New Approach** (`train_mental_health.py`):
- Predict **8 simultaneous targets** (stress, mood, energy, focus, PSS, anxiety, depression, job_sat)
- Multi-task learning with shared encoder + separate prediction heads
- Supports both LSTM and Transformer architectures

**Architecture**:
```
Input: [batch, 7 days, 17 features]
    ↓
LSTM Encoder (2 layers, 128 hidden)
    ↓
Last hidden state [batch, 128]
    ↓
8 Prediction Heads (1 per target)
    ├─→ Stress Level (regression + binary classification)
    ├─→ Mood Score
    ├─→ Energy Level
    ├─→ Focus Score
    ├─→ PSS (Perceived Stress Scale)
    ├─→ Anxiety Score
    ├─→ Depression Score
    └─→ Job Satisfaction
```

**Training**:
- Loss: Combined MSE (regression) + BCE (classification)
- Optimizer: AdamW with weight decay
- Learning rate: 1e-3 with ReduceLROnPlateau
- Epochs: 20-30 with early stopping
- Dataset: 70% train, 15% val, 15% test

**Results**:
- Job Satisfaction: 98.5% accuracy, MSE 0.15
- Depression: 98% accuracy, MSE 2.3
- Stress: 52% accuracy (harder to predict)

**Output**: `models/saved/mental_health_lstm.pt` (1.2MB)

---

### Phase 3: Google Form Integration
**Script**: `generate_profile.py` (initial version)

**Challenge**: Real users submit data via Google Forms, not CSV with perfect columns.

**Solution**: Flexible column mapping
- Google Forms creates columns like: "Day 1 - Sleep Hours", "Day 2 - Sleep Hours", etc.
- Built parser to map 113+ column variations to 17 features × 7 days
- Handles:
  - Different naming conventions ("outdoor time" vs "outdoor_time")
  - Missing days
  - Extra columns (email, timestamp, job title)

**Key Function**: `parse_google_form_csv()`
```python
def parse_google_form_csv(csv_path: Path) -> List[Dict]:
    # Maps messy Google Form columns → clean feature matrix
    # Returns list of users with 7-day behavioral sequences
```

---

### Phase 4: Profiling System Features
**Script**: `generate_profile.py` (expanding to 3089 lines)

#### Feature 1: Basic Predictions
- Load model and scaler
- Predict 8 mental health outcomes
- Classify as "at-risk" based on thresholds
- Calculate confidence scores

#### Feature 2: Risk Assessment
**Risk Factors** (automatically detected):
- At-risk PSS (≥20)
- At-risk anxiety (≥6)
- At-risk depression (≥15 in our scale)
- Low job satisfaction (<5)
- Insufficient sleep (<6h)
- Sedentary lifestyle (0-20min exercise)
- High caffeine (>400mg)
- Excessive work hours (>9h)

**Positive Factors**:
- Healthy mental health scores
- Good sleep (7-9h)
- Regular exercise (30+ min)
- Strong social connections (4+ interactions)
- Adequate outdoor time (30+ min)
- Healthy diet (7+ rating)

#### Feature 3: Job-Specific Advice (600+ lines)
**Challenge**: Generic advice is useless; advice must be context-specific.

**Solution**: 8 job category systems with tailored recommendations

**Categories**:
1. **Knowledge Worker** (software, data science): Focus on deep work, context switching, screen fatigue
2. **Healthcare** (doctors, nurses): Compassion fatigue, shift work, emotional labor
3. **Education** (teachers): Emotional regulation, boundary-setting, burnout prevention
4. **Creative** (designers, artists, film): Creative blocks, irregular schedules, project stress
5. **Management** (executives, leads): Decision fatigue, delegation, team dynamics
6. **Service** (customer service): Emotional exhaustion, difficult customers, repetition
7. **Manual Labor** (construction, warehouse): Physical recovery, injury prevention, schedule rigidity
8. **Sales/Marketing**: Rejection resilience, performance pressure, variability

**Automatic Classification**:
- Keyword matching from job title
- Example: "Film Producer" → Creative category
- Fallback: "Other" category

**Recommendation Engine**:
- Analyzes predictions + behavioral patterns
- Generates 15-30 prioritized recommendations
- Each has: category, priority (HIGH/MEDIUM/LOW), advice, rationale, job-specific tip

#### Feature 4: Contradiction Detection (20+ patterns)
**Why?** Users often have contradictory responses revealing:
- Survey mistakes
- Cognitive dissonance
- Hidden issues

**Categories of Contradictions**:

1. **Physiological Contradictions**
   - High caffeine + good sleep quality (unusual)
   - Severe sleep deficit + high energy (compensatory caffeine?)
   - Low exercise + no fatigue (sedentary adaptation)

2. **Work Contradictions**
   - High work hours + low productivity (inefficiency or burnout)
   - Minimal work + low satisfaction (underemployment)
   - High pressure + high satisfaction (thriving vs surviving?)

3. **Mental Health Contradictions**
   - At-risk + poor self-care foundation (interventions won't work)
   - Anxiety + social isolation (avoidance pattern)
   - Depression + healthy behaviors (subclinical or emerging)

4. **Behavioral Contradictions**
   - High outdoor time + high screen time (measurement error?)
   - Poor diet + regular exercise (selective health focus)
   - Social interactions + loneliness (quality vs quantity)

**Detection Function**: `detect_contradictions()`
```python
# Returns list of contradictions with:
# - severity: HIGH/MEDIUM/LOW
# - pattern description
# - reasoning (why it's unusual)
# - recommendation (what to check)
```

#### Feature 5: Mental Health History Tracking
**Challenge**: One snapshot isn't enough; need longitudinal view.

**Solution**: Multi-assessment trend analysis

**If user has history** (multiple assessments over time):
1. **Load Past Assessments**: JSON files from previous weeks/months
2. **Trend Analysis**: Calculate slopes for each target
3. **Trajectory Classification**:
   - "improving" (positive slope, at-risk → healthy)
   - "declining" (negative slope, healthy → at-risk)
   - "stable" (minimal change)
   - "mixed" (some improving, some declining)
4. **Alert Generation**:
   - Recent deterioration warnings
   - Sustained at-risk flags
   - Improvement acknowledgments
5. **Insights**:
   - "Stress improved 18% over 90 days"
   - "Anxiety worsening - 3 consecutive at-risk weeks"

**Key Function**: `analyze_mental_health_trends()`

**CRITICAL BUG FIX**: Some targets are inverted (higher = better)
```python
INVERTED_TARGETS = {
    "mood_score", 
    "energy_level", 
    "focus_score", 
    "job_satisfaction"
}
# Must flip slope interpretation for these!
```

#### Feature 6: HTML Report Generation
**Challenge**: Console output is ugly; users want beautiful reports.

**Solution**: Comprehensive HTML with embedded charts

**Report Sections**:
1. **Header**: User info, timestamp, job category, data quality
2. **Key Metrics Dashboard**: Summary cards with risk status
3. **Predictions Chart**: Interactive Chart.js visualization
   - Daily targets: Line chart with confidence bands
   - Weekly targets: Bar chart with risk thresholds
4. **Behavioral Patterns**: 7-day trends
   - Sleep, work hours, exercise, screen time
   - Identifies problematic patterns
5. **Risk Assessment**: Color-coded factors
6. **Behavioral Interventions**: ⭐ NEW
7. **Recommendations**: Expandable cards with priority badges
8. **Contradictions**: Warning boxes with explanations
9. **Mental Health History**: Timeline with trend arrows
10. **Depression Scale Guide**: Interpretation box

**Styling**:
- Responsive design (mobile-friendly)
- Color-coded risk levels (red/yellow/green)
- Print-friendly CSS
- Chart.js for interactive visualizations
- No external dependencies (embedded CDN links)

**File Size**: 28-40KB per report (fully self-contained)

---

### Phase 5: Behavioral Interventions System ⭐ LATEST
**Document**: `BEHAVIORAL_INTERVENTIONS.md`

**Challenge**: Users asked "What should I DO?" after seeing predictions.

**Initial Attempt**: Use CVAE model
- CVAE exists: `models/saved/cvae_advisor.pt` (327KB)
- Problem: Trained on OLD burnout scale (0/1/2)
- Incompatible with new 8-target system
- **Decision**: Abandon CVAE, build data-driven system

**New Approach**: Evidence-Based Interventions

**Data Source**: `data/raw/interventions.csv` (332 cases)
- 7 intervention types: therapy (56), diet_coaching (51), exercise_plan (51), meditation (48), sick_leave (47), vacation (44), workload_cap (35)
- Measured outcomes: stress reduction, anxiety improvement, etc.

**System Design**:
```python
def generate_behavioral_interventions(profile):
    # Analyzes 7-day behavioral averages
    # Compares against evidence-based thresholds
    # Returns prioritized intervention list
```

**6 Intervention Categories**:

1. **Sleep Hours**: Target 7-9h
2. **Sleep Quality**: Target 7-9/10
3. **Physical Activity**: Target 30+ min/day
4. **Work-Life Balance**: Max 8-9h work
5. **Caffeine Management**: 200-400mg/day
6. **Social Connection**: 4-6 interactions/day

**Plus**: Stress Management for at-risk individuals

**Each Intervention Shows**:
```json
{
  "category": "Sleep Hours",
  "current": "4.1h/night",
  "target": "7-9h/night",
  "change_needed": "+3.4h per night",
  "priority": "HIGH",
  "evidence": "Users who increased sleep from <7h to 7-8h saw 31% stress reduction",
  "specific_actions": [
    "Set bedtime alarm for 3.4h earlier",
    "Create wind-down routine: dim lights 1h before bed",
    "Consistent sleep schedule (even weekends)"
  ]
}
```

**Priority System**:
- **HIGH**: Critical (sleep <6h, at-risk mental health)
- **MEDIUM**: Important improvements
- **LOW**: Preventive optimizations

**Integration**:
- Console: Displayed after predictions
- JSON: `behavioral_interventions` field
- HTML: Visual cards with current→target, evidence boxes, action checklists

**Testing**: 4 real users
- Marti (teacher, severe sleep deficit): 3 HIGH priority interventions
- Daniele (knowledge worker, high caffeine): 3 MEDIUM interventions
- Flaviana (film producer, healthy baseline): 2 interventions
- Peppe (student, underemployed): Multiple interventions

---

## 🏗️ Technical Architecture

### System Flow
```
┌─────────────────────────────────────────────────────────────┐
│                     USER INPUT                               │
│   Google Form → CSV (113 columns, 7 days of behavior)       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 PARSING & PREPROCESSING                      │
│  • parse_google_form_csv() - Maps columns to features       │
│  • Handles missing data                                      │
│  • Normalizes with StandardScaler                            │
│  Output: [batch, 7, 17] tensor                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              DEEP LEARNING PREDICTION                        │
│  Model: LSTM (2 layers, 128 hidden, 8 heads)                │
│  Input: 7 days × 17 features = [7, 17]                      │
│  Output: 8 predictions + confidence scores                   │
│  • Daily: stress, mood, energy, focus                        │
│  • Weekly: PSS, anxiety, depression, job_sat                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  ANALYSIS PIPELINE                           │
│  1. Risk Assessment (at-risk classification)                 │
│  2. Job Classification (keyword matching)                    │
│  3. Contradiction Detection (20+ patterns)                   │
│  4. Behavioral Interventions (evidence-based)                │
│  5. Recommendation Generation (job-specific)                 │
│  6. History Tracking (if available)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT GENERATION                          │
│  • Console: Formatted text summary                           │
│  • JSON: Complete profile data                               │
│  • HTML: Interactive report with charts                      │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture Details

**LSTM Encoder**:
```python
class MentalHealthPredictor(nn.Module):
    def __init__(self):
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=17,      # 17 behavioral features
            hidden_size=128,    # 128 hidden units
            num_layers=2,       # 2 stacked layers
            batch_first=True,
            dropout=0.2
        )
        
        # 8 prediction heads (one per target)
        self.heads = nn.ModuleDict({
            'stress_level': PredictionHead(128, 64),
            'mood_score': PredictionHead(128, 64),
            # ... 6 more heads
        })
    
    def forward(self, x):
        # x: [batch, 7, 17]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        features = h_n[-1]  # [batch, 128]
        
        # Generate predictions
        outputs = {}
        for target, head in self.heads.items():
            reg_out, cls_out = head(features)
            outputs[target] = {
                'value': reg_out,
                'at_risk': torch.sigmoid(cls_out)
            }
        
        return outputs
```

**Transformer Alternative**:
- Positional encoding for sequence order
- Multi-head self-attention (4 heads)
- Same prediction heads as LSTM
- Slightly better accuracy but slower

### Data Flow Example

**Input** (Google Form CSV):
```csv
Email,Job Title,Day 1 - Sleep Hours,Day 2 - Sleep Hours,...,Day 7 - Focus Score
marti@edu.it,Teacher,4.5,3.8,...,5.2
```

**After Parsing**:
```python
{
    'user_id': 'Marti',
    'job_title': 'Teacher',
    'behavioral_data': DataFrame([
        # 7 rows (days) × 17 columns (features)
        [4.5, 1.0, 9.2, 5, ...],  # Day 1
        [3.8, 2.0, 8.9, 4, ...],  # Day 2
        # ...
    ])
}
```

**After Normalization** (StandardScaler):
```python
tensor([
    [-1.2, -2.8, 0.4, 0.1, ...],  # Day 1 (z-scores)
    [-1.5, -2.5, 0.2, -0.2, ...], # Day 2
    # ...
])  # Shape: [7, 17]
```

**Model Prediction**:
```python
{
    'stress_level': {'value': 5.4, 'at_risk': False, 'confidence': 0.14},
    'mood_score': {'value': 5.9, 'at_risk': False, 'confidence': 0.70},
    'perceived_stress_scale': {'value': 21.8, 'at_risk': True, 'confidence': 0.97},
    # ... 5 more targets
}
```

**Final Profile** (UserProfile dataclass):
```python
UserProfile(
    user_id='Marti',
    job_category='education',
    predictions={...},
    risk_factors=['At-risk PSS: 21.8', 'Insufficient sleep: 4.1h'],
    positive_factors=['High job satisfaction: 7.0', 'Regular exercise'],
    contradictions=[{'pattern': 'MENTAL_HEALTH_POOR_FOUNDATION', ...}],
    recommendations=[{
        'category': 'SLEEP',
        'priority': 'HIGH',
        'advice': 'Improve sleep: consistent schedule, bedtime routine...',
        'rationale': 'Sleep is foundation of mental health',
        'job_specific': 'Teachers need emotional regulation capacity...'
    }],
    behavioral_interventions=[{
        'category': 'Sleep Hours',
        'current': '4.1h/night',
        'target': '7-9h/night',
        'change_needed': '+3.4h',
        'evidence': '31% stress reduction...',
        'specific_actions': [...]
    }],
    history_analysis=None,  # First assessment
    data_quality_score=0.888
)
```

---

## 🔧 Key Components

### 1. Column Mapping System
**Challenge**: Google Forms creates inconsistent columns

**Solution**: Flexible fuzzy matching
```python
def _find_feature_column(columns, feature_name, day_num):
    # Try multiple patterns
    patterns = [
        f"Day {day_num} - {feature_name.title()}",
        f"Day {day_num} - {feature_name.replace('_', ' ').title()}",
        f"day{day_num}_{feature_name}",
        # ... 10+ patterns
    ]
    
    for pattern in patterns:
        for col in columns:
            if pattern.lower() in col.lower():
                return col
    
    return None  # Will be imputed
```

**Handles**:
- Different capitalizations
- Underscores vs spaces
- Abbreviated names
- Missing days

### 2. Missing Data Imputation
**Strategy**: Use scaler mean (trained on 500K samples)

```python
def predict_user(model, X_scaled, scaler):
    # Replace NaN with feature mean from training data
    if np.isnan(X_scaled).any():
        for i in range(X_scaled.shape[1]):  # Each feature
            if np.isnan(X_scaled[:, i]).any():
                mean_val = scaler.mean_[i]
                X_scaled[:, i] = np.nan_to_num(X_scaled[:, i], nan=mean_val)
```

**Rationale**: Better than zero-imputation; preserves feature scale

### 3. At-Risk Classification
**Thresholds** (derived from dataset statistics):
```python
AT_RISK_THRESHOLDS = {
    'perceived_stress_scale': 20,    # PSS ≥20 = high stress
    'anxiety_score': 6,              # GAD-7 ≥6 = moderate anxiety
    'depression_score': 15,          # Top 5% in our scale
    'job_satisfaction': 5,           # <5 = dissatisfied
    'stress_level': 7,               # >7 = high daily stress
    'mood_score': 4,                 # <4 = low mood (inverted)
    'energy_level': 4,               # <4 = low energy (inverted)
    'focus_score': 4,                # <4 = poor focus (inverted)
}
```

**Note**: Inverted targets flip the comparison
```python
if target in INVERTED_TARGETS:
    at_risk = value < threshold  # Lower is worse
else:
    at_risk = value >= threshold  # Higher is worse
```

### 4. Confidence Scoring
**Method**: Inverse of prediction variance

```python
# During training, track prediction variance
# At inference:
confidence = 1 / (1 + variance)

# Interpretation:
# confidence > 0.7 = reliable
# 0.4 - 0.7 = moderate
# < 0.4 = uncertain
```

**Use Case**: Flag uncertain predictions in reports

### 5. Job Classification
**Algorithm**: Keyword scoring

```python
def classify_job_category(job_title: str) -> str:
    title_lower = job_title.lower()
    
    scores = {}
    for category, keywords in JOB_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in title_lower)
        scores[category] = score
    
    # Return category with highest score
    best_category = max(scores, key=scores.get)
    
    return best_category if scores[best_category] > 0 else "other"
```

**Example**:
- "Senior Software Engineer" → knowledge_worker (matches: software, engineer)
- "Film Producer" → creative (matches: film, producer)
- "High School Math Teacher" → education (matches: teacher)

---

## 🎯 The Build Process (Step-by-Step)

### Step 1: Foundation (Week 1)
✅ Downloaded dataset from Kaggle  
✅ EDA in Jupyter notebooks  
✅ Preprocessed data → parquet files  
✅ Trained baseline ML models (RF, LR, GB)  

**Output**: `data/processed/`, accuracy baselines

### Step 2: Deep Learning (Week 2)
✅ Built MLP classifier for burnout  
✅ Developed LSTM sequence model  
✅ Experimented with Transformer  
✅ Trained CVAE for interventions  

**Output**: `models/saved/lstm_sequence.pt`, 98% accuracy

### Step 3: Multi-Target Pivot (Week 3)
✅ Realized burnout classification too simple  
✅ Redesigned for 8 mental health targets  
✅ Built multi-task learning architecture  
✅ Trained `mental_health_lstm.pt`  

**Output**: Current production model

### Step 4: Google Form Integration (Week 4)
✅ Created flexible CSV parser  
✅ Mapped 113 columns to 17 features  
✅ Handled missing data gracefully  
✅ Tested with 4 real user submissions  

**Output**: Working profile generator

### Step 5: Feature Additions (Weeks 5-6)
✅ Risk assessment system  
✅ Job-specific advice (600+ lines)  
✅ Contradiction detection (20+ patterns)  
✅ HTML report generation with charts  

**Output**: Comprehensive profiling system

### Step 6: Advanced Features (Week 7)
✅ Mental health history tracking  
✅ Trend analysis with alerts  
✅ Fixed inverted targets bug  
✅ Added depression scale guide  

**Output**: Longitudinal capabilities

### Step 7: Behavioral Interventions (Week 8 - Latest)
✅ Evaluated CVAE (incompatible)  
✅ Built data-driven intervention system  
✅ Evidence from 332 intervention cases  
✅ Integrated into console/JSON/HTML  
✅ Tested with 4 real profiles  

**Output**: Complete system with actionable advice

---

## 🔗 How Everything Connects

### Data Pipeline
```
Kaggle Dataset (1.5M rows)
    ↓
download_data.py (Kaggle API)
    ↓
data/raw/*.csv
    ↓
preprocess.py (cleaning, normalization)
    ↓
data/processed/tabular_ml_ready.parquet
    ↓
train_mental_health.py (LSTM training)
    ↓
models/saved/mental_health_lstm.pt
```

### User Pipeline
```
User fills Google Form
    ↓
CSV export (113 columns)
    ↓
generate_profile.py --csv form.csv --user-id "John"
    ↓
parse_google_form_csv() [column mapping]
    ↓
predict_user() [LSTM inference]
    ↓
Risk Assessment, Contradictions, Interventions
    ↓
generate_html_report()
    ↓
reports/report_John_TIMESTAMP.html
```

### Code Dependencies
```
generate_profile.py (3089 lines)
    ├─ Uses: mental_health_lstm.pt (model weights)
    ├─ Uses: feature_scaler.joblib (normalization)
    ├─ Uses: feature_columns.json (metadata)
    │
    ├─ parse_google_form_csv()
    │   └─ _find_feature_column() (fuzzy matching)
    │
    ├─ predict_user()
    │   ├─ load_model()
    │   └─ torch inference
    │
    ├─ classify_job_category()
    │   └─ JOB_KEYWORDS (keyword matching)
    │
    ├─ detect_contradictions()
    │   └─ 20+ pattern checks
    │
    ├─ generate_behavioral_interventions()
    │   └─ Evidence from interventions.csv
    │
    ├─ generate_job_specific_advice()
    │   └─ 8 × 75 lines (600 total)
    │
    ├─ analyze_mental_health_trends()
    │   └─ Load past JSON files
    │
    ├─ generate_html_report()
    │   ├─ Chart.js integration
    │   └─ 2000+ lines HTML template
    │
    └─ save_profile_json()
        └─ Export for history
```

---

## 📝 Usage & Examples

### Basic Usage
```bash
# Single user
python scripts/generate_profile.py \
  --csv data/martina_7day.csv \
  --user-id "Marti" \
  --html \
  --output-dir reports/profiles

# All users in CSV
python scripts/generate_profile.py \
  --csv data/survey_responses.csv \
  --all-users \
  --html \
  --output-dir reports/batch

# With history tracking (if past assessments exist)
python scripts/generate_profile.py \
  --csv new_data.csv \
  --user-id "Marti" \
  --html \
  --lookback-days 90
```

### Output Files
```
reports/profiles/
├── profile_Marti_20251203_221204.json      # Machine-readable
└── report_Marti_20251203_221204.html       # Human-readable
```

### Example Output (Console)
```
================================================================================
MENTAL HEALTH PROFILE - User Marti
================================================================================
Generated: 2025-12-03 22:12
Job Category: Education
Data Quality: 88.8%

📊 PREDICTIONS (Next Day / End of Week)
--------------------------------------------------------------------------------

🔹 DAILY PREDICTIONS (Next Day):
  ✓ Stress Level        :   5.4 (confidence: 14%)
  ✓ Mood Score          :   5.9 (confidence: 70%)
  ✓ Energy Level        :   5.4 (confidence: 48%)
  ✓ Focus Score         :   5.3 (confidence: 39%)

🔹 WEEKLY PREDICTIONS (End of Week):
  ⚠️ Perceived Stress Scale        :  21.8 (confidence: 97%)
  ⚠️ Anxiety Score                 :   6.2 (confidence: 97%)
  ✓ Depression Score              :  11.2 (confidence: 98%)
  ✓ Job Satisfaction              :   7.0 (confidence: 100%)

⚠️  RISK FACTORS (3):
  1. At-risk perceived stress scale: 21.79
  2. At-risk anxiety score: 6.25
  3. Insufficient sleep: 4.1h/day (< 6h)

✓ POSITIVE FACTORS (4):
  1. Healthy depression score: 11.2
  2. High job satisfaction: 7.0/9
  3. Regular exercise: 62min/day
  4. Good social connections: 4.9 interactions/day

🎯 BEHAVIORAL INTERVENTIONS (Top 3):
--------------------------------------------------------------------------------
  1. 🔴 Sleep Hours (HIGH PRIORITY)
     Current: 4.1h/night  →  Target: 7-9h/night
     Change needed: +3.4h per night
     Evidence: 31% stress reduction in users who increased sleep
     Actions:
       • Set bedtime alarm for 3.4h earlier
       • Create wind-down routine (dim lights, no screens)
       • Consistent schedule even on weekends
  
  2. 🔴 Stress Management Techniques (HIGH PRIORITY)
     Current: At-risk stress/anxiety levels
     Evidence: 56 therapy cases showed symptom reduction
     Actions:
       • Consider professional support (therapy/counseling)
       • Daily mindfulness practice (10min minimum)
       • Emergency coping toolkit for acute stress

  3. 🟡 Sleep Quality (MEDIUM PRIORITY)
     Current: 1.0/10  →  Target: 7-9/10
     Change needed: +7 points improvement
     ...

🔍 CONTRADICTIONS DETECTED (1):
  1. 🔴 MENTAL HEALTH POOR FOUNDATION (HIGH)
     At-risk mental health but poor foundational self-care
     💭 Therapy works better with sleep/exercise/nutrition foundation
     💡 Prioritize: (1) 7-8h sleep, (2) 30min movement, (3) regular meals

💡 RECOMMENDATIONS (Education):
  1. 🔴 SLEEP - HIGH PRIORITY
     ▸ Improve sleep: consistent schedule, bedtime routine...
     ▸ WHY: Sleep is foundation of mental and physical health
     ▸ TEACHER-SPECIFIC: Need emotional regulation capacity for classroom management

================================================================================
```

### Example HTML Report
![HTML Report Structure]
- **Header**: User ID, job, timestamp, quality score
- **Dashboard Cards**: 8 predictions with risk badges
- **Interactive Charts**: Line/bar charts with Chart.js
- **Behavioral Patterns**: 7-day trends visualization
- **Risk Assessment**: Color-coded factors
- **Interventions Section**: Visual cards with current→target
- **Recommendations**: Expandable cards with priorities
- **Contradictions**: Warning boxes
- **History**: Timeline (if available)
- **Scale Guide**: Depression score interpretation

---

## 🎓 Key Learnings

### Technical Lessons
1. **Multi-task learning** improves generalization (shared representations)
2. **LSTM vs Transformer**: LSTM faster, Transformer slightly more accurate
3. **Imputation strategy**: Use training set mean, not zero
4. **Inverted targets**: Critical to handle correctly in multi-task setting
5. **Column mapping**: Flexibility beats rigid schemas for real-world data

### Design Lessons
1. **Job-specific advice** >>> generic recommendations
2. **Contradiction detection** reveals hidden patterns
3. **Evidence-based interventions** more credible than model-generated
4. **Visual reports** essential for user engagement
5. **Confidence scores** build trust in predictions

### Project Management
1. **Start simple**, iterate complexity
2. **Test with real data early** (Google Form integration)
3. **Pivot when needed** (burnout → multi-target)
4. **Documentation critical** (3089-line script needs clear structure)
5. **User feedback drives features** (interventions requested by users)

---

## 📊 Model Performance Summary

| Target | Accuracy | MSE | F1 Score | Notes |
|--------|----------|-----|----------|-------|
| Job Satisfaction | **98.5%** | 0.15 | 0.98 | Most predictable |
| Depression | **98%** | 2.3 | 0.96 | Clinical scale, stable |
| Anxiety | **97%** | 1.8 | 0.95 | GAD-7 scale |
| PSS | **97%** | 3.5 | 0.94 | Perceived stress |
| Mood Score | **85%** | 0.8 | 0.83 | Day-to-day variance |
| Energy Level | **62%** | 1.2 | 0.61 | Harder to predict |
| Focus Score | **52%** | 1.5 | 0.51 | Most variable |
| Stress Level | **52%** | 1.4 | 0.50 | Context-dependent |

**Overall**: Strong performance on stable weekly metrics, moderate on volatile daily metrics.

---

## 🚀 Future Enhancements

### Short-term
- [ ] Add more job categories (remote workers, gig economy)
- [ ] Improve confidence calibration
- [ ] Mobile-responsive HTML improvements
- [ ] PDF export option

### Medium-term
- [ ] Real-time intervention tracking (did user follow advice?)
- [ ] Personalized intervention efficacy (what works for this user?)
- [ ] Group/team analytics (workplace-level insights)
- [ ] Integration with wearables (objective sleep data)

### Long-term
- [ ] Causal inference (what changes will improve outcomes?)
- [ ] Reinforcement learning (optimal intervention sequencing)
- [ ] Federated learning (privacy-preserving multi-organization training)
- [ ] Clinical validation with real participants

---

## 📚 Repository Stats

- **Total Lines of Code**: ~15,000+
- **Main Script**: 3,089 lines (`generate_profile.py`)
- **Models Trained**: 13 (LSTM, Transformer, MLP, CVAE, etc.)
- **Dataset Size**: 1.5M+ records
- **Features**: 17 behavioral × 7 days = 119 inputs
- **Targets**: 8 mental health outcomes
- **Test Users**: 4 real profiles generated
- **Job Categories**: 8 + generic
- **Contradictions Detected**: 20+ patterns
- **Interventions**: 6 categories + stress management
- **HTML Report Size**: 28-40KB (self-contained)

---

## 🎯 Summary

This project evolved from a simple burnout classifier into a comprehensive mental health profiling system that:

1. **Predicts** 8 mental health outcomes from 7 days of behavior
2. **Analyzes** risk factors, contradictions, and trends
3. **Recommends** job-specific, evidence-based interventions
4. **Visualizes** everything in beautiful HTML reports
5. **Tracks** longitudinal mental health trajectories

Built with PyTorch, LSTM/Transformer architectures, and 1.5M+ training samples, it achieves 98%+ accuracy on stable metrics like job satisfaction and depression, with actionable insights backed by real intervention data.

**Core Innovation**: Moving beyond generic "eat better, sleep more" advice to context-aware, data-driven, specific behavioral changes tailored to individual job roles and risk profiles.

**Real-World Impact**: Early mental health intervention, workplace wellness programs, self-monitoring tools.

---

*Document created: December 3, 2025*  
*Repository: FDS-Project*  
*Status: Production-ready profiling system*
