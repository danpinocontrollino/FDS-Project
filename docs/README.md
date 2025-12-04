# FDS Project Documentation

Complete documentation for the Mental Health Profiling System.

## 📚 Documentation Files

### 1. [COMPLETE_PROJECT_EXPLANATION.md](./COMPLETE_PROJECT_EXPLANATION.md)
**Comprehensive deep-dive** into the entire project (15,000+ words)

**Contents**:
- Project overview & goals
- Dataset description (1.5M+ records)
- Development journey (9 weeks)
- Technical architecture
- Key components breakdown
- Code walkthroughs
- Usage examples
- Performance metrics
- Future enhancements

**Best for**: Understanding the full story, technical decisions, and implementation details.

---

### 2. [VISUAL_ARCHITECTURE.md](./VISUAL_ARCHITECTURE.md)
**Visual diagrams** and system architecture

**Contents**:
- ASCII art system diagrams
- Data flow visualization
- Component hierarchy
- Development timeline
- Performance summary tables
- Command quick reference
- Repository statistics

**Best for**: Getting a high-level overview, understanding system flow, and visual learners.

---

### 3. [../BEHAVIORAL_INTERVENTIONS.md](../BEHAVIORAL_INTERVENTIONS.md)
**Detailed documentation** of the behavioral interventions system

**Contents**:
- Evidence-based recommendations
- Intervention categories (6 + conditional)
- Integration details (console/JSON/HTML)
- Implementation guide
- Design rationale
- Example outputs
- Future enhancements

**Best for**: Understanding the latest feature addition (data-driven interventions).

---

## 🚀 Quick Start

### For New Developers
1. Read: [VISUAL_ARCHITECTURE.md](./VISUAL_ARCHITECTURE.md) (30 min)
   - Understand system overview
   - See data flow diagrams
   - Review component breakdown

2. Read: [COMPLETE_PROJECT_EXPLANATION.md](./COMPLETE_PROJECT_EXPLANATION.md) (2 hours)
   - Deep technical details
   - Development journey
   - Code explanations

3. Explore: `/scripts/generate_profile.py` (production code)
   - 3,089 lines with inline comments
   - Main profiling system

### For Users
1. Read: Main [README.md](../README.md)
   - Project overview
   - Installation instructions
   - Basic usage

2. Read: [BEHAVIORAL_INTERVENTIONS.md](../BEHAVIORAL_INTERVENTIONS.md)
   - What interventions are
   - How to interpret recommendations
   - Evidence sources

### For Researchers
1. Read: [COMPLETE_PROJECT_EXPLANATION.md](./COMPLETE_PROJECT_EXPLANATION.md)
   - Model architecture
   - Training methodology
   - Performance metrics
   - Dataset analysis

2. Explore: `/notebooks/`
   - `01_eda.ipynb`: Exploratory analysis
   - `02_baseline_models.ipynb`: Classical ML
   - `03_deep_learning_mlp.ipynb`: MLP experiments
   - `04_lstm_sequence_model.ipynb`: LSTM development

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Code** |
| Total Python Files | 15 |
| Main Script Lines | 3,089 |
| Total Lines of Code | ~15,000+ |
| **Data** |
| Dataset Records | 1,568,337 |
| Training Samples | ~500,000 |
| Test Users Profiled | 4 |
| **Models** |
| Models Trained | 13 |
| Best Model | LSTM (1.2 MB) |
| Best Accuracy | 98.5% (job satisfaction) |
| **Features** |
| Input Features | 17 × 7 days = 119 |
| Output Targets | 8 mental health outcomes |
| Job Categories | 8 + generic |
| Contradiction Patterns | 20+ |
| Interventions | 6 + conditional |

---

## 🎯 Key Features

### 1. Multi-Target Prediction
Predicts 8 mental health outcomes simultaneously:
- **Daily**: stress, mood, energy, focus
- **Weekly**: PSS, anxiety, depression, job satisfaction

### 2. Job-Specific Advice
Tailored recommendations for 8 job categories:
- Knowledge Worker, Healthcare, Education, Creative
- Management, Service, Manual Labor, Sales/Marketing

### 3. Contradiction Detection
Identifies 20+ patterns of contradictory responses revealing:
- Survey mistakes
- Cognitive dissonance
- Hidden issues

### 4. Behavioral Interventions ⭐ NEW
Evidence-based recommendations from 332 intervention cases:
- Sleep, exercise, work-life balance
- Caffeine, social connection, stress management
- Current→Target with specific actions

### 5. HTML Reports
Beautiful, interactive reports with:
- Chart.js visualizations
- Color-coded risk assessments
- Expandable recommendation cards
- 28-40 KB self-contained files

### 6. History Tracking
Longitudinal mental health analysis:
- Trend detection (improving/declining/stable)
- Alert generation (deterioration warnings)
- 90-day lookback capabilities

---

## 🏗️ System Architecture

```
User Input (Google Form CSV)
    ↓
Parsing & Preprocessing
    ↓
LSTM Prediction (8 targets)
    ↓
Analysis Pipeline
├── Risk Assessment
├── Job Classification
├── Contradiction Detection
├── Behavioral Interventions
├── Recommendations
└── History Tracking
    ↓
Output Generation
├── Console (formatted text)
├── JSON (machine-readable)
└── HTML (interactive report)
```

---

## 📖 Related Files

### Root Directory
- [README.md](../README.md) - Project overview
- [requirements.txt](../requirements.txt) - Python dependencies
- [BEHAVIORAL_INTERVENTIONS.md](../BEHAVIORAL_INTERVENTIONS.md) - Intervention system docs

### Scripts
- [generate_profile.py](../scripts/generate_profile.py) - Main profiling system (3,089 lines)
- [train_mental_health.py](../scripts/train_mental_health.py) - Model training
- [preprocess.py](../scripts/preprocess.py) - Data preprocessing

### Notebooks
- [01_eda.ipynb](../notebooks/01_eda.ipynb) - Exploratory analysis
- [02_baseline_models.ipynb](../notebooks/02_baseline_models.ipynb) - Classical ML
- [03_deep_learning_mlp.ipynb](../notebooks/03_deep_learning_mlp.ipynb) - MLP experiments
- [04_lstm_sequence_model.ipynb](../notebooks/04_lstm_sequence_model.ipynb) - LSTM development

---

## 🤝 Contributing

This is a university project for **Foundations of Data Science**.

**Team Members**: Working in dev container environment.

**Current Status**: Production-ready profiling system with behavioral interventions.

---

## 📝 License

University project - check main README for licensing details.

---

*Documentation index created: December 3, 2025*
*Repository: FDS-Project*
*Status: Complete documentation suite*
