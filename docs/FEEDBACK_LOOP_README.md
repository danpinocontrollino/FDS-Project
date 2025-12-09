# 📊 Feedback Loop System - Intervention Effectiveness Tracking

## Overview

The feedback loop system tracks the effectiveness of burnout interventions and learns what recommendations work best for different job types and users.

## Architecture

```
predict_burnout.py → logs intervention
                  ↓
feedback_loop.py   → tracks recommendations
                  ↓
User takes action / provides follow-up data
                  ↓
record_follow_up() → computes effectiveness
                  ↓
analyze_feedback.py → generates insights & updates
```

## How It Works

### 1. **Intervention Logging**
When `predict_burnout.py` generates predictions, it automatically logs:
- User ID and job type
- Current burnout score
- Recommendations provided
- Baseline metrics

```python
from feedback_loop import InterventionTracker

tracker = InterventionTracker(user_id="user@example.com", job_type="teacher")
intervention = tracker.log_intervention(
    burnout_score=75.0,
    recommendations=[{"feature": "stress_level", "suggested": 5}],
    advice_category="education",
    success_metrics={"stress_level": 8, "sleep_hours": 6}
)
```

### 2. **Follow-up Recording**
Users/admins record follow-up data (e.g., 1 week, 1 month later):

```python
tracker.record_follow_up(
    intervention_id="user@example.com_20251208_143022",
    follow_up_metrics={"stress_level": 6, "sleep_hours": 7},
    days_elapsed=7
)
```

### 3. **Effectiveness Computation**
Automatically calculates:
- **0.0**: Got worse
- **0.5**: No change
- **1.0**: Perfect improvement

Factors considered:
- Inverse metrics (lower is better): burnout, stress, anxiety
- Positive metrics (higher is better): sleep, mood, energy, satisfaction

### 4. **Analysis & Learning**
Generates insights per job type:
- Which recommendations have highest success rate?
- Which features should be emphasized?
- Model weight adjustments for fine-tuning

## Data Storage

All feedback data stored in `/data/feedback/`:

```
/data/feedback/
├── interventions.jsonl      # One intervention per line (append-only)
├── effectiveness.json       # Summary report
└── model_updates.json       # Suggested weight changes
```

## Usage

### Log Interventions (Automatic)
Happens automatically when running predict_burnout.py:
```bash
python scripts/predict_burnout.py --csv responses.csv
```

### Record Follow-up Data
Via API or scripts:
```python
tracker = InterventionTracker("user@example.com", "software_engineer")
tracker.record_follow_up(
    intervention_id="...",
    follow_up_metrics={...},
    days_elapsed=7
)
```

### Analyze Feedback
```bash
# Full summary
python scripts/analyze_feedback.py

# Specific user
python scripts/analyze_feedback.py --user user@example.com

# Job type analysis
python scripts/analyze_feedback.py --job teacher

# Generate full report
python scripts/analyze_feedback.py --report
```

## Example Output

```
============================================================================
INTERVENTION EFFECTIVENESS SUMMARY
============================================================================

🎯 TEACHER
   Total interventions: 12
   Success rate: 66.7%
   Most effective recommendations:
      • sleep_hours: 83.3% (5/6)
      • exercise_minutes: 75.0% (3/4)
      • stress_level: 50.0% (2/4)

🎯 SOFTWARE_ENGINEER
   Total interventions: 8
   Success rate: 62.5%
   Most effective recommendations:
      • break_flexibility: 100.0% (2/2)
      • work_hours: 66.7% (2/3)

============================================================================
```

## Benefits

1. **Data-Driven Decisions**: Know what actually works
2. **Personalization**: Tailor recommendations by job type
3. **Continuous Learning**: Update model as more data arrives
4. **Accountability**: Track whether interventions help
5. **A/B Testing**: Compare different recommendation strategies
6. **Early Warning**: Identify when strategies stop working

## Integration Points

### predict_burnout.py
- Automatically logs interventions
- Calls `log_intervention_for_feedback()` on each prediction

### generate_profile.py
- Can log interventions when generating profiles
- Tracks multi-target mental health outcomes

### Model Training
- Use `generate_model_update_weights()` to fine-tune LSTM
- Feature weights adjusted based on real-world effectiveness

## Future Enhancements

1. **Real-time Feedback Dashboard**
   - Web UI to record follow-ups
   - Visual effectiveness tracking
   - Recommendation quality metrics

2. **Adaptive Recommendations**
   - Use feedback to rank recommendations in real-time
   - Show effectiveness % next to each suggestion
   - "This worked for 75% of similar users"

3. **A/B Testing Framework**
   - Test different intervention strategies
   - Control vs treatment groups
   - Statistical significance testing

4. **Temporal Analysis**
   - Track how effects decay over time
   - Identify seasonal patterns
   - Predict when intervention will be needed again

5. **Causal Inference**
   - Go beyond correlation (what CVAE learns)
   - Identify true causes of burnout recovery
   - Avoid spurious recommendations

## Technical Details

### Effectiveness Metric
```
effectiveness = (mean_improvement + 1) / 2

where:
- improvement = (current - baseline) / baseline
- For inverse metrics: improvement = (baseline - current) / baseline
- Range: [0, 1]
```

### Success Rate (per Job Type)
```
success_rate = successful_interventions / total_interventions
successful = effectiveness > 0.6
```

### Model Weight Updates
```
weight_delta = success_rate - 0.5

if success_rate > 0.7:  weight += 0.3
if success_rate > 0.5:  weight += 0.1
if success_rate < 0.3:  weight -= 0.2
```

## Example Workflow

```python
# 1. User fills form → predict_burnout.py runs
from feedback_loop import InterventionTracker

tracker = InterventionTracker("sarah@example.com", "software_engineer")
intervention = tracker.log_intervention(
    burnout_score=72,
    recommendations=[
        {"feature": "work_hours", "current": 9, "suggested": 8},
        {"feature": "break_flexibility", "current": 4, "suggested": 6},
    ],
    advice_category="knowledge_work",
    success_metrics={"work_hours": 9, "stress_level": 8}
)
# Returns: {"intervention_id": "sarah@example.com_20251208_143022", ...}

# 2. One week later, user fills follow-up form
tracker.record_follow_up(
    intervention_id="sarah@example.com_20251208_143022",
    follow_up_metrics={"work_hours": 8.5, "stress_level": 6},
    days_elapsed=7
)
# Computes: effectiveness = 0.75 (significant improvement!)

# 3. Analyze what worked
from feedback_loop import CohortAnalyzer
analyzer = CohortAnalyzer()
job_stats = analyzer.analyze_by_job_type()
print(job_stats["software_engineer"]["recommendations"])
# Shows: work_hours had 80% success rate, break_flexibility had 100%
```

## Contributing

To add new features to the feedback loop:

1. Update `InterventionTracker` for new fields
2. Extend `_compute_effectiveness()` for new metrics
3. Add analysis methods to `CohortAnalyzer`
4. Update `analyze_feedback.py` CLI options

---

**Status**: ✅ Fully implemented and ready for production
**Last Updated**: December 2025
**Maintainer**: FDS Project Team
