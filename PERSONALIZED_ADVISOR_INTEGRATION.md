# 🎯 Personalized Advisor Integration

## What Was Added

The **Personalized Advisor** is now fully integrated into the main pipeline.

### New Step: 3.5 🎯
Between profile generation and report creation, the pipeline now runs:

```
Step 1: Parse CSV
    ↓
Step 2: Burnout Prediction
    ↓
Step 3: Mental Health Profile
    ↓
Step 3.5: ✨ PERSONALIZED ADVISOR ✨ (NEW!)
         └─ Analyzes YOUR specific metrics
         └─ Identifies ROOT CAUSES for YOU
         └─ Prioritizes recommendations by impact
         └─ Creates ultra-specific advice with YOUR numbers
    ↓
Step 4: Generate HTML Report
    ↓
Step 5: Save Outputs (now includes advice JSON!)
```

---

## How It Works

### 1. **Analyzes Your Metrics**
```
Sarah's Metrics:
├─ Sleep: 5.5h/night
├─ Caffeine: 450mg/day
├─ Work: 11h/day
├─ Stress: 8/10
└─ Burnout: 85/100
```

### 2. **Identifies Root Causes FOR YOU**
```
NOT: "You have high stress"
YES: "Your caffeine (450mg) disrupts your sleep (5.5h)
      causing low energy, forcing longer work hours,
      increasing stress - a vicious cycle"
```

### 3. **Prioritizes by Impact**
```
#1 Sleep Deprivation - Impact 0.92 (CRITICAL for you)
#2 Excessive Work Hours - Impact 0.89 (Maintains #1)
#3 Caffeine Cycle - Impact 0.78 (Symptom of #1)
```

### 4. **Creates Specific Actions**
```
NOT: "Sleep more"
YES: "FIRST: Stop caffeine after 2pm (you have 450mg/day).
      THEN: Set alarm for consistent wake 6am.
      Enforce bedtime 10:30pm (miss 2h nightly).
      
      WHY: You're missing 2 hours sleep. This is your
           PRIMARY lever for stress reduction.
      
      TIMELINE: 3-5 days initial, 2 weeks full effect
      
      SUCCESS RATE: 92% of people with your profile succeed"
```

---

## Output Files

The pipeline now generates 4 JSON files (instead of 3):

```
reports/
├── report_sarah@example.com_20251209_143022.html     ← HTML Report
├── burnout_sarah@example.com_20251209_143022.json    ← Burnout predictions
├── profile_sarah@example.com_20251209_143022.json    ← Mental health profile
└── advice_sarah@example.com_20251209_143022.json     ← ✨ PERSONALIZED ADVICE (NEW!)
```

---

## The Advice JSON Structure

```json
[
  {
    "priority": 1,
    "impact_score": 0.92,
    "root_cause": "Sleep deprivation (5.5h/night)",
    "your_current_value": 5.5,
    "healthy_target": 7.5,
    "behavioral_chain": "Less sleep → Lower stress resilience → More stress perception → Harder to sleep (vicious cycle)",
    "specific_action": "FIRST: Stop caffeine after 2pm (you have 450mg/day). THEN: Set alarm for consistent wake time...",
    "why_it_matters": "You're missing ~2 hours of sleep each night. This is your PRIMARY lever for reducing stress and improving mood.",
    "expected_outcome": "Adding 2 hours sleep will reduce stress by ~25-30% and boost energy significantly.",
    "difficulty_level": "Medium",
    "time_to_see_results": "3-5 days for initial improvement, 2 weeks for full effect",
    "success_rate": 0.92,
    "related_problems": ["reduced_stress_resilience", "low_mood", "poor_focus"]
  },
  {
    "priority": 2,
    "impact_score": 0.89,
    "root_cause": "Excessive work hours (11.0h/day)",
    "your_current_value": 11.0,
    "healthy_target": 8.0,
    ...
  }
]
```

---

## Key Differences from Generic Advice

| Aspect | Generic | Personalized |
|--------|---------|--------------|
| **Diagnosis** | "You have low sleep" | "Caffeine (YOUR 450mg) disrupts YOUR sleep (5.5h)" |
| **Prioritization** | All important | #1 Sleep (92% impact FOR YOU) |
| **Action** | "Sleep more" | "Stop caffeine 2pm, sleep 10:30pm, wake 6am" |
| **Numbers** | None | YOUR numbers (5.5→7.5, 450→200) |
| **Timeline** | Vague | "3-5 days then 2 weeks" |
| **Success Rate** | None | "92% of people like you succeed" |

---

## Usage in Code

```python
from main_pipeline import run_pipeline

result = run_pipeline(
    csv_path="test_responses.csv",
    user_email="sarah@example.com",
    output_dir="reports/"
)

# Access the personalized advice
top_priority = result["top_priority_advice"]
print(f"What to fix FIRST: {top_priority['root_cause']}")
print(f"Action: {top_priority['specific_action']}")
print(f"Timeline: {top_priority['time_to_see_results']}")

# All advice
all_advice = result["personalized_advice"]
for advice in all_advice:
    print(f"{advice['priority']}. {advice['root_cause']}")
```

---

## Command Line

```bash
# Everything still works the same!
python scripts/main_pipeline.py --csv test_responses.csv --user sarah@example.com

# Now generates 4 files instead of 3
# Including advice_sarah@example.com_TIMESTAMP.json
```

---

## What's Personalized

✅ **Root Cause Analysis** - YOUR specific problem combination  
✅ **Impact Scoring** - What hurts YOU most  
✅ **Prioritization** - Fix THIS first (for YOU)  
✅ **Specific Actions** - WITH YOUR NUMBERS  
✅ **Behavioral Chains** - How YOUR problems connect  
✅ **Timeline** - How long FOR YOUR PROFILE  
✅ **Success Rates** - % success for people like YOU  
✅ **Related Problems** - What else this fixes for YOU  

---

## Files Modified

- ✏️ `scripts/main_pipeline.py` - Added Step 3.5 + integration
- ✏️ `scripts/validate_pipeline.py` - Added personalized_advisor.py to checks
- 🆕 `scripts/personalized_advisor.py` - NEW module (1000+ lines)

---

## Next Steps (Optional)

1. **Dashboard** - Visualize advice over time
2. **Email Report** - Send advice + report by email
3. **Web UI** - Interactive advice interface
4. **A/B Testing** - Compare advice effectiveness
5. **Historical Tracking** - Show which advice worked before

---

**The advisor is now part of the core pipeline.** Every time you run it, users get ultra-specific, personalized recommendations based on THEIR metrics. 🎯✨
