# Burnout Assessment Form - Version 2.0

## New Questions to Add

Based on research, user feedback, and **features proven useful in our training dataset**, we're adding these important dimensions:

---

## 1. Social Interaction Quality (NEW)

**Current:** We only ask about number of social interactions
**New:** Quality matters more than quantity

### New Questions:

```
Q: How would you rate the quality of your social interactions? (1-10)
   1 = Superficial/Draining
   10 = Deep/Energizing

Q: How often do you feel lonely or isolated?
   - Never
   - Rarely (few times a month)
   - Sometimes (weekly)
   - Often (multiple times per week)
   - Always

Q: Do you have someone you can talk to about personal problems?
   - Yes, multiple people
   - Yes, one person
   - Not really
   - No
```

---

## 2. Work Environment (NEW)

**Why:** Remote workers have different stressors than office workers

### New Questions:

```
Q: What is your primary work arrangement?
   - Fully remote (work from home)
   - Hybrid (mix of home and office)
   - Fully in-office
   - Field work / On-site visits
   - Varies significantly

Q: How would you describe your workspace?
   - Private office
   - Shared office (2-4 people)
   - Open floor plan
   - Home office (dedicated room)
   - Home office (shared space)
   - No dedicated workspace

Q: How distracting is your work environment? (1-10)
   1 = Very quiet and focused
   10 = Constant interruptions

Q: Do you have ergonomic setup? (desk, chair, monitor height)
   - Yes, fully ergonomic
   - Partially
   - No
```

---

## 3. Job Type / Industry Context (NEW)

**Why:** A developer's 8h screen time is different from leisure scrolling

### New Questions:

```
Q: What best describes your job type?
   - Knowledge work (software, research, writing)
   - Creative work (design, marketing, content)
   - Healthcare / Medical
   - Education / Teaching
   - Customer service / Support
   - Management / Leadership
   - Manual / Physical labor
   - Sales / Business development
   - Finance / Accounting
   - Other

Q: Does your job require extended screen time?
   - Yes, 6+ hours is essential for my work
   - Yes, 4-6 hours
   - Moderate, 2-4 hours
   - Minimal, under 2 hours
   - Not at all

Q: Can you take breaks when you need them?
   - Yes, fully flexible
   - Mostly, with some constraints
   - Sometimes, depends on workload
   - Rarely, breaks are limited
   - Never, continuous work required
```

---

## 4. Recovery & Boundaries (NEW)

**Why:** Understanding recovery capacity is key to burnout prevention

### New Questions:

```
Q: Do you have a clear boundary between work and personal time?
   - Yes, strict separation
   - Mostly, with occasional overlap
   - Somewhat blurred
   - Very blurred
   - No boundary at all

Q: How often do you check work emails/messages outside work hours?
   - Never
   - Rarely (emergencies only)
   - Sometimes (few times per week)
   - Often (daily)
   - Constantly

Q: When was your last vacation (3+ consecutive days off)?
   - Within last month
   - 1-3 months ago
   - 3-6 months ago
   - 6-12 months ago
   - Over a year ago

Q: How well do you recover after a stressful day? (1-10)
   1 = Takes days to recover
   10 = Fully recovered by next morning
```

---

## 5. Lifestyle & Physical Health (NEW - Dataset Aligned)

**Why:** These features are in our training dataset and predict burnout well

### New Questions:

```
Q: How many meaningful social interactions did you have today?
   (Count conversations, calls, quality time with others)
   - None
   - 1-2
   - 3-5
   - 6-10
   - More than 10

Q: How much time did you spend outdoors today (in minutes)?
   - None
   - 1-15 minutes
   - 15-30 minutes
   - 30-60 minutes
   - 1-2 hours
   - More than 2 hours

Q: How would you rate your diet today? (1-10)
   1 = Very poor (fast food, skipped meals, no vegetables)
   5 = Average (some balanced meals)
   10 = Excellent (balanced, nutritious, well-timed meals)

Q: How satisfied are you with your job? (1-10) [Weekly question]
   1 = Extremely dissatisfied
   5 = Neutral
   10 = Extremely satisfied
```

---

## 6. One-Time Profile Questions (NEW)

**Why:** Chronotype affects optimal work patterns

### New Questions (ask once during onboarding):

```
Q: What is your natural sleep pattern (chronotype)?
   - Morning person (naturally wake early, peak energy in morning)
   - Night owl (naturally stay up late, peak energy in evening)
   - Intermediate (flexible, no strong preference)

Q: Have you ever been diagnosed with or treated for anxiety, depression, or burnout?
   - Yes, currently receiving treatment
   - Yes, in the past but not currently
   - Never formally, but have experienced symptoms
   - No, never
   [Note: This is used only to personalize recommendations and is kept confidential]
```

---

## Updated Feature List for Model Training

### Existing Features (15):
- sleep_hours
- sleep_quality
- work_hours
- stress_level
- mood_score
- energy_level
- focus_score
- exercise_minutes
- caffeine_mg
- screen_time_hours
- meetings_count
- alcohol_units
- steps_count
- commute_minutes
- emails_received

### New Features to Add (15) - Including Dataset-Aligned:
- social_quality (1-10)
- loneliness_score (0-4 scale)
- work_arrangement (categorical: remote/hybrid/office)
- workspace_type (categorical)
- environment_distractions (1-10)
- job_requires_screen (boolean or hours)
- break_flexibility (1-5 scale)
- work_life_boundary (1-5 scale)
- after_hours_checking (0-4 scale)
- recovery_ability (1-10)

### Dataset-Aligned Features (HIGH PRIORITY - matches training data):
- **social_interactions** (count per day, 0-20)
- **outdoor_time_minutes** (minutes spent outdoors, 0-300)
- **diet_quality** (1-10 scale)
- **job_satisfaction** (1-10 scale, weekly)
- **chronotype** (morning_lark / night_owl / intermediate) - ONE-TIME profile

---

## Implementation Notes

### For the Google Form:
1. Add new sections for "Social Wellbeing" and "Work Environment"
2. Make job type question conditional - adjust screen time interpretation
3. Add logic: if job requires screen time, weight leisure screen time differently

### For the Model:
1. Update preprocessing to handle new categorical features
2. Retrain LSTM/Transformer with expanded feature set
3. Consider job_type as context for screen_time_hours interpretation
4. Add interaction terms: job_screen_req × screen_time

### For the Report:
1. Add "Work Environment" section to metrics
2. Adjust advice based on job type (e.g., don't suggest "reduce screen time" for developers)
3. Add social health indicators

---

## Sample Updated CSV Header (34 columns total)

```csv
timestamp,name,stress_level,mood_score,energy_level,focus_score,sleep_quality,sleep_hours,work_hours,commute_minutes,screen_time_hours,exercise_minutes,meetings_count,emails_received,work_pressure,caffeine_mg,alcohol_units,steps_count,social_quality,loneliness_score,work_arrangement,workspace_type,environment_distractions,job_type,job_requires_screen,break_flexibility,work_life_boundary,after_hours_checking,recovery_ability,social_interactions,outdoor_time_minutes,diet_quality,job_satisfaction,chronotype
```

---

## Feature Summary

| Category | Feature | Type | Range | Priority |
|----------|---------|------|-------|----------|
| **Core (15)** | sleep_hours, sleep_quality, work_hours, stress_level, mood_score, energy_level, focus_score, exercise_minutes, caffeine_mg, alcohol_units, steps_count, screen_time_hours, meetings_count, commute_minutes, emails_received | Numeric | Various | ⭐⭐⭐ |
| **Social** | social_quality | 1-10 | Quality of interactions | ⭐⭐ |
| **Social** | loneliness_score | 0-4 | How often lonely | ⭐⭐ |
| **Social** | social_interactions | 0-20 | Count per day | ⭐⭐⭐ |
| **Environment** | work_arrangement | Categorical | remote/hybrid/office | ⭐⭐ |
| **Environment** | workspace_type | Categorical | home/shared/private/open | ⭐⭐ |
| **Environment** | environment_distractions | 1-10 | How distracting | ⭐⭐ |
| **Job** | job_type | Categorical | Various | ⭐⭐ |
| **Job** | job_requires_screen | Categorical | Hours needed | ⭐⭐ |
| **Job** | job_satisfaction | 1-10 | Weekly rating | ⭐⭐⭐ |
| **Boundaries** | break_flexibility | 1-5 | Can take breaks | ⭐⭐ |
| **Boundaries** | work_life_boundary | 1-5 | Clarity of separation | ⭐⭐ |
| **Boundaries** | after_hours_checking | 0-4 | How often | ⭐⭐ |
| **Recovery** | recovery_ability | 1-10 | How well you recover | ⭐⭐⭐ |
| **Lifestyle** | outdoor_time_minutes | 0-300 | Minutes outdoors | ⭐⭐⭐ |
| **Lifestyle** | diet_quality | 1-10 | Nutrition quality | ⭐⭐⭐ |
| **Profile** | chronotype | Categorical | morning/night/intermediate | ⭐⭐ |

**Priority Legend:**
- ⭐⭐⭐ = Directly matches training data (highest predictive value)
- ⭐⭐ = Valuable for personalized advice
