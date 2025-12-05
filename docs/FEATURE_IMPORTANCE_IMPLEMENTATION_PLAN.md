# Feature Importance Implementation Plan

## 🎯 Goal
Use feature importance analysis to make predictions **explainable** and **actionable** for users.

## 📋 Implementation Phases

### Phase 1: Extract & Store Feature Importance (Completed in Kaggle)
- [x] Run feature importance analysis for all 5 targets
- [ ] Extract consensus rankings from notebook results
- [ ] Save to `data/processed/feature_importance_weights.json`
- [ ] Include population statistics (mean, std, min, max per feature)

**Output Structure:**
```json
{
  "stress_level": {
    "feature_weights": {
      "work_hours": 0.18,
      "sleep_quality": 0.15,
      "caffeine_mg": 0.12,
      ...
    },
    "population_stats": {
      "mean": 5.2,
      "std": 1.8
    }
  },
  "population_features": {
    "work_hours": {"mean": 8.5, "std": 2.1, "min": 0, "max": 16},
    ...
  }
}
```

### Phase 2: Create Explanation Engine
**New file:** `scripts/explain_predictions.py`

Functions:
- `calculate_feature_contributions(user_data, target, weights)` - Calculate each feature's impact
- `generate_explanation(user_data, prediction, target)` - Create natural language explanation
- `suggest_interventions(user_data, target, desired_change)` - Recommend specific actions
- `estimate_impact(current_value, target_value, feature, target)` - Predict outcome change

**Key Algorithm:**
```python
contribution = feature_weight × z_score × target_std

where:
  z_score = (user_value - population_mean) / population_std
  feature_weight = importance from RF analysis (0-1)
  target_std = std deviation of target variable
```

### Phase 3: Integrate into Profile Generation
**Modify:** `scripts/generate_profile.py`

Add sections:
1. **"Understanding Your Scores"** - Visual breakdown of contributions
2. **"What's Driving This?"** - Top 3-5 contributing factors per metric
3. **"Personalized Recommendations"** - Ranked by expected impact
4. **"If You Changed..."** - What-if scenarios for top factors

**HTML Template Addition:**
```html
<div class="driver-analysis">
  <h3>🔍 What's Driving Your Stress ({{stress_level}})</h3>
  
  <div class="contribution-bars">
    {{#each top_contributors}}
      <div class="contribution-bar {{direction}}">
        <span class="feature-name">{{feature_display_name}}</span>
        <div class="bar" style="width: {{bar_width}}%">
          <span class="value">{{contribution_display}}</span>
        </div>
      </div>
    {{/each}}
  </div>
  
  <div class="recommendations">
    <h4>💡 Top Recommendations:</h4>
    {{#each recommendations}}
      <div class="rec-card priority-{{priority}}">
        <strong>{{action}}</strong>
        <span class="impact">Expected: {{expected_change}}</span>
      </div>
    {{/each}}
  </div>
</div>
```

### Phase 4: Streamlit Demo Enhancement
**Modify:** `demo_app.py`

Add after prediction display:
```python
# Show explanations
st.subheader("🔍 Understanding Your Predictions")

for target in ['stress_level', 'burnout_score']:
    with st.expander(f"Why is your {target.replace('_', ' ')} {prediction[target]:.1f}?"):
        explanation = explain_prediction(user_input, prediction, target)
        
        # Visualization
        fig = plot_contribution_waterfall(explanation)
        st.plotly_chart(fig)
        
        # Text explanation
        st.markdown(generate_natural_language_explanation(explanation))
        
        # Recommendations
        st.markdown("**💡 What you can do:**")
        for rec in explanation['recommendations'][:3]:
            st.info(f"{rec['action']} → Expected impact: {rec['impact']:+.1f}")
```

### Phase 5: What-If Engine Integration
**Enhance:** `notebooks/05_what_if_engine.ipynb`

Auto-generate scenarios based on feature importance:
```python
# Instead of manual scenario creation
scenarios = auto_generate_scenarios(
    user_data=current_state,
    target='stress_level',
    goal='reduce',
    n_scenarios=5,
    feature_importance=importance_weights
)

# Prioritizes changes to:
# 1. High importance features
# 2. Features user can control (work_hours vs weather)
# 3. Realistic changes (not 0 sleep → 12 sleep)
```

### Phase 6: Testing & Validation
- [ ] Unit tests for contribution calculations
- [ ] Validate explanations with domain experts (psychologists)
- [ ] A/B test with users: with/without explanations
- [ ] Check if recommendations align with clinical guidelines

---

## 🎨 Visualization Ideas

### 1. Waterfall Chart (Contribution Breakdown)
```
Baseline (population avg): 5.0
  + Work Hours (11h):      +1.5
  + Caffeine (550mg):      +0.8
  + Sleep Quality (4/10):  +1.2
  - Exercise (45min):      -0.6
  = Your Stress:           7.9
```

### 2. Feature Importance Heatmap (in HTML report)
Show which features matter most for EACH metric for THIS user.

### 3. Radar Chart with Target Zones
```
Current values (red)
Optimal values (green)
Population average (gray)
```

### 4. Priority Matrix
```
High Impact, Easy Change → DO THIS FIRST
High Impact, Hard Change → Plan for long-term
Low Impact, Easy Change  → Quick wins
Low Impact, Hard Change  → Skip
```

---

## 🚀 Quick Win: Start with Phase 2

**Immediate value:** Create `explain_predictions.py` and integrate into existing profile generation.

**Why?**
- Already have all data (features, predictions, importance weights)
- No UI changes needed initially (just add to HTML reports)
- Users immediately benefit from understanding WHY predictions are what they are
- Can A/B test value before investing in Streamlit UI

**Next Steps:**
1. Export feature importance weights from Kaggle notebook
2. Create explanation engine script
3. Add 1-2 sections to HTML report template
4. Test with Sarah/Daniele/Maria profiles
5. Get feedback before expanding

---

## 💡 Advanced Ideas (Future)

### Counterfactual Explanations
"Your stress is 7.2. If you had slept 7h instead of 5h, it would be 6.0"

### Personalized Thresholds
Learn over time what levels work for THIS user (maybe 9h work is fine for them)

### Intervention Tracking
Track which recommendations users follow → measure actual vs predicted impact

### Causal Discovery
Use causal inference to distinguish correlation from causation in importance weights

---

## 📊 Success Metrics

- **User Engagement**: % of users who click "Why?" explanations
- **Action Rate**: % who implement top recommendation
- **Prediction Trust**: Survey - do users trust predictions more with explanations?
- **Behavioral Change**: Do explained predictions lead to better outcomes?

