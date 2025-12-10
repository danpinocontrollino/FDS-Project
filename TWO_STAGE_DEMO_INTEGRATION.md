# Two-Stage Pipeline Demo Integration

## Overview
Successfully integrated the two-stage hybrid prediction pipeline into the Streamlit demo app with interactive exploration of 598 real predictions.

## What Was Added

### New Demo Section: "Two-Stage Hybrid Pipeline"
Located after "Data Quality Insights" and before "Model Comparison" in the main demo app.

### Features Implemented

#### 1. **Interactive Student Selection**
- Choose from 10 StudentLife students (u00-u10)
- View timeline of predictions for each student
- Navigate through dates with dropdown selector

#### 2. **Date-by-Date Breakdown**
For each selected date, users can see:

**Stage 1: Behavioral Forecasting**
- 5 behavioral predictions with uncertainties:
  - 😴 Sleep hours (±uncertainty)
  - 🏃 Exercise minutes (±uncertainty)
  - 📱 Screen time hours (±uncertainty)
  - 👥 Social interactions (±uncertainty)
  - 🚶 Steps count (±uncertainty)

**Stage 2: Mental Health Inference**
- 8 mental health predictions with color-coded risk indicators:
  - 🔴/🟠/🟢 Stress Level (0-10)
  - 🔴/🟠/🟢 Mood Score (0-10)
  - 🔴/🟠/🟢 Anxiety Score (0-21)
  - 🔴/🟠/🟢 Energy Level (0-10)
  - 🔴/🟠/🟢 Perceived Stress Scale (0-40)
  - 🔴/🟠/🟢 Depression Score (0-27)
  - 🔴/🟠/🟢 Focus Score (0-10)
  - 🔴/🟠/🟢 Job Satisfaction (0-10)

**Error Propagation Analysis**
- Total Stage 1 uncertainty
- Confidence impact explanation
- Expandable comparison with direct prediction (placeholder for future)

#### 3. **Time Series Navigation**
- Previous/Next day buttons
- Progress indicator (Day X of Y)
- Easy temporal exploration

#### 4. **Visualization Tabs**
Four comprehensive visualizations (if available):
- 🌊 **Uncertainty Waterfall**: How confidence degrades through pipeline
- 📉 **Error Propagation**: Stage 1 uncertainty vs Stage 2 predictions (6 subplots)
- 📊 **Behavioral Trends**: Time series with confidence bands
- 🎯 **Summary Dashboard**: Complete overview with architecture diagram

Each tab includes interpretation guidance explaining what the charts show.

#### 5. **Key Research Insights**
Automatically displayed summary of findings:
- Error compounding effects (~20-30% confidence reduction)
- Real vs synthetic pattern differences
- Distribution mismatch between datasets
- Transparency benefits of cascaded systems
- Practical use cases for hybrid pipelines

#### 6. **Limitations Section**
Honest assessment of current limitations:
- Placeholder uncertainties (need Bayesian quantification)
- Lack of ground truth mental health labels
- Distribution mismatch between Stage 1 and Stage 2 training data
- Correlation vs causation assumptions

## Technical Implementation

### File Changes
- **demo_app.py**: Added `render_two_stage_pipeline_demo()` function (~300 lines)
- Located before `render_model_comparison_viewer()` section
- Integrated into main flow with proper error handling

### Data Loading
- Reads from `models/saved/two_stage_predictions.json`
- Handles both nested (metadata + predictions) and flat formats
- Graceful degradation if file not found (shows "Coming Soon" message)

### User Experience
- Collapsed by default (expander) - doesn't clutter main demo
- Can be expanded for deep dive into results
- Self-contained - doesn't interfere with other demo sections
- Clear visual hierarchy with color coding and emojis

## How to Use in Presentations

### Demo Flow Suggestion:

1. **Start with normal demo** - Show single-stage predictions with user input
2. **Scroll to Two-Stage section** - Expand the expander
3. **Select a student** (e.g., u00) - Show real StudentLife data being used
4. **Walk through a specific date**:
   - "Stage 1 predicts behaviors from sensors"
   - "Stage 2 infers mental health from predicted behaviors"
   - "Notice the uncertainties compound"
5. **Show visualizations** - Open tabs to display charts
6. **Highlight insights** - Read key findings and limitations
7. **Compare approaches** - Contrast with direct prediction approach

### Key Talking Points:

- **"This is a hybrid approach"** - Combines real sensor patterns with synthetic mental health knowledge
- **"Error propagation is explicit"** - We can track how uncertainty flows through the system
- **"Real-world applicable"** - When you have abundant sensor data but sparse mental health surveys
- **"Transparent limitations"** - We clearly state what's placeholder vs validated
- **"598 real predictions"** - Not toy example, substantial dataset

## File Locations

### Required Files:
- `models/saved/two_stage_predictions.json` - 598 predictions (24,530 lines)
- `models/saved/stage1_behavioral_forecasting.pt` - Trained Stage 1 model

### Optional Visualizations:
- `reports/two_stage_analysis/uncertainty_waterfall.png`
- `reports/two_stage_analysis/error_propagation_scatter.png`
- `reports/two_stage_analysis/behavioral_predictions_uncertainty.png`
- `reports/two_stage_analysis/pipeline_summary_dashboard.png`

## Running the Demo

```bash
# Start the demo
streamlit run demo_app.py

# Navigate in browser to the Two-Stage Pipeline section
# Expand the "Explore Two-Stage Pipeline Results" expander
# Select student and date to explore predictions
```

## Future Enhancements

Potential improvements for next iteration:

1. **Direct Comparison Feature** - Side-by-side cascaded vs direct predictions
2. **Uncertainty Visualization** - Real-time uncertainty bars for each prediction
3. **Interactive What-If** - Adjust Stage 1 predictions and see Stage 2 impact
4. **Ground Truth Overlay** - If/when mental health labels become available
5. **Confidence Bands** - Show prediction ranges instead of point estimates
6. **Student Profiles** - Brief context about each StudentLife participant
7. **Export Functionality** - Download selected predictions as CSV/JSON
8. **Animation** - Show data flow through pipeline stages

## Research Contribution

This demo makes the two-stage pipeline research **tangible and interactive**:
- Reviewers can explore actual predictions
- Patterns emerge through interactive exploration
- Limitations are transparent and well-documented
- Visualization aids understanding of complex error propagation
- Real data (StudentLife) grounds the research in practical applications

**Result**: A compelling demonstration of hybrid ML pipelines for mental health prediction that bridges the gap between real behavioral data and synthetic mental health inference.
