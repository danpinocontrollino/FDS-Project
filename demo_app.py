"""
================================================================================
STREAMLIT INTERACTIVE DEMO - Mental Health Profiling System
================================================================================
Real-time interactive demo for live presentations and demonstrations.

Run with: streamlit run demo_app.py

Features:
  - Interactive input sliders for behavioral data
  - Real-time LSTM predictions
  - Visual risk assessment
  - Job-specific recommendations
  - Professional UI for presentations

Author: FDS Project Team
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import sys

# Add scripts to path
sys.path.append(str(Path(__file__).parent / "scripts"))

st.set_page_config(
    page_title="Mental Health Profiling Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL LOADING (Simplified from generate_profile.py)
# ============================================================================

@st.cache_resource
def load_model_and_config():
    """Load LSTM model and configuration files."""
    # Load job categories
    with open("config/job_categories.json", "r") as f:
        job_config = json.load(f)
    
    # Load thresholds
    with open("config/thresholds.json", "r") as f:
        thresholds = json.load(f)
    
    # Load model (simplified version)
    model_path = Path("models/saved/mental_health_lstm.pt")
    
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model = checkpoint.get("model")
        scaler_mean = checkpoint.get("scaler_mean", np.zeros(17))
        scaler_scale = checkpoint.get("scaler_scale", np.ones(17))
        
        if model is None:
            st.error("Model not found in checkpoint")
            return None, None, None, None, None
            
        model.eval()
        return model, scaler_mean, scaler_scale, job_config, thresholds
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

def predict_mental_health(model, behavioral_data, scaler_mean, scaler_scale):
    """Run LSTM prediction on behavioral data."""
    try:
        # Normalize
        normalized = (behavioral_data - scaler_mean) / scaler_scale
        
        # Handle NaN
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        # Convert to tensor [1, 7, 17]
        X = torch.FloatTensor(normalized).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(X)
        
        # Extract predictions
        predictions = {}
        for target, output in outputs.items():
            if isinstance(output, dict):
                value = output['value'].item()
                at_risk_prob = output.get('at_risk', torch.tensor(0.5)).item()
            else:
                value = output.item()
                at_risk_prob = 0.5
            
            predictions[target] = {
                'value': value,
                'at_risk_prob': at_risk_prob
            }
        
        return predictions
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render application header."""
    st.title("🧠 Mental Health Profiling System")
    st.markdown("### Interactive Demo - Real-Time Predictions")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "LSTM (8 targets)", "98.5% accuracy")
    with col2:
        st.metric("Training Data", "500K+ samples", "1.5M records")
    with col3:
        st.metric("Prediction Time", "<100ms", "Real-time")

def render_input_sidebar():
    """Render input controls in sidebar."""
    st.sidebar.header("📊 Behavioral Inputs")
    st.sidebar.markdown("*Enter 7-day average values:*")
    
    # Sleep
    st.sidebar.subheader("😴 Sleep")
    sleep_hours = st.sidebar.slider("Sleep Hours/Night", 3.0, 12.0, 7.0, 0.5)
    sleep_quality = st.sidebar.slider("Sleep Quality (1-10)", 1.0, 10.0, 7.0, 0.5)
    
    # Work
    st.sidebar.subheader("💼 Work")
    work_hours = st.sidebar.slider("Work Hours/Day", 0.0, 16.0, 8.0, 0.5)
    meetings = st.sidebar.slider("Meetings/Day", 0, 15, 3, 1)
    tasks_completed = st.sidebar.slider("Tasks Completed/Day", 0, 20, 6, 1)
    work_pressure = st.sidebar.select_slider("Work Pressure", ["low", "medium", "high"], "medium")
    
    # Physical Health
    st.sidebar.subheader("🏃 Physical Health")
    exercise_minutes = st.sidebar.slider("Exercise Minutes/Day", 0, 180, 30, 5)
    caffeine_mg = st.sidebar.slider("Caffeine (mg/day)", 0, 800, 200, 25)
    
    # Lifestyle
    st.sidebar.subheader("🌟 Lifestyle")
    social_interactions = st.sidebar.slider("Social Interactions/Day", 0, 15, 5, 1)
    outdoor_time = st.sidebar.slider("Outdoor Time (min/day)", 0, 180, 30, 5)
    diet_quality = st.sidebar.slider("Diet Quality (1-10)", 1.0, 10.0, 7.0, 0.5)
    screen_time = st.sidebar.slider("Screen Time (hours/day)", 0.0, 18.0, 6.0, 0.5)
    
    # Job Category
    st.sidebar.subheader("👔 Job Category")
    job_title = st.sidebar.text_input("Job Title (optional)", "Software Engineer")
    
    # Convert work pressure to numeric
    pressure_map = {"low": 3, "medium": 5, "high": 8}
    
    return {
        'sleep_hours': sleep_hours,
        'sleep_quality': sleep_quality,
        'work_hours': work_hours,
        'meetings_count': meetings,
        'tasks_completed': tasks_completed,
        'work_pressure': pressure_map[work_pressure],
        'exercise_minutes': exercise_minutes,
        'caffeine_mg': caffeine_mg,
        'social_interactions': social_interactions,
        'outdoor_time_minutes': outdoor_time,
        'diet_quality': diet_quality,
        'screen_time_hours': screen_time,
        'job_title': job_title,
        # Fill remaining with defaults
        'emails_received': 50,
        'commute_minutes': 20,
        'steps_count': 5000,
        'alcohol_units': 0,
        'weather_mood_impact': 0
    }

def render_predictions(predictions, thresholds):
    """Render prediction results."""
    if predictions is None:
        st.error("No predictions available")
        return
    
    st.header("📈 Predictions")
    
    # Daily predictions
    st.subheader("🔹 Daily Predictions (Next Day)")
    daily_cols = st.columns(4)
    daily_targets = ['stress_level', 'mood_score', 'energy_level', 'focus_score']
    
    for col, target in zip(daily_cols, daily_targets):
        if target in predictions:
            value = predictions[target]['value']
            at_risk_prob = predictions[target]['at_risk_prob']
            
            # Determine if at-risk
            threshold_info = thresholds['at_risk_thresholds'].get(target, {})
            threshold = threshold_info.get('threshold', 5)
            inverted = threshold_info.get('inverted', False)
            
            if inverted:
                at_risk = value < threshold
            else:
                at_risk = value >= threshold
            
            # Color coding
            color = "🔴" if at_risk else "✅"
            
            with col:
                st.metric(
                    f"{color} {target.replace('_', ' ').title()}",
                    f"{value:.1f}",
                    delta=f"{at_risk_prob*100:.0f}% confidence",
                    delta_color="inverse" if at_risk else "normal"
                )
    
    # Weekly predictions
    st.subheader("🔹 Weekly Predictions (End of Week)")
    weekly_cols = st.columns(4)
    weekly_targets = ['perceived_stress_scale', 'anxiety_score', 'depression_score', 'job_satisfaction']
    
    for col, target in zip(weekly_cols, weekly_targets):
        if target in predictions:
            value = predictions[target]['value']
            at_risk_prob = predictions[target]['at_risk_prob']
            
            # Determine if at-risk
            threshold_info = thresholds['at_risk_thresholds'].get(target, {})
            threshold = threshold_info.get('threshold', 10)
            inverted = threshold_info.get('inverted', False)
            
            if inverted:
                at_risk = value < threshold
            else:
                at_risk = value >= threshold
            
            color = "🔴" if at_risk else "✅"
            
            with col:
                st.metric(
                    f"{color} {target.replace('_', ' ').title()}",
                    f"{value:.1f}",
                    delta=f"{at_risk_prob*100:.0f}% confidence",
                    delta_color="inverse" if at_risk else "normal"
                )

def render_risk_assessment(inputs, predictions, thresholds):
    """Render risk factors and positive factors."""
    st.header("⚠️ Risk Assessment")
    
    risk_factors = []
    positive_factors = []
    
    # Check behavioral thresholds
    if inputs['sleep_hours'] < 6:
        risk_factors.append(f"Insufficient sleep: {inputs['sleep_hours']}h/night (< 6h)")
    elif inputs['sleep_hours'] >= 7:
        positive_factors.append(f"Adequate sleep: {inputs['sleep_hours']}h/night")
    
    if inputs['exercise_minutes'] < 20:
        risk_factors.append(f"Sedentary lifestyle: {inputs['exercise_minutes']}min/day")
    elif inputs['exercise_minutes'] >= 30:
        positive_factors.append(f"Regular exercise: {inputs['exercise_minutes']}min/day")
    
    if inputs['caffeine_mg'] > 400:
        risk_factors.append(f"High caffeine: {inputs['caffeine_mg']}mg/day (> 400mg)")
    
    if inputs['work_hours'] > 9:
        risk_factors.append(f"Excessive work hours: {inputs['work_hours']}h/day")
    
    if inputs['social_interactions'] < 3:
        risk_factors.append(f"Social isolation: {inputs['social_interactions']} interactions/day")
    elif inputs['social_interactions'] >= 4:
        positive_factors.append(f"Good social connections: {inputs['social_interactions']} interactions/day")
    
    # Check prediction-based risks
    if predictions:
        for target, data in predictions.items():
            threshold_info = thresholds['at_risk_thresholds'].get(target, {})
            threshold = threshold_info.get('threshold', 5)
            inverted = threshold_info.get('inverted', False)
            value = data['value']
            
            if inverted:
                at_risk = value < threshold
            else:
                at_risk = value >= threshold
            
            if at_risk:
                risk_factors.append(f"At-risk {target.replace('_', ' ')}: {value:.1f}")
            else:
                positive_factors.append(f"Healthy {target.replace('_', ' ')}: {value:.1f}")
    
    # Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 Risk Factors")
        if risk_factors:
            for i, factor in enumerate(risk_factors[:5], 1):
                st.write(f"{i}. {factor}")
        else:
            st.success("No major risk factors detected!")
    
    with col2:
        st.subheader("✅ Positive Factors")
        if positive_factors:
            for i, factor in enumerate(positive_factors[:5], 1):
                st.write(f"{i}. {factor}")
        else:
            st.info("Building healthy habits...")

def render_quick_advice(inputs):
    """Render quick actionable advice."""
    st.header("💡 Quick Recommendations")
    
    advice = []
    
    if inputs['sleep_hours'] < 7:
        advice.append({
            'category': 'Sleep',
            'priority': 'HIGH',
            'advice': f"Increase sleep from {inputs['sleep_hours']}h to 7-9h per night",
            'action': "Set a bedtime alarm 1-2 hours earlier"
        })
    
    if inputs['exercise_minutes'] < 30:
        advice.append({
            'category': 'Exercise',
            'priority': 'MEDIUM',
            'advice': f"Increase activity from {inputs['exercise_minutes']}min to 30+ min/day",
            'action': "Add a 20-minute walk during lunch break"
        })
    
    if inputs['caffeine_mg'] > 400:
        advice.append({
            'category': 'Caffeine',
            'priority': 'MEDIUM',
            'advice': f"Reduce caffeine from {inputs['caffeine_mg']}mg to <400mg/day",
            'action': "Cut off coffee after 2pm (6-hour half-life)"
        })
    
    if inputs['work_hours'] > 9:
        advice.append({
            'category': 'Work-Life Balance',
            'priority': 'HIGH',
            'advice': f"Reduce work from {inputs['work_hours']}h to 8-9h/day maximum",
            'action': "Set hard stop time at 6pm, no emails after"
        })
    
    if advice:
        for item in advice[:3]:
            priority_color = "🔴" if item['priority'] == 'HIGH' else "🟡"
            with st.expander(f"{priority_color} {item['category']} - {item['priority']} Priority"):
                st.write(f"**Recommendation:** {item['advice']}")
                st.write(f"**Action:** {item['action']}")
    else:
        st.success("Your behavioral patterns look healthy! Keep it up. 🎉")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Load model
    model, scaler_mean, scaler_scale, job_config, thresholds = load_model_and_config()
    
    if model is None:
        st.error("Failed to load model. Please check configuration.")
        st.stop()
    
    # Render UI
    render_header()
    
    # Get inputs
    inputs = render_input_sidebar()
    
    # Prepare behavioral data (7 days, all same for demo)
    feature_order = [
        'sleep_hours', 'sleep_quality', 'work_hours', 'meetings_count',
        'tasks_completed', 'emails_received', 'commute_minutes',
        'exercise_minutes', 'steps_count', 'caffeine_mg', 'alcohol_units',
        'screen_time_hours', 'social_interactions', 'outdoor_time_minutes',
        'diet_quality', 'work_pressure', 'weather_mood_impact'
    ]
    
    behavioral_data = np.array([[inputs.get(f, 0) for f in feature_order] for _ in range(7)])
    
    # Predict button
    if st.sidebar.button("🔮 Generate Profile", type="primary"):
        with st.spinner("Running LSTM prediction..."):
            predictions = predict_mental_health(model, behavioral_data, scaler_mean, scaler_scale)
        
        if predictions:
            st.success("✓ Profile generated successfully!")
            
            # Render results
            render_predictions(predictions, thresholds)
            st.markdown("---")
            render_risk_assessment(inputs, predictions, thresholds)
            st.markdown("---")
            render_quick_advice(inputs)
            
            # Download option
            st.markdown("---")
            st.info("💡 **Tip:** For full detailed report with charts and history tracking, use `python scripts/generate_profile.py --csv your_data.csv --html`")
    
    else:
        st.info("👈 Adjust behavioral inputs in the sidebar, then click **Generate Profile** to see predictions!")
        
        # Show example
        st.markdown("---")
        st.subheader("📖 How It Works")
        st.markdown("""
        1. **Input**: Enter 7-day average behavioral data (sleep, work, exercise, etc.)
        2. **Model**: LSTM with 2 layers processes the sequence
        3. **Output**: 8 mental health predictions in <100ms
        4. **Analysis**: Automatic risk assessment and personalized advice
        
        **Model Performance:**
        - Job Satisfaction: 98.5% accuracy
        - Depression: 98% accuracy
        - Anxiety: 97% accuracy
        - Trained on 500K+ samples
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://via.placeholder.com/300x200?text=LSTM+Architecture", caption="Multi-Task LSTM Model")
        with col2:
            st.image("https://via.placeholder.com/300x200?text=Training+Data", caption="1.5M+ Training Records")

if __name__ == "__main__":
    main()
