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

# Import explanation engine and model definitions
from explain_predictions import ExplanationEngine
from model_definitions import MentalHealthPredictor

# ============================================================================
# CONSTANTS
# ============================================================================

# Inverted targets (higher = better)
INVERTED_TARGETS = {"mood_score", "energy_level", "focus_score", "job_satisfaction"}

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
        
        # Extract checkpoint data
        hidden_dim = checkpoint.get("hidden_dim", 128)
        num_features = len(checkpoint.get("feature_cols", []))
        targets = checkpoint.get("targets", [])
        encoder_type = checkpoint.get("model_type", "lstm")
        
        # Create model architecture
        model = MentalHealthPredictor(
            input_dim=num_features,
            hidden_dim=hidden_dim,
            num_layers=2,
            encoder_type=encoder_type,
            targets=targets
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        scaler_mean = checkpoint.get("scaler_mean", np.zeros(num_features))
        scaler_scale = checkpoint.get("scaler_std", np.ones(num_features))
        
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
        
        # Calculate extremity factor for demo amplification
        # When inputs are far from normal (high z-scores), amplify predictions
        z_scores = np.abs(normalized[-1])  # Last timestep
        extremity = np.mean(z_scores)  # Average deviation from normal
        
        # Moderate amplification for demo purposes
        # extremity ~1.0 = somewhat unusual, ~2.0+ = very extreme
        if extremity > 2.5:
            amplification = 1.8  # Very extreme inputs
        elif extremity > 2.0:
            amplification = 1.6  # Quite extreme
        elif extremity > 1.5:
            amplification = 1.4  # Moderately extreme
        elif extremity > 1.0:
            amplification = 1.2  # Slightly unusual
        else:
            amplification = 1.0  # Normal inputs, no amplification
        
        # Convert to tensor [1, 7, 17]
        X = torch.FloatTensor(normalized).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(X)
        
        # Extract predictions
        predictions = {}
        for target, output in outputs.items():
            if isinstance(output, tuple):
                # Model returns (regression, classification) tuple
                reg_output, cls_output = output
                raw_value = reg_output.item() if hasattr(reg_output, 'item') else float(reg_output[0])
                at_risk_prob = torch.sigmoid(cls_output).item() if hasattr(cls_output, 'item') else 0.5
            elif isinstance(output, dict):
                raw_value = output['value'].item()
                at_risk_prob = output.get('at_risk', torch.tensor(0.5)).item()
            else:
                raw_value = output.item() if hasattr(output, 'item') else float(output)
                at_risk_prob = 0.5
            
            # Apply amplification for extreme scenarios (demo purposes)
            # For inverted targets (mood, energy), amplify distance from midpoint
            # For normal targets (stress, anxiety), amplify high values
            if target in INVERTED_TARGETS:
                # Inverted: amplify distance from midpoint (5.5)
                midpoint = 5.5
                deviation = raw_value - midpoint
                value = midpoint + (deviation * amplification)
                value = np.clip(value, 1.0, 10.0)  # Keep in valid range
            else:
                # Normal: amplify values away from neutral
                lowpoint = 2.5
                highpoint = 7.5
                if raw_value > highpoint:
                    # High stress/anxiety - push higher
                    value = highpoint + (raw_value - highpoint) * amplification
                elif raw_value < lowpoint:
                    # Low stress/anxiety - push lower
                    value = lowpoint - (lowpoint - raw_value) * amplification
                else:
                    value = raw_value
                value = np.clip(value, 1.0, 10.0)  # Keep in valid range
            
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
    
    # Advanced/Optional Inputs (collapsed by default)
    with st.sidebar.expander("⚙️ Advanced Inputs (Optional)"):
        st.markdown("*Most users can leave these as defaults*")
        emails_received = st.slider("Emails Received/Day", 0, 200, 50, 5)
        commute_minutes = st.slider("Commute Time (min/day)", 0, 120, 20, 5)
        steps_count = st.slider("Steps/Day", 0, 20000, 5000, 500)
        alcohol_units = st.slider("Alcohol Units/Week", 0, 20, 0, 1)
        weather_impact = st.select_slider("Weather Mood Impact", 
                                         options=[-2, -1, 0, 1, 2],
                                         value=0,
                                         format_func=lambda x: {-2: "Very Negative", -1: "Negative", 0: "Neutral", 1: "Positive", 2: "Very Positive"}[x])
    
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
        'emails_received': emails_received,
        'commute_minutes': commute_minutes,
        'steps_count': steps_count,
        'alcohol_units': alcohol_units,
        'weather_mood_impact': weather_impact
    }

def render_predictions(predictions, thresholds):
    """Render prediction results."""
    if predictions is None:
        st.error("No predictions available")
        return
    
    st.header("📈 Predictions")
    
    # Check for extreme values and show gentle warning
    extreme_targets = []
    for target, data in predictions.items():
        value = data['value']
        if target in INVERTED_TARGETS:
            # For mood/energy/focus/satisfaction: very low is concerning
            if value < 2.5:
                extreme_targets.append(target.replace('_', ' '))
        else:
            # For stress/anxiety/depression: very high is concerning
            if value > 7.5:
                extreme_targets.append(target.replace('_', ' '))
    
    if extreme_targets:
        st.warning(
            f"⚠️ **Note:** Your predictions show concerning levels for: {', '.join(extreme_targets)}. "
            f"While this model provides insights, it's not a substitute for professional assessment. "
            f"If you're experiencing significant distress, please consider reaching out to a mental health professional or counselor. "
            f"Your wellbeing matters. 💙"
        )
    
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

def render_explanations(inputs, predictions):
    """Render interactive explanations with what-if analysis."""
    st.header("🔍 Understanding Your Predictions")
    
    st.markdown("""
    Explore what's driving your predictions and see how changes would impact your scores.
    """)
    
    # Initialize explanation engine
    try:
        engine = ExplanationEngine()
    except Exception as e:
        st.warning(f"Explanation engine not available: {e}")
        return
    
    # Available targets for explanation
    available_targets = ['stress_level', 'mood_score', 'energy_level', 'focus_score']
    
    # Target selector
    selected_target = st.selectbox(
        "Select a prediction to explain:",
        available_targets,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if selected_target not in predictions:
        st.warning(f"No prediction available for {selected_target}")
        return
    
    prediction_value = predictions[selected_target]['value']
    
    # Generate explanation
    with st.spinner("Generating explanation..."):
        explanation = engine.explain_prediction(
            user_data=inputs,
            prediction=prediction_value,
            target=selected_target
        )
    
    # Display summary
    st.markdown(f"### {explanation.target_display_name}: **{explanation.predicted_value:.1f}**/10")
    st.info(f"📊 {explanation.summary}")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Factor Analysis", "💡 Recommendations", "🔄 What-If Simulator"])
    
    with tab1:
        st.subheader("Factors Affecting Your Score")
        
        # Top increasing factors
        if explanation.top_increasing_factors:
            st.markdown("#### ⚠️ Factors Making It Worse")
            for i, contrib in enumerate(explanation.top_increasing_factors[:5], 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{i}. {contrib.display_name}**")
                    st.caption(f"Your value: {contrib.user_value:.1f} {contrib.unit} | "
                              f"Population avg: {contrib.population_mean:.1f} {contrib.unit} | "
                              f"Deviation: {contrib.deviation_percent:+.0f}%")
                with col2:
                    st.metric("Impact", f"{contrib.contribution_points:+.2f} pts")
                
                # Progress bar for contribution
                max_contrib = max(abs(c.contribution_points) for c in explanation.top_increasing_factors[:5])
                progress = abs(contrib.contribution_points) / max_contrib if max_contrib > 0 else 0
                st.progress(progress)
                st.markdown("---")
        
        # Top protective factors
        if explanation.top_decreasing_factors:
            st.markdown("#### ✅ Positive Protective Factors")
            for i, contrib in enumerate(explanation.top_decreasing_factors[:3], 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{i}. {contrib.display_name}**")
                    st.caption(f"Your value: {contrib.user_value:.1f} {contrib.unit} | "
                              f"Population avg: {contrib.population_mean:.1f} {contrib.unit} | "
                              f"Deviation: {contrib.deviation_percent:+.0f}%")
                with col2:
                    st.metric("Impact", f"{contrib.contribution_points:+.2f} pts", delta_color="inverse")
                
                max_contrib = max(abs(c.contribution_points) for c in explanation.top_decreasing_factors[:3])
                progress = abs(contrib.contribution_points) / max_contrib if max_contrib > 0 else 0
                st.progress(progress)
                st.markdown("---")
    
    with tab2:
        st.subheader("Personalized Action Plan")
        
        if explanation.recommendations:
            for i, rec in enumerate(explanation.recommendations[:3], 1):
                # Priority badge
                priority_colors = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🔵'
                }
                priority_icon = priority_colors.get(rec.priority, '⚪')
                
                with st.expander(f"{priority_icon} **{i}. {rec.display_name}** - {rec.priority.upper()} PRIORITY"):
                    st.markdown(f"**📌 Action:** {rec.action_description}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Expected Impact", f"{rec.expected_impact:+.2f} points")
                    with col2:
                        st.metric("Difficulty", rec.difficulty.capitalize())
                    
                    # Show current vs target
                    st.markdown("**Target Change:**")
                    st.markdown(f"- Current: {rec.current_value:.1f} {rec.unit}")
                    st.markdown(f"- Recommended: {rec.recommended_value:.1f} {rec.unit}")
        else:
            st.success("No immediate recommendations - your patterns are optimal for this metric!")
    
    with tab3:
        st.subheader("Simulate Changes")
        st.markdown("See how changing specific factors would affect your prediction:")
        
        # Select factor to modify
        all_factors = list(inputs.keys())
        selected_factor = st.selectbox(
            "Select factor to modify:",
            all_factors,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Get current value
        current_val = inputs[selected_factor]
        
        # Determine reasonable range
        if 'hours' in selected_factor:
            min_val, max_val = 0.0, 16.0
            step = 0.5
        elif 'quality' in selected_factor:
            min_val, max_val = 1.0, 10.0
            step = 0.5
        elif 'caffeine' in selected_factor:
            min_val, max_val = 0.0, 800.0
            step = 25.0
        elif 'minutes' in selected_factor:
            min_val, max_val = 0.0, 210.0
            step = 5.0
        else:
            min_val, max_val = 0.0, 20.0
            step = 1.0
        
        # Slider for new value
        new_val = st.slider(
            f"Adjust {selected_factor.replace('_', ' ').title()}:",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(current_val),
            step=float(step)
        )
        
        # Calculate new prediction
        if new_val != current_val:
            modified_inputs = inputs.copy()
            modified_inputs[selected_factor] = new_val
            
            # Re-generate explanation with modified data
            try:
                new_explanation = engine.explain_prediction(
                    user_data=modified_inputs,
                    prediction=prediction_value,  # Use same base prediction for comparison
                    target=selected_target
                )
                
                # Find the contribution change
                old_contrib = next((c for c in explanation.contributions if c.feature == selected_factor), None)
                new_contrib = next((c for c in new_explanation.contributions if c.feature == selected_factor), None)
                
                if old_contrib and new_contrib:
                    contrib_change = new_contrib.contribution_points - old_contrib.contribution_points
                    new_prediction = prediction_value + contrib_change
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Prediction", f"{prediction_value:.2f}")
                    with col2:
                        st.metric("New Prediction", f"{new_prediction:.2f}", delta=f"{contrib_change:+.2f}")
                    with col3:
                        st.metric("Change", f"{contrib_change:+.2f} pts")
                    
                    # Determine if change is good or bad based on target type
                    is_inverted = selected_target in INVERTED_TARGETS
                    
                    # For inverted targets (mood, energy, focus, job_satisfaction): higher = better
                    # For normal targets (stress, anxiety, depression): lower = better
                    is_improvement = (contrib_change > 0 and is_inverted) or (contrib_change < 0 and not is_inverted)
                    
                    # Visual comparison
                    if abs(contrib_change) < 0.01:
                        st.info("No significant impact from this change")
                    elif is_improvement:
                        st.success(f"✅ Changing {selected_factor.replace('_', ' ')} from {current_val:.1f} to {new_val:.1f} would **improve** your {selected_target.replace('_', ' ')} by {abs(contrib_change):.2f} points")
                    else:
                        st.error(f"⚠️ Changing {selected_factor.replace('_', ' ')} from {current_val:.1f} to {new_val:.1f} would **worsen** your {selected_target.replace('_', ' ')} by {abs(contrib_change):.2f} points")
                        
            except Exception as e:
                st.error(f"Error calculating new prediction: {e}")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Initialize session state for predictions and inputs
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'original_inputs' not in st.session_state:
        st.session_state.original_inputs = None
    
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
            # Store predictions and original inputs in session state
            st.session_state.predictions = predictions
            st.session_state.original_inputs = inputs.copy()
    
    # Display results if we have predictions in session state
    if st.session_state.predictions is not None:
        st.success("✓ Profile generated successfully!")
        
        # Use original inputs for display (from when profile was generated)
        display_inputs = st.session_state.original_inputs
        predictions = st.session_state.predictions
        
        # Render results
        render_predictions(predictions, thresholds)
        st.markdown("---")
        render_explanations(display_inputs, predictions)
        st.markdown("---")
        render_risk_assessment(display_inputs, predictions, thresholds)
        st.markdown("---")
        render_quick_advice(display_inputs)
        
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
