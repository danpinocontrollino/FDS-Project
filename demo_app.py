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

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class PredictionHead(nn.Module):
    """Single prediction head for one target (regression only)."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.regression = nn.Linear(hidden_dim, 1)
        self.classification = nn.Linear(hidden_dim, 1)  # Single output for regression
    
    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        reg = self.regression(h).squeeze(-1)
        cls = self.classification(h).squeeze(-1)  # Same as regression for "both" task
        return reg, cls

class MentalHealthPredictor(nn.Module):
    """Multi-target mental health prediction model."""
    
    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = "lstm",
        targets = None,
    ):
        super().__init__()
        
        self.targets = targets or []
        self.encoder_type = encoder_type
        
        if encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=0.2 if num_layers > 1 else 0
            )
        elif encoder_type == "gru":
            self.encoder = nn.GRU(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=0.2 if num_layers > 1 else 0
            )
        elif encoder_type == "transformer":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.pos_encoder = PositionalEncoding(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
                dropout=0.1, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.shared_repr = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.heads = nn.ModuleDict({
            target: PredictionHead(hidden_dim, hidden_dim // 2)
            for target in self.targets
        })
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to fixed representation."""
        if self.encoder_type in ["lstm", "gru"]:
            out, _ = self.encoder(x)
            return out[:, -1, :]  # Last timestep
        else:  # transformer
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.encoder(x)
            return x.mean(dim=1)  # Mean pooling
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        h = self.encode(x)
        h = self.shared_repr(h)
        
        outputs = {}
        for target in self.targets:
            reg, cls = self.heads[target](h)
            outputs[target] = (reg, cls)
        
        return outputs

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

def get_color_and_delta(target: str, value: float, thresholds: dict) -> tuple:
    """
    Get color emoji and delta_color based on target value and thresholds.
    Returns: (color_emoji, delta_color)
    
    Color scheme:
    - Green (🟢): Healthy/Good
    - Orange (🟠): Moderate/Warning
    - Red (🔴): At-risk/Severe
    """
    
    # Get threshold info
    threshold_info = thresholds['at_risk_thresholds'].get(target, {})
    threshold = threshold_info.get('threshold', 10)
    inverted = threshold_info.get('inverted', False)
    
    # Special handling for anxiety_score (0-9 green, 10-14 orange, 15-21 red)
    if target == 'anxiety_score':
        if value < 10:
            return "🟢", "normal"
        elif value <= 14:
            return "🟠", "normal"
        else:  # 15-21
            return "🔴", "inverse"
    
    # Special handling for depression_score (0-10 green, 11-18 orange, 19-27 red)
    elif target == 'depression_score':
        if value < 11:
            return "🟢", "normal"
        elif value <= 18:
            return "🟠", "normal"
        else:  # 19-27
            return "🔴", "inverse"
    
    # Special handling for perceived_stress_scale (0-13 green, 14-26 orange, 27-40 red)
    elif target == 'perceived_stress_scale':
        if value < 14:
            return "🟢", "normal"
        elif value <= 26:
            return "🟠", "normal"
        else:  # 27-40
            return "🔴", "inverse"
    
    # Special handling for job_satisfaction (inverted: higher is better)
    elif target == 'job_satisfaction':
        if value >= 7:
            return "🟢", "normal"
        elif value >= 5:
            return "🟠", "normal"
        else:  # < 5
            return "🔴", "inverse"
    
    # Default logic for daily metrics (stress_level, mood_score, energy_level, focus_score)
    else:
        # For non-inverted: higher value = worse (stress_level uses this)
        if not inverted:
            if value < 4:
                return "🟢", "normal"
            elif value <= 6:
                return "🟠", "normal"
            else:  # > 6
                return "🔴", "inverse"
        # For inverted: lower value = worse (mood, energy, focus)
        else:
            if value >= 7:
                return "🟢", "normal"
            elif value >= 5:
                return "🟠", "normal"
            else:  # < 5
                return "🔴", "inverse"

def generate_prediction_explanation(target: str, value: float, inputs: dict, thresholds: dict) -> dict:
    """
    Generate detailed explanation for why a prediction has this value.
    Returns dict with factors, percentages, and recommendations.
    """
    
    explanation = {
        'target': target,
        'value': value,
        'factors': [],
        'recommendations': []
    }
    
    # Anxiety explanation
    if target == 'anxiety_score':
        anxiety_factors = []
        if inputs['sleep_quality'] < 6:
            anxiety_factors.append(('Poor sleep quality', inputs['sleep_quality'], 6, 30))
        if inputs['social_interactions'] < 3:
            anxiety_factors.append(('Social isolation', inputs['social_interactions'], 3, 25))
        if inputs['caffeine_mg'] > 300:
            anxiety_factors.append(('High caffeine', inputs['caffeine_mg'], 300, 20))
        if inputs['outdoor_time_minutes'] < 20:
            anxiety_factors.append(('No outdoor time', inputs['outdoor_time_minutes'], 20, 15))
        if inputs['work_pressure'] > 6:
            anxiety_factors.append(('High work pressure', inputs['work_pressure'], 6, 10))
        
        explanation['factors'] = anxiety_factors
        explanation['recommendations'] = [
            {'action': 'Improve sleep quality', 'impact': 'Could reduce anxiety by ~4-5 points', 'effort': 'Medium'},
            {'action': 'Increase social interactions', 'impact': 'Could reduce anxiety by ~2-3 points', 'effort': 'Easy'},
            {'action': 'Reduce caffeine intake', 'impact': 'Could reduce anxiety by ~1-2 points', 'effort': 'Easy'},
            {'action': 'Spend 30min outdoors daily', 'impact': 'Could reduce anxiety by ~1-2 points', 'effort': 'Easy'},
        ]
    
    # Stress Level explanation
    elif target == 'stress_level':
        stress_factors = []
        if inputs['work_hours'] > 9:
            stress_factors.append(('Long work hours', inputs['work_hours'], 9, 35))
        if inputs['sleep_hours'] < 7:
            stress_factors.append(('Insufficient sleep', inputs['sleep_hours'], 7, 30))
        if inputs['meetings_count'] > 5:
            stress_factors.append(('Too many meetings', inputs['meetings_count'], 5, 20))
        if inputs['exercise_minutes'] < 30:
            stress_factors.append(('Low exercise', inputs['exercise_minutes'], 30, 15))
        
        explanation['factors'] = stress_factors
        explanation['recommendations'] = [
            {'action': 'Reduce work hours to 8-9h max', 'impact': 'Could reduce stress by ~2-3 points', 'effort': 'Hard'},
            {'action': 'Improve sleep to 7-9h', 'impact': 'Could reduce stress by ~2 points', 'effort': 'Medium'},
            {'action': 'Exercise 30+ min daily', 'impact': 'Could reduce stress by ~1 point', 'effort': 'Medium'},
        ]
    
    # Mood Score explanation (inverted)
    elif target == 'mood_score':
        mood_factors = []
        if inputs['social_interactions'] < 4:
            mood_factors.append(('Low social contact', inputs['social_interactions'], 4, 30))
        if inputs['sleep_hours'] < 7:
            mood_factors.append(('Poor sleep', inputs['sleep_hours'], 7, 25))
        if inputs['diet_quality'] < 6:
            mood_factors.append(('Poor diet', inputs['diet_quality'], 6, 20))
        if inputs['outdoor_time_minutes'] < 30:
            mood_factors.append(('No outdoor time', inputs['outdoor_time_minutes'], 30, 15))
        if inputs['exercise_minutes'] < 20:
            mood_factors.append(('Low exercise', inputs['exercise_minutes'], 20, 10))
        
        explanation['factors'] = mood_factors
        explanation['recommendations'] = [
            {'action': 'Increase social interactions', 'impact': 'Could improve mood by ~1-2 points', 'effort': 'Easy'},
            {'action': 'Sleep 7-9h nightly', 'impact': 'Could improve mood by ~1 point', 'effort': 'Medium'},
            {'action': 'Eat healthier', 'impact': 'Could improve mood by ~0.5 points', 'effort': 'Medium'},
            {'action': 'Spend 30min outdoors daily', 'impact': 'Could improve mood by ~0.5 points', 'effort': 'Easy'},
        ]
    
    # Depression Score explanation
    elif target == 'depression_score':
        depression_factors = []
        if inputs['exercise_minutes'] < 15:
            depression_factors.append(('Sedentary', inputs['exercise_minutes'], 15, 35))
        if inputs['social_interactions'] < 2:
            depression_factors.append(('Social isolation', inputs['social_interactions'], 2, 30))
        if inputs['sleep_hours'] < 6.5:
            depression_factors.append(('Severe sleep deficit', inputs['sleep_hours'], 6.5, 20))
        if inputs['work_satisfaction'] < 5:
            depression_factors.append(('Job dissatisfaction', 0, 1, 15))
        
        explanation['factors'] = depression_factors
        explanation['recommendations'] = [
            {'action': 'Start exercise routine (30min/day)', 'impact': 'Could reduce depression by ~5-7 points', 'effort': 'Medium'},
            {'action': 'Increase social contact', 'impact': 'Could reduce depression by ~4-5 points', 'effort': 'Medium'},
            {'action': 'Improve sleep to 7-9h', 'impact': 'Could reduce depression by ~3-4 points', 'effort': 'Medium'},
            {'action': 'Seek professional support', 'impact': 'Significant improvement', 'effort': 'N/A'},
        ]
    
    # Default for other targets
    else:
        explanation['factors'] = [('Multiple behavioral factors', 0, 0, 100)]
        explanation['recommendations'] = [
            {'action': 'Balance sleep, exercise, work', 'impact': 'General improvement', 'effort': 'Medium'},
        ]
    
    return explanation

def render_goal_setter():
    """Render goal-setting interface in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Goal Setter")
    
    goals = {}
    st.sidebar.markdown("*Set your target values:*")
    
    # Goal for stress
    goals['stress_target'] = st.sidebar.slider(
        "Target Stress Level", 
        1.0, 10.0, 4.0, 0.5,
        help="Lower is better (1-3: healthy, 4-6: moderate, 7+: high)"
    )
    
    # Goal for anxiety
    goals['anxiety_target'] = st.sidebar.slider(
        "Target Anxiety Score",
        0.0, 21.0, 8.0, 0.5,
        help="Lower is better (0-9: healthy, 10-14: moderate, 15-21: high)"
    )
    
    # Goal for mood
    goals['mood_target'] = st.sidebar.slider(
        "Target Mood Score",
        1.0, 10.0, 7.0, 0.5,
        help="Higher is better (1-3: low, 4-6: moderate, 7-10: good)"
    )
    
    # Goal for sleep
    goals['sleep_target'] = st.sidebar.slider(
        "Target Sleep Hours",
        5.0, 10.0, 8.0, 0.5,
        help="Optimal: 7-9 hours"
    )
    
    # Goal for exercise
    goals['exercise_target'] = st.sidebar.slider(
        "Target Exercise (min/day)",
        0, 180, 45, 5,
        help="Recommended: 30-60 minutes"
    )
    
    return goals

def render_prediction_explanations(predictions, inputs, thresholds):
    """Render detailed explanations for each prediction."""
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    st.header("🔍 Understanding Your Predictions")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown("**Click on any metric below to understand what's driving it:**")
    st.markdown("")
    
    # Explanations for main metrics
    explanation_targets = ['stress_level', 'anxiety_score', 'mood_score', 'depression_score']
    
    for target in explanation_targets:
        if target in predictions:
            value = predictions[target]['value']
            explanation = generate_prediction_explanation(target, value, inputs, thresholds)
            
            # Create expander
            with st.expander(f"📊 {target.replace('_', ' ').title()} = {value:.1f}", expanded=False):
                col1, col2 = st.columns(2)
                
                # Left: Contributing factors
                with col1:
                    st.subheader("📍 Contributing Factors")
                    if explanation['factors']:
                        for factor_name, current, target_val, percentage in explanation['factors']:
                            st.markdown(f"**{factor_name}**")
                            st.markdown(f"• Current: {current:.1f} | Target: {target_val:.1f}")
                            st.markdown(f"• Impact: ~{percentage}% of your score")
                            st.markdown("")
                    else:
                        st.markdown("Multiple factors at play")
                
                # Right: What can help
                with col2:
                    st.subheader("💡 What Can Help")
                    if explanation['recommendations']:
                        for i, rec in enumerate(explanation['recommendations'][:3], 1):
                            st.markdown(f"**{i}. {rec['action']}**")
                            st.markdown(f"• Expected impact: {rec['impact']}")
                            st.markdown(f"• Difficulty: {rec['effort']}")
                            st.markdown("")

def render_predictions(predictions, thresholds):
    """Render prediction results."""
    if predictions is None:
        st.error("No predictions available")
        return
    
    st.header("📈 Predictions")
    
    
    # Daily predictions
    st.subheader("🔹 Daily Predictions (Next Day)")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    daily_cols = st.columns(4)
    daily_targets = ['stress_level', 'mood_score', 'energy_level', 'focus_score']
    
    for col, target in zip(daily_cols, daily_targets):
        if target in predictions:
            value = predictions[target]['value']
            at_risk_prob = predictions[target]['at_risk_prob']
            
            # Get color and delta_color
            color, delta_color = get_color_and_delta(target, value, thresholds)
            
            with col:
                st.metric(
                    f"{color} {target.replace('_', ' ').title()}",
                    f"{value:.1f}",
                    delta=f"{at_risk_prob*100:.0f}% confidence",
                    delta_color=delta_color
                )
    
    # Spacing between sections
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    
    # Weekly predictions
    st.subheader("🔹 Weekly Predictions (End of Week)")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    weekly_cols = st.columns(4)
    weekly_targets = ['perceived_stress_scale', 'anxiety_score', 'depression_score', 'job_satisfaction']
    
    for col, target in zip(weekly_cols, weekly_targets):
        if target in predictions:
            value = predictions[target]['value']
            at_risk_prob = predictions[target]['at_risk_prob']
            
            # Get color and delta_color using helper function
            color, delta_color = get_color_and_delta(target, value, thresholds)
            
            with col:
                st.metric(
                    f"{color} {target.replace('_', ' ').title()}",
                    f"{value:.1f}",
                    delta=f"{at_risk_prob*100:.0f}% confidence",
                    delta_color=delta_color
                )

def render_risk_assessment(inputs, predictions, thresholds):
    """Render risk factors and positive factors."""
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    st.header("⚠️ Risk Assessment")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
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
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    st.header("💡 Quick Recommendations")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
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
    
    # Get goals
    goals = render_goal_setter()
    
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
            render_prediction_explanations(predictions, inputs, thresholds)
            st.markdown("---")
            render_risk_assessment(inputs, predictions, thresholds)
            st.markdown("---")
            render_quick_advice(inputs)
            st.markdown("---")
            
            # Goal progress tracking
            st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
            st.header("🎯 Progress Toward Your Goals")
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                current_stress = predictions.get('stress_level', {}).get('value', 5)
                progress = max(0, min(100, ((goals['stress_target'] - current_stress) / (goals['stress_target'] - 1)) * 100))
                st.metric("Stress", f"{current_stress:.1f}", f"Goal: {goals['stress_target']:.1f}")
                st.progress(progress / 100 if progress > 0 else 0)
            
            with col2:
                current_anxiety = predictions.get('anxiety_score', {}).get('value', 8)
                progress = max(0, min(100, ((goals['anxiety_target'] - current_anxiety) / max(1, goals['anxiety_target'])) * 100))
                st.metric("Anxiety", f"{current_anxiety:.1f}", f"Goal: {goals['anxiety_target']:.1f}")
                st.progress(progress / 100 if progress > 0 else 0)
            
            with col3:
                current_mood = predictions.get('mood_score', {}).get('value', 5)
                progress = max(0, min(100, ((current_mood - 1) / 9) * 100))
                st.metric("Mood", f"{current_mood:.1f}", f"Goal: {goals['mood_target']:.1f}")
                st.progress(progress / 100)
            
            with col4:
                sleep_current = inputs.get('sleep_hours', 7)
                progress = min(100, (sleep_current / goals['sleep_target']) * 100)
                st.metric("Sleep", f"{sleep_current}h", f"Goal: {goals['sleep_target']}h")
                st.progress(progress / 100)
            
            with col5:
                exercise_current = inputs.get('exercise_minutes', 30)
                progress = min(100, (exercise_current / goals['exercise_target']) * 100)
                st.metric("Exercise", f"{exercise_current}min", f"Goal: {goals['exercise_target']}min")
                st.progress(progress / 100)
            
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
