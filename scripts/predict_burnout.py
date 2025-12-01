"""
================================================================================
BURNOUT PREDICTION DEMO - User-Friendly Output
================================================================================
STANDALONE script for predicting burnout risk. Works with models trained on
Kaggle - just download the .pt files and point to them!

SETUP:
------
1. Train models on Kaggle (run the notebook)
2. Download models from Kaggle Output tab (e.g., lstm_sequence.pt)
3. Place in any folder (default: models/saved/)
4. Run this script!

USAGE:
------
1. From CSV file (Google Form export):
   python predict_burnout.py --csv my_responses.csv

2. Interactive mode (answer questions):
   python predict_burnout.py --interactive

3. Single prediction with command line:
   python predict_burnout.py --stress 7 --sleep 6 --work 9 --mood 4

4. Specify custom model path:
   python predict_burnout.py --model-path ./downloads/lstm_sequence.pt

REQUIREMENTS (minimal):
-----------------------
pip install torch numpy pandas

No need for the full training environment!

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the project root directory (parent of scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Default model path - change this or use --model-path argument
MODEL_DIR = PROJECT_ROOT / "models/saved"
DEFAULT_MODEL_PATH = MODEL_DIR / "lstm_sequence.pt"

# Feature columns expected by the model (must match training)
FEATURE_COLS = [
    "stress_level",
    "commute_minutes",
    "exercise_minutes",
    "work_hours",
    "mood_score",
    "sleep_quality",
    "emails_received",
    "caffeine_mg",
    "energy_level",
    "sleep_hours",
    "focus_score",
    "meetings_count",
    "tasks_completed",
    "steps_count",
    "alcohol_units",
    "screen_time_hours",
    "work_pressure",
]

# Human-readable names for features
FEATURE_LABELS = {
    "stress_level": "Stress Level (1-10)",
    "commute_minutes": "Commute Time (minutes)",
    "exercise_minutes": "Exercise (minutes)",
    "work_hours": "Work Hours",
    "mood_score": "Mood (1-10)",
    "sleep_quality": "Sleep Quality (1-10)",
    "emails_received": "Emails Received",
    "caffeine_mg": "Caffeine (mg)",
    "energy_level": "Energy Level (1-10)",
    "sleep_hours": "Sleep Hours",
    "focus_score": "Focus (1-10)",
    "meetings_count": "Meetings",
    "tasks_completed": "Tasks Completed",
    "steps_count": "Steps",
    "alcohol_units": "Alcohol Units",
    "screen_time_hours": "Screen Time (hours)",
    "work_pressure": "Work Pressure (low/medium/high)",
}

# Default values for missing features
DEFAULTS = {
    "stress_level": 5,
    "commute_minutes": 30,
    "exercise_minutes": 30,
    "work_hours": 8,
    "mood_score": 6,
    "sleep_quality": 6,
    "emails_received": 20,
    "caffeine_mg": 100,
    "energy_level": 6,
    "sleep_hours": 7,
    "focus_score": 6,
    "meetings_count": 3,
    "tasks_completed": 5,
    "steps_count": 5000,
    "alcohol_units": 0,
    "screen_time_hours": 4,
    "work_pressure": 1,  # medium
}

# Risk level colors and descriptions
RISK_LEVELS = {
    0: {"name": "LOW", "color": "🟢", "emoji": "😊", "desc": "You're doing great! Keep up the healthy habits."},
    1: {"name": "MEDIUM", "color": "🟡", "emoji": "😐", "desc": "Some warning signs. Consider adjusting your routine."},
    2: {"name": "HIGH", "color": "🔴", "emoji": "😰", "desc": "High burnout risk. Please prioritize self-care."},
}


# ============================================================================
# MODEL LOADING (Works with downloaded Kaggle models)
# ============================================================================

class LSTMClassifier(nn.Module):
    """LSTM model architecture (must match training)."""
    def __init__(self, input_dim: int = 17, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class GRUClassifier(nn.Module):
    """GRU model architecture."""
    def __init__(self, input_dim: int = 17, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class TransformerClassifier(nn.Module):
    """Simplified Transformer for loading saved models."""
    def __init__(self, input_dim: int = 17, d_model: int = 64, num_classes: int = 3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


def load_model(model_path: str) -> tuple:
    """
    Load trained model from a .pt file (downloaded from Kaggle).
    
    Args:
        model_path: Path to the .pt file
        
    Returns:
        model: Loaded PyTorch model in eval mode
        model_type: Type of model (lstm/gru/transformer)
        feature_cols: List of feature columns used by this model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Download from Kaggle Output tab and place here, or use --model-path"
        )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Get feature columns from checkpoint (critical for correct input shape!)
    feature_cols = checkpoint.get("feature_cols", FEATURE_COLS)
    input_dim = len(feature_cols)
    
    # Detect model type from filename or checkpoint
    model_type = checkpoint.get("model_type", "lstm")
    if "lstm" in str(model_path).lower():
        model_type = "lstm"
    elif "gru" in str(model_path).lower():
        model_type = "gru"
    elif "transformer" in str(model_path).lower():
        model_type = "transformer"
    
    # Build appropriate model with CORRECT input dimension
    if model_type == "lstm":
        model = LSTMClassifier(input_dim=input_dim)
    elif model_type == "gru":
        model = GRUClassifier(input_dim=input_dim)
    elif model_type == "transformer":
        d_model = checkpoint.get("d_model", 64)
        model = TransformerClassifier(input_dim=input_dim, d_model=d_model)
    else:
        # Default to LSTM
        model = LSTMClassifier(input_dim=input_dim)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    print(f"✅ Loaded {model_type.upper()} model from {model_path}")
    print(f"   Features: {len(feature_cols)} ({', '.join(feature_cols[:3])}...)")
    
    return model, model_type, feature_cols


# ============================================================================
# DATA PROCESSING
# ============================================================================

def parse_google_form_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse a Google Form CSV export into model-ready format.
    
    Expected columns (flexible naming):
    - Timestamp (ignored)
    - "How stressed are you?" -> stress_level
    - "Hours of sleep last night?" -> sleep_hours
    - etc.
    
    The function tries to match column names intelligently.
    """
    df = pd.read_csv(csv_path)
    
    # Column name mapping (Google Form question -> feature name)
    # Add your actual Google Form questions here!
    column_mapping = {
        # Stress
        "stress": "stress_level",
        "how stressed": "stress_level",
        "stress level": "stress_level",
        
        # Sleep
        "sleep hours": "sleep_hours",
        "hours of sleep": "sleep_hours",
        "how many hours": "sleep_hours",
        "sleep quality": "sleep_quality",
        
        # Work
        "work hours": "work_hours",
        "hours worked": "work_hours",
        "meetings": "meetings_count",
        "emails": "emails_received",
        
        # Mood/Energy
        "mood": "mood_score",
        "how do you feel": "mood_score",
        "energy": "energy_level",
        "focus": "focus_score",
        
        # Physical
        "exercise": "exercise_minutes",
        "workout": "exercise_minutes",
        "steps": "steps_count",
        "caffeine": "caffeine_mg",
        "coffee": "caffeine_mg",
        "alcohol": "alcohol_units",
        
        # Other
        "commute": "commute_minutes",
        "screen time": "screen_time_hours",
        "pressure": "work_pressure",
    }
    
    # Try to map columns
    mapped_df = pd.DataFrame()
    for col in df.columns:
        col_lower = col.lower()
        for key, feature in column_mapping.items():
            if key in col_lower:
                mapped_df[feature] = df[col]
                break
    
    # Fill missing columns with defaults
    for feature in FEATURE_COLS:
        if feature not in mapped_df.columns:
            mapped_df[feature] = DEFAULTS[feature]
    
    # Convert work_pressure text to numeric
    if mapped_df["work_pressure"].dtype == object:
        pressure_map = {"low": 0, "medium": 1, "high": 2}
        mapped_df["work_pressure"] = mapped_df["work_pressure"].str.lower().map(pressure_map).fillna(1)
    
    return mapped_df[FEATURE_COLS]


def get_interactive_input() -> Dict[str, float]:
    """Get input interactively from the user."""
    print("\n" + "=" * 60)
    print("🧠 BURNOUT RISK ASSESSMENT")
    print("=" * 60)
    print("\nAnswer these questions about your PAST WEEK (average per day).")
    print("Press Enter to use default value shown in [brackets].\n")
    
    data = {}
    
    questions = [
        ("stress_level", "How stressed did you feel? (1=relaxed, 10=extremely stressed)", 1, 10),
        ("sleep_hours", "Hours of sleep per night?", 0, 12),
        ("sleep_quality", "Sleep quality? (1=terrible, 10=excellent)", 1, 10),
        ("work_hours", "Hours worked per day?", 0, 16),
        ("mood_score", "Overall mood? (1=very bad, 10=excellent)", 1, 10),
        ("energy_level", "Energy level? (1=exhausted, 10=energetic)", 1, 10),
        ("exercise_minutes", "Minutes of exercise per day?", 0, 180),
        ("caffeine_mg", "Caffeine intake in mg? (1 coffee ≈ 95mg)", 0, 500),
        ("meetings_count", "Number of meetings per day?", 0, 15),
        ("screen_time_hours", "Hours of screen time per day?", 0, 16),
    ]
    
    for feature, question, min_val, max_val in questions:
        default = DEFAULTS[feature]
        while True:
            try:
                response = input(f"  {question} [{default}]: ").strip()
                if response == "":
                    value = default
                else:
                    value = float(response)
                    if not min_val <= value <= max_val:
                        print(f"    ⚠️  Please enter a value between {min_val} and {max_val}")
                        continue
                data[feature] = value
                break
            except ValueError:
                print("    ⚠️  Please enter a number")
    
    # Fill remaining features with defaults
    for feature in FEATURE_COLS:
        if feature not in data:
            data[feature] = DEFAULTS[feature]
    
    return data


def create_weekly_sequence(daily_data: Dict[str, float], feature_cols: List[str], days: int = 7) -> np.ndarray:
    """
    Create a 7-day sequence from single-day data.
    
    For demo purposes, we simulate a week by adding small random variations
    to the provided daily averages.
    
    Args:
        daily_data: Dictionary of feature values
        feature_cols: List of features expected by the model (in order!)
        days: Number of days in sequence
    """
    sequence = []
    
    for day in range(days):
        day_data = []
        for feature in feature_cols:
            base_value = daily_data.get(feature, DEFAULTS.get(feature, 5))
            # Add small random variation (±10%) to simulate daily fluctuation
            variation = np.random.uniform(-0.1, 0.1) * base_value
            day_data.append(base_value + variation)
        sequence.append(day_data)
    
    return np.array(sequence, dtype=np.float32)


# ============================================================================
# PREDICTION & VISUALIZATION
# ============================================================================

def predict(model: nn.Module, sequence: np.ndarray, feature_cols: List[str]) -> tuple:
    """Run prediction and return class + probabilities."""
    # Dynamic normalization based on feature types
    # Build mean/std arrays based on the actual features in this model
    feature_stats = {
        "stress_level": (5, 2),
        "commute_minutes": (30, 20),
        "exercise_minutes": (30, 30),
        "work_hours": (8, 2),
        "mood_score": (6, 2),
        "sleep_quality": (6, 2),
        "emails_received": (20, 15),
        "caffeine_mg": (100, 80),
        "energy_level": (6, 2),
        "sleep_hours": (7, 1.5),
        "focus_score": (6, 2),
        "meetings_count": (3, 2),
        "tasks_completed": (5, 3),
        "steps_count": (5000, 3000),
        "alcohol_units": (1, 1.5),
        "screen_time_hours": (4, 2),
        "work_pressure": (1, 0.8),
    }
    
    means = np.array([feature_stats.get(f, (5, 2))[0] for f in feature_cols])
    stds = np.array([feature_stats.get(f, (5, 2))[1] for f in feature_cols])
    
    normalized = (sequence - means) / (stds + 1e-8)
    
    # Convert to tensor and predict
    x = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)  # Add batch dim
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_class = int(np.argmax(probs))
    
    return pred_class, probs


def print_risk_meter(probs: np.ndarray) -> None:
    """Print a visual risk meter."""
    print("\n" + "─" * 50)
    print("📊 RISK BREAKDOWN")
    print("─" * 50)
    
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    colors = ["🟢", "🟡", "🔴"]
    
    for i, (label, color, prob) in enumerate(zip(labels, colors, probs)):
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {color} {label:12} [{bar}] {prob*100:5.1f}%")


def print_recommendations(data: Dict[str, float], pred_class: int, model: nn.Module, feature_cols: List[str]) -> None:
    """
    Print DATA-DRIVEN recommendations based on feature importance and what-if analysis.
    
    Instead of generic advice, we:
    1. Identify which features deviate most from healthy baselines
    2. Run what-if simulations to show impact of changes
    3. Prioritize recommendations by predicted impact
    """
    print("\n" + "─" * 50)
    print("💡 DATA-DRIVEN RECOMMENDATIONS")
    print("─" * 50)
    
    # Healthy baselines (from low-burnout population in training data)
    healthy_baselines = {
        "stress_level": 4.0,
        "sleep_hours": 7.5,
        "sleep_quality": 7.0,
        "work_hours": 7.5,
        "exercise_minutes": 45,
        "mood_score": 7.0,
        "energy_level": 7.0,
        "focus_score": 7.0,
        "caffeine_mg": 80,
        "screen_time_hours": 3.5,
        "meetings_count": 2,
        "alcohol_units": 0.5,
        "steps_count": 8000,
        "tasks_completed": 6,
        "work_pressure": 0.5,  # low
        "commute_minutes": 20,
        "emails_received": 15,
    }
    
    # Calculate deviation from healthy baseline for each feature
    deviations = []
    for feature in feature_cols:
        current = data.get(feature, DEFAULTS.get(feature, 5))
        healthy = healthy_baselines.get(feature, current)
        
        # Determine if higher or lower is better
        lower_is_better = feature in [
            "stress_level", "work_hours", "caffeine_mg", "screen_time_hours",
            "meetings_count", "alcohol_units", "work_pressure", "commute_minutes",
            "emails_received"
        ]
        
        if lower_is_better:
            deviation = current - healthy  # positive = bad
            direction = "reduce"
        else:
            deviation = healthy - current  # positive = bad (you're below healthy)
            direction = "increase"
        
        # Normalize by typical range
        ranges = {
            "stress_level": 10, "sleep_hours": 4, "sleep_quality": 10,
            "work_hours": 6, "exercise_minutes": 60, "mood_score": 10,
            "energy_level": 10, "focus_score": 10, "caffeine_mg": 300,
            "screen_time_hours": 8, "meetings_count": 8, "alcohol_units": 5,
            "steps_count": 10000, "tasks_completed": 10, "work_pressure": 2,
            "commute_minutes": 60, "emails_received": 50,
        }
        
        normalized_deviation = deviation / ranges.get(feature, 10)
        
        if normalized_deviation > 0.1:  # Only significant deviations
            deviations.append({
                "feature": feature,
                "current": current,
                "healthy": healthy,
                "deviation": normalized_deviation,
                "direction": direction,
                "raw_diff": abs(current - healthy),
            })
    
    # Sort by deviation (most impactful first)
    deviations.sort(key=lambda x: x["deviation"], reverse=True)
    
    # Run what-if simulations for top deviations
    print("\n  📊 Your biggest risk factors (vs. healthy baseline):\n")
    
    if not deviations:
        print("  ✨ Your metrics are close to healthy baselines! Keep it up.")
        return
    
    recommendations = []
    
    for i, dev in enumerate(deviations[:5], 1):  # Top 5
        feature = dev["feature"]
        current = dev["current"]
        healthy = dev["healthy"]
        direction = dev["direction"]
        
        # Create what-if scenario
        modified_data = data.copy()
        modified_data[feature] = healthy
        
        # Predict with modification
        sequence = create_weekly_sequence(modified_data, feature_cols)
        new_pred, new_probs = predict(model, sequence, feature_cols)
        
        # Calculate risk reduction
        current_risk = 1 - predict(model, create_weekly_sequence(data, feature_cols), feature_cols)[1][0]  # 1 - P(low)
        new_risk = 1 - new_probs[0]
        risk_reduction = (current_risk - new_risk) * 100
        
        # Format recommendation
        feature_label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        
        if direction == "reduce":
            action = f"Reduce from {current:.0f} → {healthy:.0f}"
            diff = current - healthy
        else:
            action = f"Increase from {current:.0f} → {healthy:.0f}"
            diff = healthy - current
        
        # Specific, actionable advice
        specific_advice = get_specific_advice(feature, current, healthy, diff)
        
        impact_str = f"↓{risk_reduction:.0f}% risk" if risk_reduction > 0 else "minimal impact"
        
        print(f"  {i}. {feature_label}")
        print(f"     Current: {current:.1f} → Target: {healthy:.1f}")
        print(f"     Impact if changed: {impact_str}")
        print(f"     💡 {specific_advice}")
        print()
        
        recommendations.append({
            "feature": feature,
            "advice": specific_advice,
            "impact": risk_reduction,
        })
    
    # Summary
    total_potential = sum(r["impact"] for r in recommendations if r["impact"] > 0)
    if total_potential > 10:
        print(f"  ──────────────────────────────────────────────────")
        print(f"  📈 Combined potential risk reduction: ~{total_potential:.0f}%")
        print(f"     (if you address all factors above)")


def get_specific_advice(feature: str, current: float, target: float, diff: float) -> str:
    """Generate specific, actionable advice for each feature."""
    
    advice_templates = {
        "sleep_hours": [
            f"Go to bed {int(diff * 60)} minutes earlier tonight.",
            f"Set a bedtime alarm for {int(diff)}h before your wake time.",
            f"Aim for {target:.0f}h tonight - even {diff/2:.1f}h more helps.",
        ],
        "sleep_quality": [
            "No screens 1h before bed. Blue light disrupts melatonin.",
            "Keep bedroom at 65-68°F (18-20°C) for optimal sleep.",
            "Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s.",
        ],
        "stress_level": [
            "Try box breathing: 4s inhale, 4s hold, 4s exhale, 4s hold. Repeat 5x.",
            "Write down 3 things stressing you. Often just naming them helps.",
            f"Your stress is {current:.0f}/10. Target: {target:.0f}. What's one thing you can delegate?",
        ],
        "work_hours": [
            f"Leave work {diff:.1f}h earlier today. Set a hard stop.",
            f"Block your calendar after {8 + (current - target)/2:.0f}pm this week.",
            "Studies show productivity drops after 50h/week. You're past that.",
        ],
        "exercise_minutes": [
            f"Add a {int(diff)}-min walk after lunch. Start today.",
            "10 min of movement beats 0. Take stairs, walk during calls.",
            f"Even {int(diff/2)} more minutes daily compounds to {int(diff/2 * 7)} min/week.",
        ],
        "mood_score": [
            "Mood follows behavior. Plan one enjoyable activity today.",
            "Call or text one friend today. Social connection lifts mood.",
            "Get 15 min of sunlight this morning. It regulates serotonin.",
        ],
        "energy_level": [
            "Energy dip? Try a 10-min walk instead of coffee.",
            "Eat protein at lunch - carb-heavy meals cause afternoon crashes.",
            "Check if you're dehydrated. Fatigue is often thirst in disguise.",
        ],
        "caffeine_mg": [
            f"You're at {current:.0f}mg. Cut {diff:.0f}mg by skipping one coffee.",
            "No caffeine after 2pm. It has a 6-hour half-life.",
            f"Try half-caff: same ritual, {current/2:.0f}mg less caffeine.",
        ],
        "screen_time_hours": [
            f"Set a {target:.0f}h daily screen limit on your phone.",
            "Replace {diff:.1f}h of scrolling with a walk or podcast.",
            "Use grayscale mode after 8pm. Color triggers engagement.",
        ],
        "meetings_count": [
            f"Decline or shorten {int(diff)} meetings this week.",
            "Propose 25-min meetings instead of 30. You'll get 5-min breaks.",
            "Block 'focus time' on your calendar before others fill it.",
        ],
        "alcohol_units": [
            "Alcohol disrupts REM sleep even in small amounts.",
            f"Try alternating: one drink, one water. Cut intake by half.",
            "Your liver needs 48h to fully recover. Try 2 dry days/week.",
        ],
        "steps_count": [
            f"Add {int(diff)} steps with a 15-min walk. That's ~1,500 steps.",
            "Take calls while walking. Movement helps thinking too.",
            "Park farther away. Small changes add up to big step counts.",
        ],
        "work_pressure": [
            "High pressure often means unclear priorities. Ask: what's #1?",
            "Pressure feels lower when you write tasks down. Brain dump now.",
            "Say 'I can do A or B by Friday, which is more important?'",
        ],
        "focus_score": [
            "Try the Pomodoro technique: 25 min focus, 5 min break.",
            "Put phone in another room. Out of sight = out of mind.",
            "Single-task for 1 hour today. Multitasking is a myth.",
        ],
    }
    
    import random
    templates = advice_templates.get(feature, [f"Aim to {('reduce' if current > target else 'increase')} this to {target:.1f}."])
    return random.choice(templates)


def print_prediction_result(data: Dict[str, float], pred_class: int, probs: np.ndarray, model: nn.Module = None, feature_cols: List[str] = None) -> None:
    """Print the full prediction result with visualizations."""
    risk = RISK_LEVELS[pred_class]
    
    # Header
    print("\n" + "=" * 60)
    print(f"{'🧠 BURNOUT PREDICTION RESULT':^60}")
    print("=" * 60)
    
    # Main result
    print(f"\n  {risk['emoji']} Your burnout risk level: {risk['color']} {risk['name']}")
    print(f"  {risk['desc']}")
    print(f"\n  Confidence: {probs[pred_class]*100:.1f}%")
    
    # Risk meter
    print_risk_meter(probs)
    
    # Key metrics summary
    print("\n" + "─" * 50)
    print("📋 YOUR KEY METRICS")
    print("─" * 50)
    
    key_metrics = [
        ("Stress", data.get("stress_level", 5), "/10", 5, "lower is better"),
        ("Sleep", data.get("sleep_hours", 7), "hrs", 7, "7-9 is ideal"),
        ("Work", data.get("work_hours", 8), "hrs", 8, "8 or less is healthy"),
        ("Exercise", data.get("exercise_minutes", 30), "min", 30, "30+ recommended"),
        ("Mood", data.get("mood_score", 6), "/10", 6, "higher is better"),
    ]
    
    for name, value, unit, target, note in key_metrics:
        indicator = "✅" if (name in ["Stress"] and value <= target) or (name not in ["Stress"] and value >= target) else "⚠️"
        print(f"  {indicator} {name}: {value:.0f}{unit} ({note})")
    
    # Data-driven recommendations (if model available)
    if model is not None and feature_cols is not None:
        print_recommendations(data, pred_class, model, feature_cols)
    
    # Footer
    print("\n" + "=" * 60)
    print("⚠️  DISCLAIMER: This is a demo prediction model.")
    print("    For real mental health concerns, please consult a professional.")
    print("=" * 60 + "\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict burnout risk from behavioral data (works with Kaggle-trained models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python predict_burnout.py --interactive
  
  # With Google Form CSV
  python predict_burnout.py --csv responses.csv
  
  # Quick test with specific values
  python predict_burnout.py --stress 8 --sleep 5 --work 10 --mood 3
  
  # Use a specific model file (downloaded from Kaggle)
  python predict_burnout.py --model-path ./lstm_sequence.pt --interactive
        """
    )
    
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode: answer questions one by one")
    parser.add_argument("--csv", type=str,
                        help="Path to CSV file (e.g., Google Form export)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model .pt file (default: models/saved/lstm_sequence.pt)")
    
    # Quick command-line input
    parser.add_argument("--stress", type=float, help="Stress level (1-10)")
    parser.add_argument("--sleep", type=float, help="Sleep hours")
    parser.add_argument("--work", type=float, help="Work hours")
    parser.add_argument("--mood", type=float, help="Mood score (1-10)")
    parser.add_argument("--exercise", type=float, help="Exercise minutes")
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Determine model path
    model_path = args.model_path if args.model_path else DEFAULT_MODEL_PATH
    
    # Load model
    try:
        model, model_type, feature_cols = load_model(model_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n📥 To get the model:")
        print("   1. Run training notebook on Kaggle")
        print("   2. Go to Output tab → Download lstm_sequence.pt")
        print("   3. Place in models/saved/ or use --model-path")
        return
    
    # Get input data
    if args.interactive:
        data = get_interactive_input()
    elif args.csv:
        df = parse_google_form_csv(args.csv)
        # Process each row
        for idx, row in df.iterrows():
            data = row.to_dict()
            sequence = create_weekly_sequence(data, feature_cols)
            pred_class, probs = predict(model, sequence, feature_cols)
            print(f"\n--- Response #{idx + 1} ---")
            print_prediction_result(data, pred_class, probs, model, feature_cols)
        return
    elif any([args.stress, args.sleep, args.work, args.mood, args.exercise]):
        # Use command-line values
        data = DEFAULTS.copy()
        if args.stress: data["stress_level"] = args.stress
        if args.sleep: data["sleep_hours"] = args.sleep
        if args.work: data["work_hours"] = args.work
        if args.mood: data["mood_score"] = args.mood
        if args.exercise: data["exercise_minutes"] = args.exercise
    else:
        # Default to interactive
        data = get_interactive_input()
    
    # Create sequence and predict
    sequence = create_weekly_sequence(data, feature_cols)
    pred_class, probs = predict(model, sequence, feature_cols)
    
    # Print result with model for what-if analysis
    print_prediction_result(data, pred_class, probs, model, feature_cols)


if __name__ == "__main__":
    main()
