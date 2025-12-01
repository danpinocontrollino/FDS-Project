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

# Default model path - change this or use --model-path argument
MODEL_DIR = Path("models/saved")
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
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Download from Kaggle Output tab and place here, or use --model-path"
        )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Detect model type from filename or checkpoint
    model_type = checkpoint.get("model_type", "lstm")
    if "lstm" in str(model_path).lower():
        model_type = "lstm"
    elif "gru" in str(model_path).lower():
        model_type = "gru"
    elif "transformer" in str(model_path).lower():
        model_type = "transformer"
    
    # Build appropriate model
    if model_type == "lstm":
        model = LSTMClassifier()
    elif model_type == "gru":
        model = GRUClassifier()
    elif model_type == "transformer":
        d_model = checkpoint.get("d_model", 64)
        model = TransformerClassifier(d_model=d_model)
    else:
        # Default to LSTM
        model = LSTMClassifier()
    
    # Load weights
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    print(f"✅ Loaded {model_type.upper()} model from {model_path}")
    
    return model, model_type


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


def create_weekly_sequence(daily_data: Dict[str, float], days: int = 7) -> np.ndarray:
    """
    Create a 7-day sequence from single-day data.
    
    For demo purposes, we simulate a week by adding small random variations
    to the provided daily averages.
    """
    sequence = []
    
    for day in range(days):
        day_data = []
        for feature in FEATURE_COLS:
            base_value = daily_data.get(feature, DEFAULTS[feature])
            # Add small random variation (±10%) to simulate daily fluctuation
            variation = np.random.uniform(-0.1, 0.1) * base_value
            day_data.append(base_value + variation)
        sequence.append(day_data)
    
    return np.array(sequence, dtype=np.float32)


# ============================================================================
# PREDICTION & VISUALIZATION
# ============================================================================

def predict(model: nn.Module, sequence: np.ndarray) -> tuple:
    """Run prediction and return class + probabilities."""
    # Normalize (simple z-score based on typical ranges)
    # In production, use saved scaler from training
    means = np.array([5, 30, 30, 8, 6, 6, 20, 100, 6, 7, 6, 3, 5, 5000, 1, 4, 1])
    stds = np.array([2, 20, 30, 2, 2, 2, 15, 80, 2, 1.5, 2, 2, 3, 3000, 1.5, 2, 0.8])
    
    normalized = (sequence - means) / (stds + 1e-8)
    
    # Convert to tensor and predict
    x = torch.from_numpy(normalized).unsqueeze(0)  # Add batch dim
    
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


def print_recommendations(data: Dict[str, float], pred_class: int) -> None:
    """Print personalized recommendations based on input data."""
    print("\n" + "─" * 50)
    print("💡 PERSONALIZED RECOMMENDATIONS")
    print("─" * 50)
    
    recommendations = []
    
    # Analyze each factor
    if data.get("sleep_hours", 7) < 7:
        recommendations.append("😴 Try to get at least 7 hours of sleep. Sleep debt compounds over time.")
    
    if data.get("stress_level", 5) > 6:
        recommendations.append("🧘 High stress detected. Consider daily meditation or breathing exercises.")
    
    if data.get("exercise_minutes", 30) < 20:
        recommendations.append("🏃 Aim for at least 20-30 minutes of exercise daily. Even walking helps!")
    
    if data.get("work_hours", 8) > 9:
        recommendations.append("⏰ Working long hours increases burnout risk. Try to set boundaries.")
    
    if data.get("caffeine_mg", 100) > 300:
        recommendations.append("☕ High caffeine intake can affect sleep. Consider reducing after 2 PM.")
    
    if data.get("screen_time_hours", 4) > 6:
        recommendations.append("📱 High screen time. Take regular breaks and try the 20-20-20 rule.")
    
    if data.get("mood_score", 6) < 5:
        recommendations.append("💬 Low mood persisting? Consider talking to someone you trust.")
    
    if not recommendations:
        recommendations.append("✨ Your habits look healthy! Keep maintaining this balance.")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")


def print_prediction_result(data: Dict[str, float], pred_class: int, probs: np.ndarray) -> None:
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
    
    # Recommendations
    print_recommendations(data, pred_class)
    
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
        model, model_type = load_model(model_path)
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
            sequence = create_weekly_sequence(data)
            pred_class, probs = predict(model, sequence)
            print(f"\n--- Response #{idx + 1} ---")
            print_prediction_result(data, pred_class, probs)
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
    sequence = create_weekly_sequence(data)
    pred_class, probs = predict(model, sequence)
    
    # Print result
    print_prediction_result(data, pred_class, probs)


if __name__ == "__main__":
    main()
