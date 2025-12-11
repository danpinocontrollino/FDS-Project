"""
================================================================================
MULTI-TARGET MENTAL HEALTH PREDICTION - Inference Script
================================================================================
Load trained mental health models and make predictions on new data.

Works with models trained by train_mental_health.py:
- mental_health_lstm.pt
- mental_health_transformer.pt

USAGE:
    python scripts/predict_mental_health.py --interactive
    python scripts/predict_mental_health.py --csv data.csv
    python scripts/predict_mental_health.py --user-id 42

Author: University Project - Mental Health Prediction
================================================================================
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Feature columns (must match training)
FEATURE_COLS = [
    "sleep_hours", "sleep_quality", "work_hours", "meetings_count",
    "tasks_completed", "emails_received", "commute_minutes", "exercise_minutes",
    "steps_count", "caffeine_mg", "alcohol_units", "screen_time_hours",
    "social_interactions", "outdoor_time_minutes", "diet_quality",
    "work_pressure", "weather_mood_impact",
]

# All targets
DAILY_TARGETS = ["stress_level", "mood_score", "energy_level", "focus_score"]
WEEKLY_TARGETS = ["perceived_stress_scale", "anxiety_score", "depression_score", "job_satisfaction"]
ALL_TARGETS = DAILY_TARGETS + WEEKLY_TARGETS

# Risk thresholds (from training)
RISK_THRESHOLDS = {
    "stress_level": 6,
    "mood_score": 4,
    "energy_level": 4,
    "focus_score": 4,
    "perceived_stress_scale": 20,
    "anxiety_score": 6,
    "depression_score": 15,
    "job_satisfaction": 4,
}

INVERTED_TARGETS = {"mood_score", "energy_level", "focus_score", "job_satisfaction"}

# Human-readable labels
TARGET_LABELS = {
    "stress_level": "Daily Stress",
    "mood_score": "Mood Score",
    "energy_level": "Energy Level",
    "focus_score": "Focus Score",
    "perceived_stress_scale": "Perceived Stress (PSS)",
    "anxiety_score": "Anxiety Score",
    "depression_score": "Depression Score",
    "job_satisfaction": "Job Satisfaction",
}


# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.regression = nn.Linear(hidden_dim, 1)
        self.classification = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        h = self.shared(x)
        return self.regression(h).squeeze(-1), self.classification(h).squeeze(-1)


class MentalHealthPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = "lstm",
        targets: List[str] = None,
    ):
        super().__init__()
        self.targets = targets or ALL_TARGETS
        self.encoder_type = encoder_type
        
        if encoder_type == "lstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                   batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        elif encoder_type == "gru":
            self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                                  batch_first=True, dropout=0.2 if num_layers > 1 else 0)
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
        if self.encoder_type in ["lstm", "gru"]:
            out, _ = self.encoder(x)
            return out[:, -1, :]
        else:
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.encoder(x)
            return x.mean(dim=1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.encode(x)
        h = self.shared_repr(h)
        return {target: self.heads[target](h) for target in self.targets}


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path: Path = None) -> Tuple[MentalHealthPredictor, dict]:
    """Load a trained mental-health model and return model + metadata.

    I attempt to locate a serialized checkpoint (LSTM preferred, then
    transformer), restore the model weights, and return a small `stats`
    dict that contains scaler parameters and feature ordering to ensure
    consistent preprocessing during inference.
    """
    if model_path is None:
        # Try LSTM first, then transformer
        model_path = MODEL_DIR / "mental_health_lstm.pt"
        if not model_path.exists():
            model_path = MODEL_DIR / "mental_health_transformer.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train with train_mental_health.py first.")
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    model = MentalHealthPredictor(
        input_dim=len(checkpoint.get("feature_cols", FEATURE_COLS)),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        encoder_type=checkpoint.get("model_type", "lstm"),
        targets=checkpoint.get("targets", ALL_TARGETS),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    stats = {
        "scaler_mean": checkpoint.get("scaler_mean"),
        "scaler_std": checkpoint.get("scaler_std"),
        "feature_cols": checkpoint.get("feature_cols", FEATURE_COLS),
        "targets": checkpoint.get("targets", ALL_TARGETS),
        "risk_thresholds": checkpoint.get("risk_thresholds", RISK_THRESHOLDS),
    }
    
    print(f"✅ Loaded {checkpoint.get('model_type', 'lstm').upper()} model from {model_path}")
    return model, stats


# ============================================================================
# PREDICTION
# ============================================================================

def predict(model: MentalHealthPredictor, sequence: np.ndarray, stats: dict) -> Dict[str, dict]:
    """Run inference on a 7-day sequence and return structured results.

    The returned mapping contains for each target: the continuous
    `value`, a probabilistic `risk_prob` from the classification head,
    and a boolean `at_risk` computed using the provided thresholds.
    """
    # Normalize
    mean = stats.get("scaler_mean")
    std = stats.get("scaler_std")
    
    if mean is not None and std is not None:
        sequence = (sequence - mean) / (std + 1e-8)
    
    # Convert to tensor
    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(x)
    
    results = {}
    thresholds = stats.get("risk_thresholds", RISK_THRESHOLDS)
    
    for target in stats.get("targets", ALL_TARGETS):
        reg_pred, cls_logit = outputs[target]
        value = reg_pred.item()
        risk_prob = torch.sigmoid(cls_logit).item()
        
        threshold = thresholds.get(target, 5)
        if target in INVERTED_TARGETS:
            at_risk = value <= threshold
        else:
            at_risk = value >= threshold
        
        results[target] = {
            "value": value,
            "risk_prob": risk_prob,
            "at_risk": at_risk,
            "threshold": threshold,
        }
    
    return results


def create_sequence_from_data(data: pd.DataFrame, feature_cols: List[str], window: int = 7) -> np.ndarray:
    """Construct the model input sequence from the most recent `window` days.

    I validate that at least `window` days are present and order features to
    match the training `feature_cols` to avoid silent feature-misalignment
    that could invalidate predictions.
    """
    if len(data) < window:
        raise ValueError(f"Need at least {window} days of data, got {len(data)}")
    
    # Take last `window` days
    recent = data.tail(window)
    sequence = recent[feature_cols].values.astype(np.float32)
    return sequence


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def print_dashboard(results: Dict[str, dict]) -> None:
    """Print a mental health dashboard."""
    print("\n" + "=" * 70)
    print("🧠 MENTAL HEALTH PREDICTION DASHBOARD")
    print("=" * 70)
    
    # Daily predictions
    print("\n📅 DAILY INDICATORS (next-day prediction)")
    print("-" * 50)
    
    for target in DAILY_TARGETS:
        if target not in results:
            continue
        r = results[target]
        label = TARGET_LABELS.get(target, target)
        value = r["value"]
        risk = r["at_risk"]
        threshold = r["threshold"]
        
        icon = "🔴" if risk else "🟢"
        direction = "≤" if target in INVERTED_TARGETS else "≥"
        status = "AT RISK" if risk else "OK"
        
        print(f"  {icon} {label:20} {value:5.1f}  ({status}, threshold: {direction}{threshold})")
    
    # Weekly predictions  
    print("\n📊 WEEKLY INDICATORS (end-of-week prediction)")
    print("-" * 50)
    
    for target in WEEKLY_TARGETS:
        if target not in results:
            continue
        r = results[target]
        label = TARGET_LABELS.get(target, target)
        value = r["value"]
        risk = r["at_risk"]
        threshold = r["threshold"]
        
        icon = "🔴" if risk else "🟢"
        direction = "≤" if target in INVERTED_TARGETS else "≥"
        status = "AT RISK" if risk else "OK"
        
        print(f"  {icon} {label:20} {value:5.1f}  ({status}, threshold: {direction}{threshold})")
    
    # Overall assessment
    at_risk_count = sum(1 for r in results.values() if r["at_risk"])
    total = len(results)
    
    print("\n" + "-" * 50)
    if at_risk_count == 0:
        print("✨ OVERALL: All indicators healthy!")
    elif at_risk_count <= 2:
        print(f"⚠️  OVERALL: {at_risk_count}/{total} indicators at risk - monitor closely")
    else:
        print(f"🚨 OVERALL: {at_risk_count}/{total} indicators at risk - consider intervention")
    
    print("=" * 70 + "\n")


def generate_html_report(results: Dict[str, dict], output_path: str, name: str = "User") -> None:
    """Generate HTML report for mental health predictions."""
    from datetime import datetime
    
    at_risk_count = sum(1 for r in results.values() if r["at_risk"])
    total = len(results)
    
    if at_risk_count == 0:
        overall_status = ("Healthy", "🟢", "#28a745", "All indicators are within healthy ranges.")
    elif at_risk_count <= 2:
        overall_status = ("Caution", "🟡", "#ffc107", f"{at_risk_count}/{total} indicators need attention.")
    else:
        overall_status = ("At Risk", "🔴", "#dc3545", f"{at_risk_count}/{total} indicators at risk. Consider intervention.")
    
    # Build indicator cards
    daily_cards = ""
    for target in DAILY_TARGETS:
        if target not in results:
            continue
        r = results[target]
        label = TARGET_LABELS.get(target, target)
        color = "#dc3545" if r["at_risk"] else "#28a745"
        icon = "⚠️" if r["at_risk"] else "✓"
        
        daily_cards += f"""
        <div class="indicator-card" style="border-left: 4px solid {color}">
            <div class="indicator-header">
                <span class="indicator-icon">{icon}</span>
                <span class="indicator-name">{label}</span>
            </div>
            <div class="indicator-value" style="color: {color}">{r['value']:.1f}</div>
            <div class="indicator-threshold">Threshold: {r['threshold']}</div>
        </div>
        """
    
    weekly_cards = ""
    for target in WEEKLY_TARGETS:
        if target not in results:
            continue
        r = results[target]
        label = TARGET_LABELS.get(target, target)
        color = "#dc3545" if r["at_risk"] else "#28a745"
        icon = "⚠️" if r["at_risk"] else "✓"
        
        weekly_cards += f"""
        <div class="indicator-card" style="border-left: 4px solid {color}">
            <div class="indicator-header">
                <span class="indicator-icon">{icon}</span>
                <span class="indicator-name">{label}</span>
            </div>
            <div class="indicator-value" style="color: {color}">{r['value']:.1f}</div>
            <div class="indicator-threshold">Threshold: {r['threshold']}</div>
        </div>
        """
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Dashboard - {name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ text-align: center; color: white; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2.2rem; margin-bottom: 8px; }}
        .header p {{ opacity: 0.9; }}
        .card {{
            background: white;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }}
        .overall-status {{
            text-align: center;
            padding: 30px;
        }}
        .status-icon {{ font-size: 3.5rem; margin-bottom: 10px; }}
        .status-text {{ font-size: 1.8rem; font-weight: bold; margin-bottom: 10px; }}
        .status-desc {{ color: #666; font-size: 1.1rem; }}
        .section-title {{
            font-size: 1.3rem;
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .indicators-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .indicator-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 18px;
        }}
        .indicator-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }}
        .indicator-icon {{ font-size: 1.2rem; }}
        .indicator-name {{ font-weight: 500; color: #333; font-size: 0.95rem; }}
        .indicator-value {{ font-size: 2rem; font-weight: bold; margin-bottom: 5px; }}
        .indicator-threshold {{ font-size: 0.85rem; color: #888; }}
        .disclaimer {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            font-size: 0.9rem;
            color: #856404;
        }}
        .footer {{ text-align: center; color: white; opacity: 0.7; margin-top: 20px; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Mental Health Dashboard</h1>
            <p>Report for {name} • {datetime.now().strftime('%B %d, %Y')}</p>
        </div>
        
        <div class="card overall-status">
            <div class="status-icon">{overall_status[1]}</div>
            <div class="status-text" style="color: {overall_status[2]}">{overall_status[0]}</div>
            <div class="status-desc">{overall_status[3]}</div>
        </div>
        
        <div class="card">
            <div class="section-title">📅 Daily Indicators</div>
            <div class="indicators-grid">
                {daily_cards}
            </div>
        </div>
        
        <div class="card">
            <div class="section-title">📊 Weekly Indicators</div>
            <div class="indicators-grid">
                {weekly_cards}
            </div>
        </div>
        
        <div class="card">
            <div class="disclaimer">
                ⚠️ <strong>Disclaimer:</strong> This is a predictive model for educational purposes.
                For real mental health concerns, please consult a healthcare professional.
            </div>
        </div>
        
        <div class="footer">
            Generated by Multi-Target Mental Health Prediction System
        </div>
    </div>
</body>
</html>"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"✅ HTML report saved to: {output_path}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_user_data(user_id: int, window: int = 7) -> pd.DataFrame:
    """Load data for a specific user from processed data."""
    daily_path = PROCESSED_DIR / "daily_with_burnout.parquet"
    
    if not daily_path.exists():
        raise FileNotFoundError("Run scripts/create_burnout_labels.py first")
    
    daily = pd.read_parquet(daily_path)
    
    # Filter to user
    user_data = daily[daily["user_id"] == user_id].copy()
    if len(user_data) == 0:
        raise ValueError(f"User {user_id} not found in data")
    
    user_data = user_data.sort_values("date").reset_index(drop=True)
    
    # Convert work_pressure if needed
    if user_data["work_pressure"].dtype == object:
        pressure_map = {"low": 0, "medium": 1, "high": 2}
        user_data["work_pressure"] = user_data["work_pressure"].map(pressure_map).fillna(1)
    
    return user_data


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Mental Health Prediction Dashboard")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--user-id", type=int, help="User ID to predict for")
    parser.add_argument("--csv", type=str, help="CSV file with behavioral data")
    parser.add_argument("--html", type=str, help="Output path for HTML report")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    model_path = Path(args.model_path) if args.model_path else None
    model, stats = load_model(model_path)
    feature_cols = stats.get("feature_cols", FEATURE_COLS)
    
    # Get data
    if args.user_id:
        print(f"Loading data for user {args.user_id}...")
        data = load_user_data(args.user_id)
        sequence = create_sequence_from_data(data, feature_cols)
        name = f"User {args.user_id}"
        
    elif args.csv:
        print(f"Loading data from {args.csv}...")
        data = pd.read_csv(args.csv)
        sequence = create_sequence_from_data(data, feature_cols)
        name = Path(args.csv).stem
        
    else:
        # Demo: use random user
        print("No input specified. Using random user from dataset...")
        daily_path = PROCESSED_DIR / "daily_with_burnout.parquet"
        if daily_path.exists():
            daily = pd.read_parquet(daily_path)
            user_id = np.random.choice(daily["user_id"].unique())
            data = load_user_data(user_id)
            sequence = create_sequence_from_data(data, feature_cols)
            name = f"User {user_id}"
        else:
            print("❌ No data found. Run scripts/create_burnout_labels.py first.")
            return
    
    # Predict
    results = predict(model, sequence, stats)
    
    # Output
    print_dashboard(results)
    
    if args.html:
        generate_html_report(results, args.html, name)


if __name__ == "__main__":
    main()
