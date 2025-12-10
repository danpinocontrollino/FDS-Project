"""
================================================================================
TWO-STAGE PIPELINE: HTML PROFILE GENERATOR
================================================================================
Generates HTML reports from two-stage hybrid pipeline predictions.

FEATURES:
  ✓ Parses Google Form CSV with 7-day behavioral data
  ✓ Stage 1: Behavioral forecasting with uncertainty quantification
  ✓ Stage 2: Mental health inference from predicted behaviors
  ✓ Professional HTML reports with clear data source labels
  ✓ Warnings about error propagation and limitations
  ✓ Behavioral recommendations based on Stage 1 predictions

USAGE:
    python scripts/generate_two_stage_profile.py --csv data/form_responses.csv --all-users
    python scripts/generate_two_stage_profile.py --csv data/form.csv --user-id john@example.com

REQUIREMENTS:
  - Stage 1 model: models/saved/stage1_behavioral_forecasting.pt
  - Stage 2 model: models/saved/mental_health_lstm.pt
  - Google Form CSV with 6 behavioral features × 7 days

Author: University Project - Two-Stage Mental Health Prediction
================================================================================
"""

import argparse

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = Path("models/saved")
OUTPUT_DIR = Path("reports/two_stage_profiles")
STAGE1_MODEL = MODEL_DIR / "stage1_behavioral_forecasting.pt"
STAGE2_MODEL = MODEL_DIR / "mental_health_lstm.pt"

# Stage 1: Behavioral features (StudentLife)
STAGE1_FEATURES = [
    "sleep_hours",
    "work_hours",
    "exercise_minutes",
    "steps_count",
    "screen_time_hours",
    "social_interactions",
]

# Stage 1: Targets (behavioral predictions)
STAGE1_TARGETS = [
    "sleep_hours",
    "exercise_minutes",
    "screen_time_hours",
    "social_interactions",
    "steps_count",
]

# Stage 2: Targets (mental health)
STAGE2_TARGETS = [
    "stress_level",
    "mood_score",
    "energy_level",
    "focus_score",
    "perceived_stress_scale",
    "anxiety_score",
    "depression_score",
    "job_satisfaction",
]

# Target scales for display
TARGET_SCALES = {
    "stress_level": 10,
    "mood_score": 10,
    "energy_level": 10,
    "focus_score": 10,
    "perceived_stress_scale": 40,
    "anxiety_score": 21,
    "depression_score": 27,
    "job_satisfaction": 10,
}

# Inverted targets (higher = better)
INVERTED_TARGETS = {"mood_score", "energy_level", "focus_score", "job_satisfaction"}

# Risk thresholds
RISK_THRESHOLDS = {
    "stress_level": 7.0,  # >7/10 = high stress
    "anxiety_score": 10.0,  # >10/21 = moderate anxiety
    "depression_score": 14.0,  # >14/27 = moderate depression
    "perceived_stress_scale": 20.0,  # >20/40 = high stress
    "mood_score": 4.0,  # <4/10 = low mood (inverted)
    "energy_level": 4.0,  # <4/10 = low energy (inverted)
    "focus_score": 4.0,  # <4/10 = poor focus (inverted)
    "job_satisfaction": 5.0,  # <5/10 = low satisfaction (inverted)
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TwoStageProfile:
    """Complete two-stage prediction profile."""
    user_id: str
    timestamp: datetime
    
    # Input features (7 days)
    input_features: pd.DataFrame
    
    # Stage 1: Behavioral predictions
    stage1_predictions: Dict[str, float]
    stage1_uncertainties: Dict[str, float]
    stage1_avg_uncertainty_pct: float
    
    # Stage 2: Mental health predictions
    stage2_predictions: Dict[str, float]
    stage2_clipped: Dict[str, float]
    
    # Error propagation
    total_stage1_uncertainty: float
    confidence_note: str
    
    # Risk assessments
    at_risk_targets: List[str]
    healthy_targets: List[str]
    
    # Recommendations
    behavioral_recommendations: List[Dict[str, Any]]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_stage1_model(model_path: Path = STAGE1_MODEL):
    """Load Stage 1: Behavioral forecasting model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Stage 1 model not found: {model_path}")
    
    # Import model architecture
    import sys
    sys.path.append(str(Path(__file__).parent))
    from model_definitions import MentalHealthPredictor  # Reuse architecture
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Build model
    model = MentalHealthPredictor(
        input_dim=len(checkpoint['feature_cols']),
        hidden_dim=checkpoint.get('hidden_dim', 32),
        num_layers=checkpoint.get('num_layers', 1),
        encoder_type='lstm',
        targets=checkpoint['targets']
    )
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"✓ Stage 1 Model Loaded: {checkpoint['targets']}")
    return model, checkpoint


def load_stage2_model(model_path: Path = STAGE2_MODEL):
    """Load Stage 2: Mental health prediction model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Stage 2 model not found: {model_path}")
    
    import sys
    sys.path.append(str(Path(__file__).parent))
    from model_definitions import MentalHealthPredictor
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = MentalHealthPredictor(
        input_dim=len(checkpoint['feature_cols']),
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=2,
        encoder_type=checkpoint['model_type'],
        targets=checkpoint['targets']
    )
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"✓ Stage 2 Model Loaded: {checkpoint['targets']}")
    return model, checkpoint


# ============================================================================
# CSV PARSING
# ============================================================================

def parse_google_form_csv(csv_path: Path, window: int = 7) -> Dict[str, pd.DataFrame]:
    """
    Parse Google Form CSV with 7-day behavioral data.
    
    Expected columns (for each day 1-7):
      - Sleep Hours (Day X)
      - Work Hours (Day X)
      - Exercise Minutes (Day X)
      - Steps Count (Day X)
      - Screen Time Hours (Day X)
      - Social Interactions (Day X)
    
    Returns:
        Dict mapping user_id → DataFrame with 7 days × 6 features
    """
    df = pd.read_csv(csv_path)
    
    print(f"📂 Loaded CSV: {len(df)} responses, {len(df.columns)} columns")
    
    # Detect user ID column
    user_id_col = None
    for col in df.columns:
        if "timestamp" in col.lower() or "email" in col.lower() or "user" in col.lower():
            user_id_col = col
            break
    
    if user_id_col is None:
        print("⚠️  No user ID column found, using row index")
        df["user_id"] = df.index.astype(str)
        user_id_col = "user_id"
    
    # Build column mapping
    column_mapping = {}
    
    feature_patterns = {
        'sleep_hours': ['sleep hours', 'hours of sleep', 'sleep'],
        'work_hours': ['work hours', 'hours worked', 'working hours'],
        'exercise_minutes': ['exercise minutes', 'exercise', 'workout'],
        'steps_count': ['steps count', 'steps', 'step count'],
        'screen_time_hours': ['screen time hours', 'screen time', 'screen'],
        'social_interactions': ['social interactions', 'social', 'interactions'],
    }
    
    for day in range(1, window + 1):
        for feature in STAGE1_FEATURES:
            patterns = feature_patterns.get(feature, [feature.replace('_', ' ')])
            
            for pattern in patterns:
                search_patterns = [
                    f"day {day}",
                    f"(day {day})",
                    f"d{day}",
                ]
                
                matched = False
                for col in df.columns:
                    col_lower = col.lower()
                    if pattern in col_lower and any(sp in col_lower for sp in search_patterns):
                        column_mapping[col] = (feature, day)
                        matched = True
                        break
                if matched:
                    break
    
    print(f"✓ Mapped {len(column_mapping)} columns to features")
    
    # Parse each user's data
    user_data = {}
    
    for idx, row in df.iterrows():
        user_id = str(row[user_id_col])
        
        # Build 7-day sequence
        sequence = pd.DataFrame(index=range(window), columns=STAGE1_FEATURES, dtype=float)
        
        for col, (feature, day) in column_mapping.items():
            value = row[col]
            if pd.notna(value):
                try:
                    sequence.loc[day - 1, feature] = float(value)
                except (ValueError, TypeError):
                    pass
        
        # Fill missing values with reasonable defaults
        defaults = {
            "sleep_hours": 7.0,
            "work_hours": 8.0,
            "exercise_minutes": 30.0,
            "steps_count": 5000.0,
            "screen_time_hours": 6.0,
            "social_interactions": 10.0,
        }
        
        for feature in STAGE1_FEATURES:
            if sequence[feature].isna().all():
                sequence[feature] = defaults.get(feature, sequence[feature].mean())
            else:
                sequence[feature].fillna(sequence[feature].mean(), inplace=True)
        
        user_data[user_id] = sequence
    
    print(f"✓ Parsed {len(user_data)} user profiles")
    return user_data


# ============================================================================
# TWO-STAGE PREDICTION
# ============================================================================

def predict_two_stage(
    user_data: pd.DataFrame,
    stage1_model: nn.Module,
    stage1_checkpoint: Dict,
    stage2_model: nn.Module,
    stage2_checkpoint: Dict,
) -> Tuple[Dict, Dict, Dict]:
    """
    Run two-stage prediction pipeline.
    
    Returns:
        stage1_preds, stage1_uncertainties, stage2_preds
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === STAGE 1: Behavioral Forecasting ===
    
    # Normalize input
    X = user_data[stage1_checkpoint['feature_cols']].values
    scaler_mean = stage1_checkpoint['scaler_mean']
    scaler_std = stage1_checkpoint['scaler_std']
    X_normalized = (X - scaler_mean) / (scaler_std + 1e-8)
    
    # Predict
    X_tensor = torch.FloatTensor(X_normalized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        stage1_outputs = stage1_model(X_tensor)
    
    # Extract predictions and uncertainties
    stage1_preds = {}
    stage1_uncertainties = {}
    
    for target in stage1_checkpoint['targets']:
        reg_output, uncertainty_output = stage1_outputs[target]
        
        # Denormalize
        pred_normalized = reg_output.cpu().item()
        target_idx = stage1_checkpoint['targets'].index(target)
        target_mean = stage1_checkpoint['target_means'][target_idx]
        target_std = stage1_checkpoint['target_stds'][target_idx]
        
        pred_value = pred_normalized * target_std + target_mean
        uncertainty = torch.exp(uncertainty_output).cpu().item() * target_std
        
        stage1_preds[target] = pred_value
        stage1_uncertainties[target] = uncertainty
    
    # === STAGE 2: Mental Health Inference ===
    
    # Use Stage 1 predictions as input
    stage2_input = pd.DataFrame([stage1_preds])
    
    # Match Stage 2 feature order
    X2 = stage2_input[stage2_checkpoint['feature_cols']].values
    scaler_mean2 = stage2_checkpoint['scaler_mean']
    scaler_std2 = stage2_checkpoint['scaler_std']
    X2_normalized = (X2 - scaler_mean2) / (scaler_std2 + 1e-8)
    
    # Predict
    X2_tensor = torch.FloatTensor(X2_normalized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        stage2_outputs = stage2_model(X2_tensor)
    
    # Extract predictions
    stage2_preds = {}
    
    for target in stage2_checkpoint['targets']:
        reg_output, _ = stage2_outputs[target]
        
        # Denormalize
        pred_normalized = reg_output.cpu().item()
        target_idx = stage2_checkpoint['targets'].index(target)
        target_mean = stage2_checkpoint['target_means'][target_idx]
        target_std = stage2_checkpoint['target_stds'][target_idx]
        
        pred_value = pred_normalized * target_std + target_mean
        stage2_preds[target] = pred_value
    
    return stage1_preds, stage1_uncertainties, stage2_preds


# ============================================================================
# PROFILE GENERATION
# ============================================================================

def generate_profile(
    user_id: str,
    user_data: pd.DataFrame,
    stage1_model: nn.Module,
    stage1_checkpoint: Dict,
    stage2_model: nn.Module,
    stage2_checkpoint: Dict,
) -> TwoStageProfile:
    """Generate complete two-stage profile."""
    
    # Run predictions
    stage1_preds, stage1_uncs, stage2_preds = predict_two_stage(
        user_data, stage1_model, stage1_checkpoint, stage2_model, stage2_checkpoint
    )
    
    # Calculate average Stage 1 uncertainty percentage
    unc_pcts = []
    for target, unc in stage1_uncs.items():
        pred_val = stage1_preds[target]
        if pred_val > 0:
            unc_pcts.append((unc / pred_val) * 100)
    avg_unc_pct = np.mean(unc_pcts) if unc_pcts else 0
    
    # Clip Stage 2 predictions to valid ranges
    stage2_clipped = {}
    for target, value in stage2_preds.items():
        max_val = TARGET_SCALES.get(target, 10)
        stage2_clipped[target] = np.clip(value, 0, max_val)
    
    # Risk assessment
    at_risk = []
    healthy = []
    
    for target, value in stage2_clipped.items():
        threshold = RISK_THRESHOLDS.get(target)
        if threshold is None:
            continue
        
        is_inverted = target in INVERTED_TARGETS
        
        if is_inverted:
            if value < threshold:
                at_risk.append(target)
            else:
                healthy.append(target)
        else:
            if value > threshold:
                at_risk.append(target)
            else:
                healthy.append(target)
    
    # Behavioral recommendations
    recommendations = generate_behavioral_recommendations(stage1_preds, stage1_uncs)
    
    # Total Stage 1 uncertainty
    total_unc = sum(stage1_uncs.values())
    
    confidence_note = (
        f"Stage 1 predictions have ±{avg_unc_pct:.0f}% average uncertainty. "
        f"This uncertainty compounds in Stage 2 mental health predictions."
    )
    
    return TwoStageProfile(
        user_id=user_id,
        timestamp=datetime.now(),
        input_features=user_data,
        stage1_predictions=stage1_preds,
        stage1_uncertainties=stage1_uncs,
        stage1_avg_uncertainty_pct=avg_unc_pct,
        stage2_predictions=stage2_preds,
        stage2_clipped=stage2_clipped,
        total_stage1_uncertainty=total_unc,
        confidence_note=confidence_note,
        at_risk_targets=at_risk,
        healthy_targets=healthy,
        behavioral_recommendations=recommendations,
    )


def generate_behavioral_recommendations(
    stage1_preds: Dict[str, float],
    stage1_uncs: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Generate evidence-based behavioral recommendations."""
    recommendations = []
    
    # Sleep recommendation
    sleep = stage1_preds.get('sleep_hours', 7)
    if sleep < 7:
        recommendations.append({
            'category': 'Sleep',
            'priority': 'HIGH',
            'current': f"{sleep:.1f} hours/night",
            'target': '7-9 hours/night',
            'evidence': 'Sleep deprivation linked to increased stress, anxiety, and depression risk',
            'actions': [
                'Set consistent bedtime and wake time',
                'Avoid screens 1 hour before bed',
                'Keep bedroom cool and dark',
            ]
        })
    
    # Exercise recommendation
    exercise = stage1_preds.get('exercise_minutes', 30)
    if exercise < 30:
        recommendations.append({
            'category': 'Physical Activity',
            'priority': 'MEDIUM',
            'current': f"{exercise:.0f} minutes/day",
            'target': '30-60 minutes/day',
            'evidence': 'Regular exercise reduces anxiety and depression by 20-30%',
            'actions': [
                'Start with 10-minute walks, gradually increase',
                'Find enjoyable activities (dance, sports, cycling)',
                'Exercise with others for social benefits',
            ]
        })
    
    # Screen time recommendation
    screen = stage1_preds.get('screen_time_hours', 6)
    if screen > 8:
        recommendations.append({
            'category': 'Screen Time',
            'priority': 'MEDIUM',
            'current': f"{screen:.1f} hours/day",
            'target': '< 6 hours/day (non-work)',
            'evidence': 'Excessive screen time associated with poor sleep and mood',
            'actions': [
                'Use 20-20-20 rule: every 20 min, look 20 feet away for 20 seconds',
                'Set screen-free zones (bedroom, dining)',
                'Replace scrolling with walks or reading',
            ]
        })
    
    # Social interaction recommendation
    social = stage1_preds.get('social_interactions', 10)
    if social < 10:
        recommendations.append({
            'category': 'Social Connection',
            'priority': 'HIGH',
            'current': f"{social:.0f} interactions/day",
            'target': '10-20 meaningful interactions/day',
            'evidence': 'Social isolation increases depression risk by 50-100%',
            'actions': [
                'Schedule regular video calls with friends/family',
                'Join clubs or groups matching your interests',
                'Practice active listening in conversations',
            ]
        })
    
    return recommendations


# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_html_report(profile: TwoStageProfile, output_path: Path):
    """Generate professional HTML report with clear data source labels."""
    
    timestamp_str = profile.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Two-Stage Mental Health Profile - {profile.user_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .header .subtitle {{
            font-size: 18px;
            opacity: 0.95;
        }}
        
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px 40px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .metadata-item {{
            text-align: center;
        }}
        
        .metadata-label {{
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        
        .metadata-value {{
            font-size: 18px;
            font-weight: 600;
            color: #495057;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #2d3748;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section-subtitle {{
            font-size: 14px;
            color: #718096;
            margin-bottom: 20px;
            font-style: italic;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .badge-real {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-synthetic {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border: 2px solid #e9ecef;
        }}
        
        .card-header {{
            font-size: 13px;
            color: #6c757d;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .card-value {{
            font-size: 28px;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 5px;
        }}
        
        .card-uncertainty {{
            font-size: 12px;
            color: #e67e22;
            font-weight: 600;
        }}
        
        .prediction-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #6c757d;
        }}
        
        .prediction-card.at-risk {{
            border-color: #dc3545;
            background: #fff5f5;
        }}
        
        .prediction-card.healthy {{
            border-color: #28a745;
            background: #f0fff4;
        }}
        
        .prediction-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .prediction-name {{
            font-size: 16px;
            font-weight: 600;
            color: #2d3748;
        }}
        
        .prediction-value {{
            font-size: 24px;
            font-weight: 700;
        }}
        
        .prediction-value.at-risk {{
            color: #dc3545;
        }}
        
        .prediction-value.healthy {{
            color: #28a745;
        }}
        
        .recommendation-card {{
            background: white;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .recommendation-card.high {{
            border-color: #e74c3c;
        }}
        
        .recommendation-card.medium {{
            border-color: #f39c12;
        }}
        
        .recommendation-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .recommendation-title {{
            font-size: 18px;
            font-weight: 600;
            color: #2d3748;
        }}
        
        .priority-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .priority-badge.high {{
            background: #fee;
            color: #c00;
        }}
        
        .priority-badge.medium {{
            background: #fff4e6;
            color: #d68910;
        }}
        
        .recommendation-metrics {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        
        .metric-item {{
            display: flex;
            flex-direction: column;
        }}
        
        .metric-label {{
            font-size: 12px;
            color: #6c757d;
            margin-bottom: 4px;
        }}
        
        .metric-value {{
            font-size: 14px;
            font-weight: 600;
            color: #495057;
        }}
        
        .recommendation-evidence {{
            font-size: 14px;
            color: #495057;
            font-style: italic;
            margin-bottom: 15px;
            padding-left: 15px;
            border-left: 3px solid #e9ecef;
        }}
        
        .action-list {{
            list-style: none;
        }}
        
        .action-list li {{
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
            font-size: 14px;
            color: #495057;
        }}
        
        .action-list li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            color: #28a745;
            font-weight: bold;
        }}
        
        .warning-box {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .warning-box-title {{
            font-size: 16px;
            font-weight: 600;
            color: #856404;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .warning-box-content {{
            font-size: 14px;
            color: #856404;
            line-height: 1.6;
        }}
        
        .info-box {{
            background: #d1ecf1;
            border: 1px solid #0c5460;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .info-box-title {{
            font-size: 16px;
            font-weight: 600;
            color: #0c5460;
            margin-bottom: 10px;
        }}
        
        .info-box-content {{
            font-size: 14px;
            color: #0c5460;
            line-height: 1.6;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
                border-radius: 0;
                max-width: 100%;
            }}
            
            .header {{
                background: #667eea;
                color: white;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            
            .section {{
                page-break-inside: avoid;
            }}
            
            * {{
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🧠 Two-Stage Mental Health Profile</h1>
            <div class="subtitle">Hybrid Behavioral Forecasting + Mental Health Inference</div>
        </div>
        
        <!-- Metadata -->
        <div class="metadata">
            <div class="metadata-item">
                <div class="metadata-label">User ID</div>
                <div class="metadata-value">{profile.user_id}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Generated</div>
                <div class="metadata-value">{timestamp_str}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Stage 1 Uncertainty</div>
                <div class="metadata-value">±{profile.stage1_avg_uncertainty_pct:.0f}%</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">At-Risk Factors</div>
                <div class="metadata-value">{len(profile.at_risk_targets)}</div>
            </div>
        </div>
        
        <div class="content">
"""
    
    # Warning about pipeline
    html += """
            <!-- Pipeline Warning -->
            <div class="warning-box">
                <div class="warning-box-title">⚠️ Understanding This Two-Stage Report</div>
                <div class="warning-box-content">
                    <p><strong>This report uses a hybrid prediction pipeline:</strong></p>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li><strong>Stage 1 (Real Data):</strong> Predicts your next-day behaviors using patterns from actual students</li>
                        <li><strong>Stage 2 (Synthetic Data):</strong> Infers mental health from predicted behaviors using synthetic training data</li>
                    </ul>
                    <p style="margin-top: 10px;"><strong>Limitations:</strong> Stage 1 uncertainties compound in Stage 2. Mental health predictions are based on behavioral forecasts, not direct measurements. Use this as a screening tool, not a clinical diagnosis.</p>
                </div>
            </div>
"""
    
    # Stage 1: Behavioral Predictions
    html += f"""
            <!-- Stage 1: Behavioral Forecasting -->
            <div class="section">
                <div class="section-title">
                    🟦 Stage 1: Behavioral Forecasting
                    <span class="badge badge-real">REAL DATA</span>
                </div>
                <div class="section-subtitle">
                    LSTM trained on StudentLife dataset → predicts next-day behaviors from your 7-day history
                </div>
                
                <div class="card-grid">
"""
    
    # Behavioral predictions cards
    behavioral_display = {
        'sleep_hours': ('😴 Sleep', 'h', None),
        'exercise_minutes': ('🏃 Exercise', 'min', None),
        'screen_time_hours': ('📱 Screen Time', 'h', None),
        'social_interactions': ('👥 Social', '', None),
        'steps_count': ('🚶 Steps', '', None),
    }
    
    for target, (label, unit, _) in behavioral_display.items():
        if target in profile.stage1_predictions:
            pred = profile.stage1_predictions[target]
            unc = profile.stage1_uncertainties[target]
            unc_pct = (unc / pred * 100) if pred > 0 else 0
            
            html += f"""
                    <div class="card">
                        <div class="card-header">
                            <span>{label}</span>
                        </div>
                        <div class="card-value">{pred:.1f if target != 'steps_count' else pred:.0f}{unit}</div>
                        <div class="card-uncertainty">±{unc_pct:.0f}%</div>
                    </div>
"""
    
    html += """
                </div>
                
                <p style="font-size: 14px; color: #6c757d; margin-top: 10px;">
                    <strong>Note:</strong> These predictions show what your behavior is likely to be tomorrow, 
                    based on patterns learned from real university students. Uncertainty values reflect prediction confidence.
                </p>
            </div>
"""
    
    # Stage 2: Mental Health Predictions
    html += f"""
            <!-- Stage 2: Mental Health Inference -->
            <div class="section">
                <div class="section-title">
                    🟩 Stage 2: Mental Health Inference
                    <span class="badge badge-synthetic">SYNTHETIC DATA</span>
                </div>
                <div class="section-subtitle">
                    LSTM trained on 1.5M synthetic records → infers mental health from predicted behaviors
                </div>
                
                <div class="info-box">
                    <div class="info-box-title">ℹ️ Confidence Note</div>
                    <div class="info-box-content">{profile.confidence_note}</div>
                </div>
                
                <div class="card-grid">
"""
    
    # Mental health predictions
    mental_display = {
        'stress_level': ('Stress Level', 10, False),
        'mood_score': ('Mood', 10, True),
        'energy_level': ('Energy', 10, True),
        'focus_score': ('Focus', 10, True),
        'perceived_stress_scale': ('PSS', 40, False),
        'anxiety_score': ('Anxiety', 21, False),
        'depression_score': ('Depression', 27, False),
        'job_satisfaction': ('Job Satisfaction', 10, True),
    }
    
    for target, (label, max_val, is_inverted) in mental_display.items():
        if target in profile.stage2_clipped:
            value = profile.stage2_clipped[target]
            is_at_risk = target in profile.at_risk_targets
            risk_class = 'at-risk' if is_at_risk else 'healthy'
            
            html += f"""
                    <div class="prediction-card {risk_class}">
                        <div class="prediction-header">
                            <div class="prediction-name">{label}</div>
                            <div class="prediction-value {risk_class}">{value:.1f}/{max_val}</div>
                        </div>
                        <div style="font-size: 12px; color: #6c757d;">
                            {'⚠️ At-risk threshold exceeded' if is_at_risk else '✓ Within healthy range'}
                        </div>
                    </div>
"""
    
    html += """
                </div>
                
                <p style="font-size: 14px; color: #6c757d; margin-top: 10px;">
                    <strong>Important:</strong> These mental health scores are inferred from predicted behaviors, not measured directly. 
                    The model was trained on synthetic data patterns, which may not perfectly match real-world relationships. 
                    If you're concerned about any scores, please consult a mental health professional.
                </p>
            </div>
"""
    
    # Behavioral Recommendations
    if profile.behavioral_recommendations:
        html += """
            <!-- Behavioral Recommendations -->
            <div class="section">
                <div class="section-title">💡 Behavioral Recommendations</div>
                <div class="section-subtitle">
                    Evidence-based actions to improve your predicted behavioral patterns
                </div>
"""
        
        for rec in profile.behavioral_recommendations:
            priority = rec['priority'].lower()
            html += f"""
                <div class="recommendation-card {priority}">
                    <div class="recommendation-header">
                        <div class="recommendation-title">{rec['category']}</div>
                        <div class="priority-badge {priority}">{rec['priority']} PRIORITY</div>
                    </div>
                    
                    <div class="recommendation-metrics">
                        <div class="metric-item">
                            <div class="metric-label">Current</div>
                            <div class="metric-value">{rec['current']}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Target</div>
                            <div class="metric-value">{rec['target']}</div>
                        </div>
                    </div>
                    
                    <div class="recommendation-evidence">
                        📊 {rec['evidence']}
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <strong style="font-size: 14px; color: #2d3748;">Suggested Actions:</strong>
                        <ul class="action-list">
"""
            for action in rec['actions']:
                html += f"""
                            <li>{action}</li>
"""
            html += """
                        </ul>
                    </div>
                </div>
"""
        
        html += """
            </div>
"""
    
    # Footer
    html += f"""
            <!-- Footer -->
            <div class="section">
                <div style="border-top: 2px solid #e9ecef; padding-top: 20px; font-size: 12px; color: #6c757d;">
                    <p><strong>Disclaimer:</strong> This report is generated by machine learning models and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions you may have regarding mental health.</p>
                    <p style="margin-top: 10px;"><strong>Data Sources:</strong></p>
                    <ul style="margin-left: 20px; margin-top: 5px;">
                        <li>Stage 1 Model: Trained on StudentLife dataset (Dartmouth College, 2013)</li>
                        <li>Stage 2 Model: Trained on synthetic mental health dataset (1.5M records)</li>
                        <li>Recommendations: Based on peer-reviewed mental health research</li>
                    </ul>
                    <p style="margin-top: 10px;">Report generated: {timestamp_str}</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ HTML report saved: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate two-stage mental health profiles from Google Form CSV")
    parser.add_argument("--csv", type=Path, required=True, help="Path to Google Form CSV")
    parser.add_argument("--user-id", type=str, help="Specific user ID to process")
    parser.add_argument("--all-users", action="store_true", help="Process all users in CSV")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory for reports")
    parser.add_argument("--stage1-model", type=Path, default=STAGE1_MODEL, help="Stage 1 model path")
    parser.add_argument("--stage2-model", type=Path, default=STAGE2_MODEL, help="Stage 2 model path")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TWO-STAGE PROFILE GENERATOR")
    print("="*80)
    
    # Load models
    print("\n🔄 Loading models...")
    stage1_model, stage1_checkpoint = load_stage1_model(args.stage1_model)
    stage2_model, stage2_checkpoint = load_stage2_model(args.stage2_model)
    
    # Parse CSV
    print(f"\n📂 Parsing CSV: {args.csv}")
    user_data_dict = parse_google_form_csv(args.csv)
    
    # Determine which users to process
    if args.all_users:
        user_ids = list(user_data_dict.keys())
    elif args.user_id:
        if args.user_id not in user_data_dict:
            print(f"❌ User ID '{args.user_id}' not found in CSV")
            return
        user_ids = [args.user_id]
    else:
        print("❌ Must specify --user-id or --all-users")
        return
    
    print(f"\n🔄 Generating profiles for {len(user_ids)} user(s)...")
    
    # Generate profiles
    for user_id in user_ids:
        print(f"\n  Processing: {user_id}")
        
        user_data = user_data_dict[user_id]
        
        # Generate profile
        profile = generate_profile(
            user_id,
            user_data,
            stage1_model,
            stage1_checkpoint,
            stage2_model,
            stage2_checkpoint,
        )
        
        # Save JSON
        json_filename = f"profile_{user_id}_{profile.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        json_path = args.output_dir / json_filename
        
        profile_dict = {
            'user_id': profile.user_id,
            'timestamp': profile.timestamp.isoformat(),
            'stage1_predictions': profile.stage1_predictions,
            'stage1_uncertainties': profile.stage1_uncertainties,
            'stage1_avg_uncertainty_pct': profile.stage1_avg_uncertainty_pct,
            'stage2_predictions': profile.stage2_predictions,
            'stage2_clipped': profile.stage2_clipped,
            'at_risk_targets': profile.at_risk_targets,
            'healthy_targets': profile.healthy_targets,
            'recommendations': profile.behavioral_recommendations,
        }
        
        with open(json_path, 'w') as f:
            json.dump(profile_dict, f, indent=2)
        
        print(f"    ✓ JSON saved: {json_path}")
        
        # Generate HTML
        html_filename = f"report_{user_id}_{profile.timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        html_path = args.output_dir / html_filename
        
        generate_html_report(profile, html_path)
        
        print(f"    ✓ HTML report: {html_path}")
        print(f"    📊 Stage 1 Uncertainty: ±{profile.stage1_avg_uncertainty_pct:.0f}%")
        print(f"    ⚠️  At-risk factors: {len(profile.at_risk_targets)}")
        print(f"    ✅ Healthy factors: {len(profile.healthy_targets)}")
    
    print("\n" + "="*80)
    print("✓ PROFILE GENERATION COMPLETE")
    print("="*80)
    print(f"\nReports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
