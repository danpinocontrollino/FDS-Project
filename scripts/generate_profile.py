"""
================================================================================
COMPREHENSIVE MENTAL HEALTH PROFILING GENERATOR
================================================================================
Generates complete mental health profiles from Google Form CSV responses using
the multi-target prediction model.

FEATURES:
  ✓ Google Form CSV parsing with flexible column mapping
  ✓ Multi-target predictions (8 mental health outcomes)
  ✓ Job-specific advice and recommendations
  ✓ Contradiction detection in user responses
  ✓ Mental health history tracking and analysis
  ✓ Professional HTML report generation

USAGE:
    python scripts/generate_profile.py --csv form_responses.csv --user-id 12345
    python scripts/generate_profile.py --csv responses.csv --all-users
    python scripts/generate_profile.py --csv data.csv --output-dir reports/

REQUIREMENTS:
  - Trained model: models/saved/mental_health_lstm.pt (or transformer)
  - Google Form CSV with behavioral features (17 features × 7 days)

Author: University Project - Mental Health Prediction
================================================================================
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = Path("models/saved")
OUTPUT_DIR = Path("reports")
DEFAULT_MODEL = MODEL_DIR / "mental_health_lstm.pt"

# Expected feature columns (17 behavioral features)
FEATURE_COLS = [
    "sleep_hours",
    "sleep_quality",
    "work_hours",
    "meetings_count",
    "tasks_completed",
    "emails_received",
    "commute_minutes",
    "exercise_minutes",
    "steps_count",
    "caffeine_mg",
    "alcohol_units",
    "screen_time_hours",
    "social_interactions",
    "outdoor_time_minutes",
    "diet_quality",
    "work_pressure",
    "weather_mood_impact",
]

# Target names
DAILY_TARGETS = ["stress_level", "mood_score", "energy_level", "focus_score"]
WEEKLY_TARGETS = ["perceived_stress_scale", "anxiety_score", "depression_score", "job_satisfaction"]
ALL_TARGETS = DAILY_TARGETS + WEEKLY_TARGETS

# Job categories for personalized advice
JOB_CATEGORIES = [
    "knowledge_worker",      # Software, data science, research
    "healthcare",            # Doctors, nurses, therapists
    "education",             # Teachers, professors
    "creative",              # Designers, artists, content creators
    "management",            # Managers, executives, team leads
    "service",               # Customer service, hospitality
    "manual_labor",          # Construction, warehouse, delivery
    "sales_marketing",       # Sales, marketing, business development
    "other",                 # Default fallback
]

# Job category keyword mapping for automatic classification
JOB_KEYWORDS = {
    "knowledge_worker": ["software", "developer", "engineer", "data", "scientist", "analyst", "programmer", "tech", "research", "architect"],
    "healthcare": ["doctor", "nurse", "physician", "therapist", "medical", "healthcare", "clinical", "hospital", "patient"],
    "education": ["teacher", "professor", "educator", "instructor", "academic", "tutor", "faculty", "lecturer"],
    "creative": ["designer", "artist", "creative", "writer", "content", "marketing creative", "ux", "ui", "graphic"],
    "management": ["manager", "director", "executive", "lead", "supervisor", "ceo", "cto", "head of", "vp"],
    "service": ["customer service", "support", "hospitality", "retail", "waiter", "server", "receptionist"],
    "manual_labor": ["construction", "warehouse", "delivery", "driver", "mechanic", "technician", "maintenance", "labor"],
    "sales_marketing": ["sales", "marketing", "business development", "account", "representative", "consultant"],
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class UserProfile:
    """Complete user profile with predictions and metadata."""
    user_id: str
    timestamp: datetime
    job_category: Optional[str]
    
    # Input data (7 days of behavioral features)
    behavioral_data: pd.DataFrame
    
    # Predictions
    predictions: Dict[str, Dict[str, Any]]  # target → {value, at_risk, confidence}
    
    # Analysis flags
    contradictions: List[Dict[str, str]]  # List of detected contradictions
    risk_factors: List[str]                # High-risk areas
    positive_factors: List[str]            # Protective factors
    
    # Job-specific advice
    recommendations: List[Dict[str, Any]]  # Job-tailored recommendations
    
    # Mental health history tracking
    history_analysis: Optional[Dict[str, Any]]  # Trends from past assessments
    
    # Metadata
    data_quality_score: float              # 0-1 score for input data quality
    missing_features: List[str]            # Features with missing/imputed values


# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PredictionHead(nn.Module):
    """Single prediction head for one target."""
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
        reg_out = self.regression(h).squeeze(-1)
        cls_out = self.classification(h).squeeze(-1)
        return reg_out, cls_out


class MentalHealthPredictor(nn.Module):
    """Multi-target mental health prediction model."""
    
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
            return out[:, -1, :]
        else:  # transformer
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.encoder(x)
            return x.mean(dim=1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.encode(x)
        h = self.shared_repr(h)
        outputs = {}
        for target in self.targets:
            reg, cls = self.heads[target](h)
            outputs[target] = (reg, cls)
        return outputs


# ============================================================================
# CSV PARSING
# ============================================================================

def parse_google_form_csv(csv_path: Path, window: int = 7) -> Dict[str, pd.DataFrame]:
    """
    Parse Google Form CSV with flexible column mapping.
    
    Handles various naming conventions from Google Forms:
      - "Sleep hours (Day 1)" → sleep_hours
      - "Day 1: Sleep Quality" → sleep_quality
      - Fuzzy matching for feature names
    
    Returns:
        Dict mapping user_id → DataFrame with 7 days × 17 features
    """
    df = pd.read_csv(csv_path)
    
    print(f"📂 Loaded CSV: {len(df)} responses, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns[:5])}... (showing first 5)")
    
    # Detect user ID column (timestamp or explicit ID)
    user_id_col = None
    for col in df.columns:
        if "timestamp" in col.lower() or "user" in col.lower() or "id" in col.lower():
            user_id_col = col
            break
    
    if user_id_col is None:
        print("⚠️  No user ID column found, using row index")
        df["user_id"] = df.index.astype(str)
        user_id_col = "user_id"
    
    # Job category detection
    job_col = None
    for col in df.columns:
        if "job" in col.lower() or "occupation" in col.lower() or "role" in col.lower():
            job_col = col
            break
    
    # Build mapping from form columns to feature names
    # Expected pattern: "Day X: Feature Name" or "Feature Name (Day X)"
    column_mapping = {}
    
    for day in range(1, window + 1):
        for feature in FEATURE_COLS:
            # Try various naming patterns
            patterns = [
                f"Day {day}: {feature.replace('_', ' ').title()}",
                f"Day {day} - {feature.replace('_', ' ').title()}",
                f"{feature.replace('_', ' ').title()} (Day {day})",
                f"D{day}: {feature.replace('_', ' ').title()}",
            ]
            
            for pattern in patterns:
                # Case-insensitive fuzzy match
                for col in df.columns:
                    if pattern.lower() in col.lower() or col.lower() in pattern.lower():
                        column_mapping[col] = (feature, day)
                        break
    
    print(f"✓ Mapped {len(column_mapping)} columns to features")
    
    # Parse each user's data
    user_data = {}
    
    for idx, row in df.iterrows():
        user_id = str(row[user_id_col])
        
        # Extract job category if available
        job = None
        if job_col and pd.notna(row[job_col]):
            job = str(row[job_col]).lower()
        
        # Build 7-day sequence
        sequence = pd.DataFrame(index=range(window), columns=FEATURE_COLS, dtype=float)
        
        for col, (feature, day) in column_mapping.items():
            value = row[col]
            if pd.notna(value):
                try:
                    sequence.loc[day - 1, feature] = float(value)
                except (ValueError, TypeError):
                    # Handle categorical (e.g., work_pressure: low/medium/high)
                    if feature == "work_pressure":
                        pressure_map = {"low": 0, "medium": 1, "high": 2}
                        sequence.loc[day - 1, feature] = pressure_map.get(str(value).lower(), 1)
        
        # Fill missing values with column means (basic imputation)
        for feature in FEATURE_COLS:
            if sequence[feature].isna().all():
                # Use reasonable defaults
                defaults = {
                    "sleep_hours": 7.0,
                    "sleep_quality": 5.0,
                    "work_hours": 8.0,
                    "work_pressure": 1.0,
                    "diet_quality": 5.0,
                }
                sequence[feature] = defaults.get(feature, sequence[feature].mean())
            else:
                sequence[feature].fillna(sequence[feature].mean(), inplace=True)
        
        user_data[user_id] = {
            "data": sequence,
            "job": job,
            "timestamp": datetime.now(),
        }
    
    print(f"✓ Parsed {len(user_data)} user profiles")
    return user_data


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def load_model(model_path: Path = DEFAULT_MODEL) -> Tuple[MentalHealthPredictor, Dict[str, Any]]:
    """Load trained model and metadata."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Build model
    model = MentalHealthPredictor(
        input_dim=len(FEATURE_COLS),
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=2,
        encoder_type=checkpoint["model_type"],
        targets=checkpoint["targets"],
    )
    
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded {checkpoint['model_type'].upper()} model from {model_path}")
    return model, checkpoint


def predict_user(
    model: MentalHealthPredictor,
    user_data: pd.DataFrame,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    risk_thresholds: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """
    Generate predictions for a single user.
    
    Returns:
        Dict mapping target → {value, at_risk, confidence, percentile}
    """
    device = next(model.parameters()).device
    
    # Normalize (fill NaN with scaler mean)
    X = user_data[FEATURE_COLS].values.astype(np.float32)
    
    # Fill any NaN values with the scaler mean (trained distribution center)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        for i in range(X.shape[1]):
            if np.isnan(X[:, i]).any():
                X[np.isnan(X[:, i]), i] = scaler_mean[i]
    
    X_scaled = (X - scaler_mean) / scaler_std
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(X_tensor)
    
    # Process predictions
    predictions = {}
    inverted_targets = {"mood_score", "energy_level", "focus_score", "job_satisfaction"}
    
    for target in ALL_TARGETS:
        reg_pred, cls_logit = outputs[target]
        
        value = reg_pred.item()
        at_risk_prob = torch.sigmoid(cls_logit).item()
        
        # Determine risk based on threshold
        threshold = risk_thresholds[target]
        if target in inverted_targets:
            at_risk = value <= threshold
        else:
            at_risk = value >= threshold
        
        predictions[target] = {
            "value": round(value, 2),
            "at_risk": at_risk,
            "at_risk_probability": round(at_risk_prob, 3),
            "confidence": round(abs(at_risk_prob - 0.5) * 2, 3),  # 0=uncertain, 1=confident
            "threshold": threshold,
        }
    
    return predictions


# ============================================================================
# PROFILE ANALYSIS
# ============================================================================

def detect_contradictions(user_data: pd.DataFrame, predictions: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Advanced contradiction detection between behaviors and predicted outcomes.
    
    Detects 15+ contradiction patterns across:
      - Sleep & stimulants
      - Work-life balance
      - Physical activity & energy
      - Social factors
      - Mental health consistency
      - Data quality issues
    """
    contradictions = []
    
    # Calculate behavioral statistics
    avg = user_data.mean()
    std = user_data.std()
    
    # ========================================================================
    # SLEEP & STIMULANT CONTRADICTIONS
    # ========================================================================
    
    # High caffeine + good sleep quality
    if avg["caffeine_mg"] > 400 and avg["sleep_quality"] > 7:
        severity = "high" if avg["caffeine_mg"] > 600 else "medium"
        contradictions.append({
            "type": "caffeine_sleep_quality",
            "description": f"Very high caffeine intake ({avg['caffeine_mg']:.0f}mg/day) but reports good sleep quality ({avg['sleep_quality']:.1f}/10). This is physiologically unusual.",
            "severity": severity,
            "explanation": "Caffeine has 5-6hr half-life. >400mg typically disrupts sleep architecture even if subjectively 'good'. May indicate tolerance, measurement error, or atypical metabolism.",
            "recommendation": "Consider objective sleep tracking (fitness tracker) to validate subjective quality. Reduce caffeine after 2pm.",
        })
    
    # High caffeine + sufficient sleep hours
    if avg["caffeine_mg"] > 500 and avg["sleep_hours"] > 8:
        contradictions.append({
            "type": "caffeine_sleep_hours",
            "description": f"Excessive caffeine ({avg['caffeine_mg']:.0f}mg/day) yet sleeping {avg['sleep_hours']:.1f}h/night. May mask underlying fatigue.",
            "severity": "medium",
            "explanation": "High caffeine with long sleep suggests compensating for poor sleep quality or chronic fatigue with both stimulants and extended sleep.",
            "recommendation": "Evaluate for sleep disorders (apnea, restless leg). Check iron/B12 levels. Taper caffeine gradually.",
        })
    
    # Alcohol + good sleep
    if avg["alcohol_units"] > 2 and avg["sleep_quality"] > 7:
        contradictions.append({
            "type": "alcohol_sleep",
            "description": f"Regular alcohol consumption ({avg['alcohol_units']:.1f} units/day) but reports good sleep quality. Alcohol fragments REM sleep.",
            "severity": "medium",
            "explanation": "Alcohol induces sedation (feels like sleep) but disrupts sleep cycles, reducing restorative REM sleep. Subjective quality may not reflect objective quality.",
            "recommendation": "Track sleep with device. Note hangover/grogginess. Consider alcohol-free experiment for 2 weeks to compare.",
        })
    
    # Sleep deprivation + high energy
    if avg["sleep_hours"] < 6 and predictions["energy_level"]["value"] > 7:
        contradictions.append({
            "type": "sleep_debt_energy",
            "description": f"Chronic sleep deprivation ({avg['sleep_hours']:.1f}h/night) but predicted high energy ({predictions['energy_level']['value']:.1f}/10). Red flag for burnout masking.",
            "severity": "high",
            "explanation": "High energy despite inadequate sleep suggests running on stress hormones (cortisol, adrenaline). This 'tired but wired' state precedes crashes.",
            "recommendation": "URGENT: This pattern predicts burnout crash within weeks. Prioritize sleep immediately. May need medical evaluation for thyroid/adrenal issues.",
        })
    
    # ========================================================================
    # WORK-LIFE BALANCE CONTRADICTIONS
    # ========================================================================
    
    # Overwork + low stress
    if avg["work_hours"] > 10 and predictions["stress_level"]["value"] < 4:
        contradictions.append({
            "type": "overwork_low_stress",
            "description": f"Long work hours ({avg['work_hours']:.1f}h/day) but predicted low stress ({predictions['stress_level']['value']:.1f}/9). Possible stress denial or habituation.",
            "severity": "medium",
            "explanation": "Working 50+ hours/week with 'low stress' suggests: (1) disconnection from body signals, (2) workaholic identity, or (3) fear of admitting overwhelm.",
            "recommendation": "Check for physical stress symptoms (tension, digestive issues, sleep problems). Practice body-scan meditation. Track cortisol if possible.",
        })
    
    # Overwork + high job satisfaction
    if avg["work_hours"] > 11 and predictions["job_satisfaction"]["value"] > 7:
        contradictions.append({
            "type": "overwork_satisfaction",
            "description": f"Extreme work hours ({avg['work_hours']:.1f}h/day) with high job satisfaction. Sustainable only short-term; burnout risk.",
            "severity": "medium",
            "explanation": "Passion-driven overwork feels good temporarily but depletes resources. 'Love what you do' doesn't make 55+ hr weeks sustainable.",
            "recommendation": "Set hard boundaries now while still satisfied. Passion projects need pacing. Measure 3-month sustainability, not weekly.",
        })
    
    # Low work hours + at-risk job satisfaction
    if avg["work_hours"] < 5 and predictions["job_satisfaction"]["at_risk"]:
        contradictions.append({
            "type": "underwork_dissatisfaction",
            "description": f"Minimal work hours ({avg['work_hours']:.1f}h/day) but low job satisfaction. May indicate underemployment or disengagement.",
            "severity": "low",
            "explanation": "Part-time hours with dissatisfaction suggests role mismatch, underutilization, or financial stress from insufficient work.",
            "recommendation": "Clarify if this is desired part-time or forced underemployment. If latter, explore full-time opportunities or additional income streams.",
        })
    
    # High meetings + high productivity claim
    if avg["meetings_count"] > 6 and avg["tasks_completed"] > 15:
        contradictions.append({
            "type": "meeting_productivity",
            "description": f"Excessive meetings ({avg['meetings_count']:.0f}/day) yet high task completion ({avg['tasks_completed']:.0f}/day). Questionable sustainability or quality.",
            "severity": "low",
            "explanation": "6+ meetings/day leaves ~4hrs for deep work. High task completion suggests: (1) tasks are trivial, (2) working after-hours uncounted, or (3) multitasking in meetings.",
            "recommendation": "Audit meeting necessity. Block 'no meeting' days. Question if task quantity reflects quality/impact.",
        })
    
    # ========================================================================
    # PHYSICAL ACTIVITY & ENERGY CONTRADICTIONS
    # ========================================================================
    
    # Sedentary + high energy
    if avg["exercise_minutes"] < 15 and avg["steps_count"] < 3000 and predictions["energy_level"]["value"] > 8:
        contradictions.append({
            "type": "sedentary_high_energy",
            "description": f"Very sedentary lifestyle (exercise: {avg['exercise_minutes']:.0f}min/day, steps: {avg['steps_count']:.0f}) but high predicted energy. Unusual pattern.",
            "severity": "medium",
            "explanation": "High energy without movement suggests: (1) young age compensating, (2) nervous energy (anxiety), or (3) measurement error in activity tracking.",
            "recommendation": "Add light activity (15min walks) to test if energy is 'real' or anxious restlessness. May reveal underlying anxiety.",
        })
    
    # High exercise + low energy
    if avg["exercise_minutes"] > 90 and predictions["energy_level"]["value"] < 4:
        contradictions.append({
            "type": "overtraining_fatigue",
            "description": f"High exercise volume ({avg['exercise_minutes']:.0f}min/day) but low predicted energy. Possible overtraining syndrome.",
            "severity": "high",
            "explanation": "Excessive training without adequate recovery depletes energy. Overtraining syndrome includes fatigue, mood issues, weakened immunity.",
            "recommendation": "Take 3-5 day rest period. Check for overtraining signs: elevated resting HR, irritability, decreased performance. May need deload week.",
        })
    
    # High steps + reported low activity
    if avg["steps_count"] > 12000 and avg["exercise_minutes"] < 20:
        contradictions.append({
            "type": "steps_exercise_mismatch",
            "description": f"High daily steps ({avg['steps_count']:.0f}) but minimal reported exercise. Definitional confusion or occupational activity.",
            "severity": "low",
            "explanation": "Steps indicate active job (retail, nursing, delivery) but not counted as 'exercise'. This is still beneficial physical activity.",
            "recommendation": "Occupational activity counts! Consider adding strength training since you're already active cardio-wise.",
        })
    
    # ========================================================================
    # SOCIAL & MENTAL HEALTH CONTRADICTIONS
    # ========================================================================
    
    # Social isolation + good mood
    if avg["social_interactions"] < 1 and predictions["mood_score"]["value"] > 8:
        contradictions.append({
            "type": "isolation_good_mood",
            "description": f"Very low social contact ({avg['social_interactions']:.1f}/day) but high predicted mood. Introvert pattern or temporary?",
            "severity": "low",
            "explanation": "Introverts thrive with low social contact. However, sustained isolation risks depression even for introverts. Humans are social species.",
            "recommendation": "If genuinely content, this is fine. Monitor for creeping loneliness. Ensure quality connections exist, even if infrequent.",
        })
    
    # High social + poor mood
    if avg["social_interactions"] > 8 and predictions["mood_score"]["value"] < 5:
        contradictions.append({
            "type": "social_low_mood",
            "description": f"High social activity ({avg['social_interactions']:.0f}/day) but low predicted mood. Quality vs quantity issue.",
            "severity": "medium",
            "explanation": "Many shallow interactions don't provide emotional support. May be draining customer service, conflict-heavy relationships, or forced socializing.",
            "recommendation": "Audit relationship quality. Are these meaningful connections or obligations? Prioritize depth over breadth. Say no to draining social events.",
        })
    
    # No outdoor time + good mood in winter
    if avg["outdoor_time_minutes"] < 15 and predictions["mood_score"]["value"] > 7:
        # Check if this might be seasonal (harder to detect without date, but flag it)
        contradictions.append({
            "type": "indoor_mood",
            "description": f"Minimal outdoor time ({avg['outdoor_time_minutes']:.0f}min/day) but good mood. Watch for seasonal affective disorder (SAD).",
            "severity": "low",
            "explanation": "Currently good mood, but lack of sunlight/nature exposure increases SAD risk. Vitamin D deficiency affects mood regulation.",
            "recommendation": "Increase outdoor time to 30min/day. Consider vitamin D supplementation (especially winter). Use SAD lamp if needed.",
        })
    
    # ========================================================================
    # STRESS & COPING CONTRADICTIONS
    # ========================================================================
    
    # High stress + no stress management behaviors
    if (predictions["stress_level"]["at_risk"] and 
        avg["exercise_minutes"] < 30 and 
        avg["outdoor_time_minutes"] < 20 and 
        avg["social_interactions"] < 2):
        contradictions.append({
            "type": "stress_no_coping",
            "description": f"High predicted stress but minimal stress-relief activities (exercise, nature, social support). Vulnerability to burnout.",
            "severity": "high",
            "explanation": "Stress without coping mechanisms accumulates. Lacking exercise, nature, and social support means no release valve for stress.",
            "recommendation": "URGENT: Add at least ONE stress-relief practice daily. Start with easiest: 20min walk outside. Build from there.",
        })
    
    # At-risk mental health + inconsistent self-care
    if (predictions["depression_score"]["at_risk"] or predictions["anxiety_score"]["at_risk"]) and \
       (avg["sleep_hours"] < 7 or avg["exercise_minutes"] < 20 or avg["diet_quality"] < 5):
        contradictions.append({
            "type": "mental_health_poor_foundation",
            "description": f"At-risk mental health but poor foundational self-care (sleep/exercise/diet). Interventions less effective without basics.",
            "severity": "high",
            "explanation": "Therapy and medication work better with sleep/exercise/nutrition foundation. Can't out-therapy a terrible lifestyle.",
            "recommendation": "While seeking professional help, prioritize: (1) 7-8h sleep, (2) 30min daily movement, (3) regular meals. These amplify treatment.",
        })
    
    # ========================================================================
    # DATA QUALITY CONTRADICTIONS
    # ========================================================================
    
    # Suspiciously consistent data (low variance)
    low_variance_features = []
    for col in ["sleep_hours", "work_hours", "mood_score", "energy_level"]:
        if col in user_data.columns and std[col] < 0.3:
            low_variance_features.append(col)
    
    if len(low_variance_features) >= 3:
        contradictions.append({
            "type": "suspiciously_consistent_data",
            "description": f"Multiple features show unrealistic consistency ({', '.join(low_variance_features)}). Possible inaccurate self-reporting or rounding.",
            "severity": "low",
            "explanation": "Real life has variance. Perfectly consistent daily values suggest: (1) retrospective estimates, (2) social desirability bias, or (3) not actually tracking.",
            "recommendation": "Use momentary tracking (log events as they happen) rather than end-of-day estimates. Be honest - data is for you, not judgment.",
        })
    
    # Extreme values cluster
    extreme_count = 0
    if avg["work_hours"] > 12: extreme_count += 1
    if avg["screen_time_hours"] > 14: extreme_count += 1
    if avg["caffeine_mg"] > 600: extreme_count += 1
    if avg["alcohol_units"] > 4: extreme_count += 1
    
    if extreme_count >= 3:
        contradictions.append({
            "type": "multiple_extremes",
            "description": f"Multiple extreme behaviors reported simultaneously. Verify data accuracy or recognize unsustainable lifestyle.",
            "severity": "high",
            "explanation": "Working 60+hrs, 14+hrs screens, 600+mg caffeine daily is extreme even separately. Together suggests crisis mode or data error.",
            "recommendation": "If accurate: This is emergency-level burnout trajectory. Immediate intervention needed. If inaccurate: Review tracking methods for errors.",
        })
    
    # Perfect mental health despite poor behaviors
    healthy_predictions = sum([1 for t in ALL_TARGETS if not predictions[t]["at_risk"]])
    poor_behaviors = sum([
        1 if avg["sleep_hours"] < 6 else 0,
        1 if avg["exercise_minutes"] < 15 else 0,
        1 if avg["work_hours"] > 10 else 0,
        1 if avg["caffeine_mg"] > 400 else 0,
        1 if avg["alcohol_units"] > 2 else 0,
    ])
    
    if healthy_predictions >= 7 and poor_behaviors >= 4:
        contradictions.append({
            "type": "good_outcomes_poor_inputs",
            "description": f"Excellent predicted mental health despite multiple poor health behaviors. Unusually resilient or early-stage decline not yet manifesting.",
            "severity": "medium",
            "explanation": "Young age, genetics, or acute crisis response can mask poor lifestyle impact temporarily. Consequences often appear suddenly after accumulation period.",
            "recommendation": "Don't assume invincibility. Poor inputs predict future decline even if current outcomes are good. Prevention easier than treatment.",
        })
    
    return contradictions


def identify_risk_factors(predictions: Dict[str, Dict[str, Any]], user_data: pd.DataFrame) -> List[str]:
    """Identify key risk factors based on predictions and behaviors."""
    risks = []
    avg = user_data.mean()
    
    # Mental health risks
    for target in ALL_TARGETS:
        if predictions[target]["at_risk"]:
            risks.append(f"At-risk {target.replace('_', ' ')}: {predictions[target]['value']}")
    
    # Behavioral risks
    if avg["sleep_hours"] < 6:
        risks.append(f"Insufficient sleep: {avg['sleep_hours']:.1f}h/day (< 6h)")
    
    if avg["work_hours"] > 10:
        risks.append(f"Overwork: {avg['work_hours']:.1f}h/day (> 10h)")
    
    if avg["exercise_minutes"] < 20:
        risks.append(f"Sedentary lifestyle: {avg['exercise_minutes']:.0f}min/day exercise")
    
    if avg["caffeine_mg"] > 400:
        risks.append(f"High caffeine: {avg['caffeine_mg']:.0f}mg/day (> 400mg)")
    
    if avg["alcohol_units"] > 2:
        risks.append(f"Elevated alcohol: {avg['alcohol_units']:.1f} units/day")
    
    if avg["screen_time_hours"] > 12:
        risks.append(f"Excessive screen time: {avg['screen_time_hours']:.1f}h/day")
    
    return risks


def identify_positive_factors(predictions: Dict[str, Dict[str, Any]], user_data: pd.DataFrame) -> List[str]:
    """Identify protective/positive factors."""
    positives = []
    avg = user_data.mean()
    
    # Healthy predictions
    if not predictions["depression_score"]["at_risk"]:
        positives.append(f"Healthy depression score: {predictions['depression_score']['value']:.1f}")
    
    if not predictions["anxiety_score"]["at_risk"]:
        positives.append(f"Low anxiety: {predictions['anxiety_score']['value']:.1f}")
    
    if predictions["job_satisfaction"]["value"] > 7:
        positives.append(f"High job satisfaction: {predictions['job_satisfaction']['value']:.1f}/9")
    
    # Healthy behaviors
    if avg["sleep_hours"] >= 7 and avg["sleep_hours"] <= 9:
        positives.append(f"Optimal sleep: {avg['sleep_hours']:.1f}h/day")
    
    if avg["exercise_minutes"] >= 30:
        positives.append(f"Regular exercise: {avg['exercise_minutes']:.0f}min/day")
    
    if avg["social_interactions"] >= 3:
        positives.append(f"Good social connections: {avg['social_interactions']:.1f} interactions/day")
    
    if avg["outdoor_time_minutes"] >= 30:
        positives.append(f"Adequate outdoor time: {avg['outdoor_time_minutes']:.0f}min/day")
    
    if avg["diet_quality"] >= 7:
        positives.append(f"Healthy diet: {avg['diet_quality']:.1f}/10")
    
    return positives


def calculate_data_quality(user_data: pd.DataFrame) -> Tuple[float, List[str]]:
    """Calculate data quality score and identify missing/imputed features."""
    missing = []
    
    for col in FEATURE_COLS:
        if user_data[col].isna().any():
            missing.append(col)
    
    # Quality score based on completeness and variance
    completeness = 1 - (len(missing) / len(FEATURE_COLS))
    
    # Check for suspiciously constant values (possible data quality issue)
    constant_features = 0
    for col in FEATURE_COLS:
        if user_data[col].std() < 0.01:
            constant_features += 1
    
    variance_penalty = constant_features / len(FEATURE_COLS) * 0.3
    quality_score = max(0, completeness - variance_penalty)
    
    return quality_score, missing


# ============================================================================
# JOB-SPECIFIC ADVICE SYSTEM
# ============================================================================

def classify_job_category(job_string: Optional[str]) -> str:
    """Automatically classify job into category based on keywords."""
    if not job_string:
        return "other"
    
    job_lower = job_string.lower()
    
    # Check each category's keywords
    for category, keywords in JOB_KEYWORDS.items():
        for keyword in keywords:
            if keyword in job_lower:
                return category
    
    return "other"


def generate_job_specific_advice(
    job_category: str,
    predictions: Dict[str, Dict[str, Any]],
    user_data: pd.DataFrame,
    risk_factors: List[str],
) -> List[Dict[str, Any]]:
    """
    Generate job-specific recommendations based on predictions and behaviors.
    
    Returns list of recommendations with:
      - category: stress/sleep/work_life_balance/etc
      - priority: high/medium/low
      - advice: Specific actionable recommendation
      - rationale: Why this matters for their job
    """
    recommendations = []
    avg = user_data.mean()
    
    # ========================================================================
    # KNOWLEDGE WORKERS (Software, Data Science, Research)
    # ========================================================================
    if job_category == "knowledge_worker":
        
        # High stress + long hours
        if predictions["stress_level"]["at_risk"] and avg["work_hours"] > 9:
            recommendations.append({
                "category": "work_life_balance",
                "priority": "high",
                "advice": "Implement strict work boundaries: use Pomodoro technique (25min focus + 5min break), set hard stop time at 6pm, disable work notifications after hours",
                "rationale": "Knowledge work requires sustained cognitive focus. Without breaks, mental fatigue compounds stress and reduces code quality/decision-making.",
                "job_specific": "Schedule 'deep work' blocks in morning when cognitive capacity is highest. Use afternoon for meetings and lighter tasks.",
            })
        
        # Poor focus + high screen time
        if predictions["focus_score"]["at_risk"] and avg["screen_time_hours"] > 10:
            recommendations.append({
                "category": "cognitive_health",
                "priority": "high",
                "advice": "Reduce digital fatigue: follow 20-20-20 rule (every 20min, look at something 20ft away for 20sec), use blue light filters after 6pm, take 10min screen-free breaks every 2h",
                "rationale": "Extended screen exposure degrades attention span and working memory - critical for debugging, system design, and complex problem-solving.",
                "job_specific": "Use tools like RescueTime to track focus patterns. Block distracting sites during deep work. Consider dual monitor setup to reduce context switching.",
            })
        
        # Low energy + sedentary
        if predictions["energy_level"]["at_risk"] and avg["exercise_minutes"] < 20:
            recommendations.append({
                "category": "physical_health",
                "priority": "medium",
                "advice": "Combat sitting fatigue: stand-up desk sessions (alternate every hour), walk during calls, 5-min stretching routine every 2h, aim for 7k+ steps/day",
                "rationale": "Prolonged sitting reduces blood flow to brain, lowering energy and cognitive performance. Movement boosts alertness and creative problem-solving.",
                "job_specific": "Walk while debugging rubber-duck style, do standing desk during code reviews, take stairs for short building meetings.",
            })
        
        # High anxiety + poor sleep
        if predictions["anxiety_score"]["at_risk"] and avg["sleep_hours"] < 7:
            recommendations.append({
                "category": "sleep",
                "priority": "high",
                "advice": "Improve sleep hygiene: stop coding 1h before bed, dim lights after 9pm, keep bedroom cool (65-68°F), maintain consistent sleep schedule (even weekends)",
                "rationale": "Sleep deprivation amplifies anxiety and impairs logical reasoning. Tech workers often have delayed circadian rhythms from late-night coding.",
                "job_specific": "No debugging in bed. Use f.lux/Night Shift on all devices. If mind races about code, keep 'brain dump' notepad by bed.",
            })
        
        # High caffeine
        if avg["caffeine_mg"] > 400:
            recommendations.append({
                "category": "substance",
                "priority": "medium",
                "advice": "Reduce caffeine dependence: cut off coffee after 2pm, switch afternoon coffee to green tea (lower caffeine), drink water between coffees, aim for <300mg/day",
                "rationale": "Excessive caffeine creates dependence cycle, disrupts sleep, and increases jitteriness during high-pressure debugging or presentations.",
                "job_specific": "Don't use caffeine to mask poor sleep. Front-load caffeine in morning. Consider L-theanine supplement with coffee for smoother focus.",
            })
        
        # Low job satisfaction
        if predictions["job_satisfaction"]["at_risk"]:
            recommendations.append({
                "category": "career",
                "priority": "high",
                "advice": "Address job dissatisfaction: schedule 1-on-1 with manager to discuss growth opportunities, identify skill gaps to learn, explore internal mobility, set clear 3-month goals",
                "rationale": "Low satisfaction in knowledge work often stems from lack of autonomy, mastery, or purpose. Prolonged dissatisfaction leads to burnout.",
                "job_specific": "Request challenging projects aligned with interests. Join tech communities for external validation. Consider if current tech stack matches career goals.",
            })
    
    # ========================================================================
    # HEALTHCARE WORKERS
    # ========================================================================
    elif job_category == "healthcare":
        
        # High stress + emotional burden
        if predictions["stress_level"]["at_risk"] or predictions["anxiety_score"]["at_risk"]:
            recommendations.append({
                "category": "stress",
                "priority": "high",
                "advice": "Manage compassion fatigue: establish post-shift decompression routine (15min), seek peer support groups, practice emotional boundary-setting, consider professional counseling",
                "rationale": "Healthcare workers face vicarious trauma and life-or-death decisions. Without emotional processing, stress accumulates into burnout.",
                "job_specific": "Use 'emotional hand-washing' ritual after difficult patients (deep breaths, visualize washing away the emotional weight). Debrief with colleagues.",
            })
        
        # Irregular sleep (common in shift work)
        if avg["sleep_hours"] < 7 or avg["sleep_quality"] < 5:
            recommendations.append({
                "category": "sleep",
                "priority": "high",
                "advice": "Optimize shift work sleep: blackout curtains + eye mask for day sleep, maintain same bedtime on days off, strategic naps (20min before night shifts), melatonin for circadian reset",
                "rationale": "Shift work disrupts circadian rhythm, impairing clinical judgment and increasing medical errors. Sleep debt accumulates dangerously.",
                "job_specific": "Request consistent shift patterns where possible. Use bright light therapy after night shifts. Prioritize sleep over social obligations on work days.",
            })
        
        # Long work hours + no breaks
        if avg["work_hours"] > 10:
            recommendations.append({
                "category": "work_life_balance",
                "priority": "high",
                "advice": "Enforce break discipline: use full lunch break away from work area, decline shift extensions when safe, strictly protect days off, rotate call duties",
                "rationale": "Extended shifts without breaks increase medical errors and decrease empathy. Martyrdom culture harms both caregivers and patients.",
                "job_specific": "Coordinate with team for coverage during breaks. If short-staffed, escalate to management - patient safety requires rested providers.",
            })
        
        # Low social support
        if avg["social_interactions"] < 2:
            recommendations.append({
                "category": "social",
                "priority": "medium",
                "advice": "Build professional support network: join nursing/physician support group, schedule regular coffee with work friends, attend hospital social events, debrief with partner",
                "rationale": "Healthcare work is emotionally isolating. Peer support reduces compassion fatigue and provides perspective on challenging cases.",
                "job_specific": "Find mentor who understands healthcare stress. Use Signal/WhatsApp for quick check-ins with colleagues during tough shifts.",
            })
        
        # Depression risk
        if predictions["depression_score"]["at_risk"]:
            recommendations.append({
                "category": "mental_health",
                "priority": "high",
                "advice": "Seek professional support NOW: contact Employee Assistance Program (EAP), schedule appointment with therapist familiar with healthcare workers, consider psychiatry referral if severe",
                "rationale": "Healthcare workers have elevated suicide risk. Depression impairs clinical judgment and patient care. Early intervention is critical.",
                "job_specific": "Many hospitals have confidential mental health resources specifically for staff. Depression is not weakness - it's a medical condition requiring treatment.",
            })
    
    # ========================================================================
    # EDUCATORS (Teachers, Professors)
    # ========================================================================
    elif job_category == "education":
        
        # High stress + overwhelming workload
        if predictions["stress_level"]["at_risk"] and avg["work_hours"] > 9:
            recommendations.append({
                "category": "workload",
                "priority": "high",
                "advice": "Set sustainable boundaries: batch-grade assignments (dedicated 2h blocks), reuse lesson plans with minor updates, say no to extra committees, leave work at school",
                "rationale": "Teacher burnout often stems from unlimited work expanding to fill all time. Perfectionism and guilt about students drive overwork.",
                "job_specific": "Students benefit more from a rested teacher than a perfect lesson. Collaborate with colleagues to share resources. Quality over quantity in feedback.",
            })
        
        # Emotional exhaustion
        if predictions["anxiety_score"]["at_risk"] or predictions["depression_score"]["at_risk"]:
            recommendations.append({
                "category": "emotional_health",
                "priority": "high",
                "advice": "Address emotional labor: establish routine to 'leave school at school' (change clothes, listen to music), join teacher support group, practice self-compassion (you can't save every student)",
                "rationale": "Educators carry emotional weight of students' struggles. Without boundaries, vicarious trauma and helplessness accumulate.",
                "job_specific": "Remind yourself: you're a teacher, not a therapist or parent. Refer struggling students to counselors. Celebrate small wins.",
            })
        
        # Poor work-life balance
        if avg["work_hours"] > 10 or avg["outdoor_time_minutes"] < 20:
            recommendations.append({
                "category": "work_life_balance",
                "priority": "medium",
                "advice": "Reclaim personal time: set hard stop for grading (e.g., 7pm), schedule non-negotiable weekend activities, use planning periods for planning (not meetings), take mental health days",
                "rationale": "Teaching is never 'done' - there's always more to do. Without boundaries, personal life erodes, leading to resentment and burnout.",
                "job_specific": "Block calendar for lesson prep so admin can't schedule meetings. Use summer break for genuine rest, not just planning next year.",
            })
        
        # Low job satisfaction
        if predictions["job_satisfaction"]["at_risk"]:
            recommendations.append({
                "category": "career",
                "priority": "high",
                "advice": "Reconnect with purpose: reflect on why you chose teaching, seek out positive student interactions, join professional learning community, consider grade/subject change if needed",
                "rationale": "Teaching satisfaction erodes from lack of autonomy, unsupportive admin, or feeling ineffective. Rekindling purpose prevents dropout.",
                "job_specific": "Keep 'smile file' of positive student notes/emails. If systemic issues (admin, resources), consider school change. Teaching elsewhere ≠ failure.",
            })
    
    # ========================================================================
    # CREATIVE PROFESSIONALS (Designers, Artists, Content Creators)
    # ========================================================================
    elif job_category == "creative":
        
        # Creative blocks + high stress
        if predictions["focus_score"]["at_risk"] or predictions["stress_level"]["at_risk"]:
            recommendations.append({
                "category": "creative_process",
                "priority": "high",
                "advice": "Combat creative burnout: schedule 'whitespace' time (no deliverables, just exploration), alternate between creative and administrative tasks, embrace 'ugly first drafts', take inspiration walks",
                "rationale": "Creative work requires cognitive flexibility. Stress and pressure trigger rigid thinking, blocking innovation and original ideas.",
                "job_specific": "Separate ideation from execution. Morning for creation, afternoon for revisions. Keep 'inspiration swipe file' for dry periods.",
            })
        
        # Irregular work patterns
        if avg["sleep_quality"] < 6 or avg["work_hours"] > 10:
            recommendations.append({
                "category": "routine",
                "priority": "medium",
                "advice": "Create sustainable routine: set regular work hours (creativity needs structure), protect morning creative time, schedule client meetings in afternoon, batch similar tasks",
                "rationale": "Freelance/agency creatives often have chaotic schedules. While flexibility is nice, routine enhances creative output and prevents burnout.",
                "job_specific": "Track when you're most creative (morning/evening). Design ideal week template. Learn to estimate project time realistically (add 50%).",
            })
        
        # Client/stakeholder stress
        if predictions["anxiety_score"]["at_risk"]:
            recommendations.append({
                "category": "client_management",
                "priority": "medium",
                "advice": "Set client boundaries: establish revision limits (e.g., 2 rounds), require consolidated feedback, educate clients on creative process, charge for scope creep",
                "rationale": "Anxiety in creative work often stems from unclear expectations, endless revisions, or clients who don't respect expertise.",
                "job_specific": "Use contracts with clear deliverables. Present 2-3 options max (not infinite revisions). Walk away from toxic clients - not worth mental health.",
            })
        
        # Isolation (common in freelance)
        if avg["social_interactions"] < 2:
            recommendations.append({
                "category": "social",
                "priority": "medium",
                "advice": "Build creative community: join coworking space, attend design/art meetups, find accountability partner, collaborate on side projects, engage in online communities",
                "rationale": "Creative work is often solitary. Isolation breeds self-doubt and depression. Community provides feedback, support, and collaboration opportunities.",
                "job_specific": "Schedule weekly coffee with other creatives. Share work-in-progress for feedback. Separate work-alone time from community time.",
            })
    
    # ========================================================================
    # MANAGEMENT / LEADERSHIP
    # ========================================================================
    elif job_category == "management":
        
        # High stress + decision fatigue
        if predictions["stress_level"]["at_risk"] and avg["work_hours"] > 9:
            recommendations.append({
                "category": "leadership_stress",
                "priority": "high",
                "advice": "Manage decision fatigue: delegate more decisions to team, create decision frameworks (reduce re-thinking), time-box decision-making, protect personal time ruthlessly",
                "rationale": "Leaders face constant decisions and emotional labor. Without boundaries, responsibility becomes overwhelming, impairing judgment.",
                "job_specific": "Use Eisenhower matrix daily. Block 'no meeting' days for strategic thinking. Remember: your health enables team performance.",
            })
        
        # Poor work-life balance
        if avg["work_hours"] > 10 or avg["emails_received"] > 100:
            recommendations.append({
                "category": "work_life_balance",
                "priority": "high",
                "advice": "Model healthy boundaries: set working hours expectations with team, use email delay-send for off-hours, take full vacation (no check-ins), delegate more authority",
                "rationale": "Managers who work unsustainable hours signal to team that overwork is expected. This creates toxic culture and burnout cascade.",
                "job_specific": "Your availability doesn't prove leadership quality. Empower team to solve problems without you. Trust enables scale.",
            })
        
        # Emotional exhaustion from team issues
        if predictions["anxiety_score"]["at_risk"] or predictions["mood_score"]["at_risk"]:
            recommendations.append({
                "category": "emotional_labor",
                "priority": "medium",
                "advice": "Manage emotional toll: seek executive coach or therapist, join peer manager group, separate team problems from personal identity, practice emotional detachment",
                "rationale": "Managers absorb team stress, handle conflicts, and make difficult decisions. Without emotional processing, empathy erodes into cynicism.",
                "job_specific": "Not every team problem is yours to solve. Set boundaries around 'taking work home' emotionally. Celebrate small wins to counter negativity bias.",
            })
        
        # Low job satisfaction
        if predictions["job_satisfaction"]["at_risk"]:
            recommendations.append({
                "category": "career",
                "priority": "high",
                "advice": "Reassess management fit: reflect if management aligns with values, discuss concerns with your manager, clarify decision-making authority, consider individual contributor return",
                "rationale": "Not everyone thrives in management. If misalignment is systemic (company culture, lack of support), no amount of self-care will fix it.",
                "job_specific": "Management isn't the only path to impact. If you miss hands-on work, IC senior roles exist. It's okay to step back.",
            })
    
    # ========================================================================
    # SERVICE WORKERS (Customer Service, Hospitality)
    # ========================================================================
    elif job_category == "service":
        
        # Emotional exhaustion from customer interactions
        if predictions["stress_level"]["at_risk"] or predictions["mood_score"]["at_risk"]:
            recommendations.append({
                "category": "emotional_labor",
                "priority": "high",
                "advice": "Manage emotional labor: practice 'customer service voice' as professional mask (not authentic self), debrief difficult customers with colleagues, use breaks to decompress",
                "rationale": "Service work requires constant emotional regulation - smiling through rudeness, suppressing frustration. This is exhausting and alienating.",
                "job_specific": "After difficult customer, take 2min breathing break. Remind yourself: rudeness isn't personal, it's their bad day. Find humor with coworkers.",
            })
        
        # Physical fatigue from standing/moving
        if predictions["energy_level"]["at_risk"] or avg["steps_count"] > 15000:
            recommendations.append({
                "category": "physical_health",
                "priority": "medium",
                "advice": "Reduce physical strain: wear compression socks, use supportive shoes (invest in quality), stretch between shifts, ice sore areas, request standing mat if cashier",
                "rationale": "Prolonged standing/walking causes cumulative physical damage - foot pain, varicose veins, back problems. Physical exhaustion drains mental energy.",
                "job_specific": "Alternate standing position (shift weight). Use breaks to sit with feet elevated. Consider orthotics if chronic pain.",
            })
        
        # Irregular schedules
        if avg["sleep_hours"] < 7 or avg["sleep_quality"] < 6:
            recommendations.append({
                "category": "sleep",
                "priority": "high",
                "advice": "Cope with shift work: request consistent schedule where possible, use blackout curtains, maintain same sleep time on days off, strategic caffeine (not after 2pm)",
                "rationale": "Rotating or late shifts disrupt circadian rhythm. Poor sleep compounds stress from customer interactions and physical demands.",
                "job_specific": "If closing shift, wind-down routine before bed (no screens). If opening shift, prepare night before to reduce morning stress.",
            })
        
        # Low job satisfaction + limited growth
        if predictions["job_satisfaction"]["at_risk"]:
            recommendations.append({
                "category": "career",
                "priority": "medium",
                "advice": "Explore growth: ask about supervisor training, cross-train in other departments, use job as funding for school/certifications, network for opportunities",
                "rationale": "Service jobs often lack clear advancement paths. Without goals, work feels meaningless, increasing burnout risk.",
                "job_specific": "Set 6-month goal (promotion, new skill, side business). Even if staying temporarily, progress mindset reduces resentment.",
            })
    
    # ========================================================================
    # MANUAL LABOR / SKILLED TRADES
    # ========================================================================
    elif job_category == "manual_labor":
        
        # Physical strain + fatigue
        if predictions["energy_level"]["at_risk"] or avg["work_hours"] > 9:
            recommendations.append({
                "category": "physical_health",
                "priority": "high",
                "advice": "Prevent injury: proper lifting technique always (bend knees, not back), stretch before/after shift, use ergonomic tools, report pain early before chronic",
                "rationale": "Physical jobs have cumulative injury risk. Ignoring pain leads to chronic conditions that can end careers. Prevention > treatment.",
                "job_specific": "Wear proper PPE even when inconvenient. Use mechanical assistance (dollies, lifts) - don't be 'tough'. Ice sore muscles after shift.",
            })
        
        # Sleep deprivation from early starts
        if avg["sleep_hours"] < 7 or avg["sleep_quality"] < 6:
            recommendations.append({
                "category": "sleep",
                "priority": "high",
                "advice": "Prioritize recovery sleep: go to bed 8h before wake-up (non-negotiable), keep bedroom dark and cool, limit alcohol (disrupts deep sleep), nap on breaks if needed",
                "rationale": "Physical recovery happens during sleep. Sleep deprivation increases injury risk, slows healing, and impairs safety judgment.",
                "job_specific": "If 5am start, in bed by 9pm. Use alarm across room to ensure waking. Weekend sleep-in doesn't compensate for weekday debt.",
            })
        
        # High stress from deadlines/quotas
        if predictions["stress_level"]["at_risk"]:
            recommendations.append({
                "category": "stress",
                "priority": "medium",
                "advice": "Manage work pressure: communicate realistic timelines to supervisors, pace yourself to avoid injury, take full breaks (don't skip lunch), use weekends for full rest",
                "rationale": "Pressure to work faster increases injury risk and decreases quality. Rushing is how accidents happen.",
                "job_specific": "Safety over speed, always. If quotas are unrealistic, document and escalate. Your body has to last 40+ years.",
            })
        
        # Low job satisfaction
        if predictions["job_satisfaction"]["at_risk"]:
            recommendations.append({
                "category": "career",
                "priority": "medium",
                "advice": "Explore advancement: pursue certifications (electrician, HVAC, etc.), join union for better conditions, consider foreman/management track, research better-paying companies",
                "rationale": "Manual labor satisfaction often tied to pay, respect, and autonomy. Skilled trades offer good pay but require investment in training.",
                "job_specific": "Research apprenticeship programs. Specialized skills (welding, electrical) command higher pay and respect. Network with senior workers.",
            })
    
    # ========================================================================
    # SALES & MARKETING
    # ========================================================================
    elif job_category == "sales_marketing":
        
        # High stress from targets/quotas
        if predictions["stress_level"]["at_risk"] or predictions["anxiety_score"]["at_risk"]:
            recommendations.append({
                "category": "performance_pressure",
                "priority": "high",
                "advice": "Manage quota stress: focus on activities (calls, meetings) not just outcomes, track leading indicators, celebrate small wins, don't personalize rejection",
                "rationale": "Sales stress comes from outcome focus and rejection. Without reframing, anxiety compounds, harming performance (desperation repels buyers).",
                "job_specific": "Keep 'activity scoreboard' - controllable metrics. Bad month? Review process, not self-worth. Top sellers have thick skin from practice.",
            })
        
        # Irregular schedule + client demands
        if avg["work_hours"] > 9 or predictions["mood_score"]["at_risk"]:
            recommendations.append({
                "category": "work_life_balance",
                "priority": "medium",
                "advice": "Set client boundaries: establish communication hours (no nights/weekends), batch client calls, use scheduling tools (Calendly), protect personal time",
                "rationale": "Sales culture often glorifies 'always on' availability. This leads to burnout, resentment, and actually reduces effectiveness.",
                "job_specific": "Prospects respect boundaries (shows professionalism). Use auto-responders for off-hours. Top performers work smart, not endless.",
            })
        
        # Emotional rollercoaster from wins/losses
        if predictions["mood_score"]["at_risk"] or avg["mood_score"] < 6:
            recommendations.append({
                "category": "emotional_regulation",
                "priority": "medium",
                "advice": "Stabilize emotional swings: keep gratitude journal (3 wins daily), separate identity from performance, build non-work interests, talk to supportive friends",
                "rationale": "Sales creates emotional volatility - highs from wins, lows from losses. Without emotional stability, burnout or substance abuse risk increases.",
                "job_specific": "Don't let one bad call ruin the day. Use 'reset routine' (walk, music) between meetings. Remember: 70% close rate means 30% rejection is normal.",
            })
        
        # Low job satisfaction from misaligned values
        if predictions["job_satisfaction"]["at_risk"]:
            recommendations.append({
                "category": "career",
                "priority": "high",
                "advice": "Reassess product fit: ensure you believe in what you sell (ethic misalignment destroys soul), discuss concerns with leadership, consider industry/company change",
                "rationale": "Selling something you don't believe in creates cognitive dissonance and moral injury. No commission is worth self-respect.",
                "job_specific": "If product is genuinely helpful, reframe sales as problem-solving. If not, exit strategy. Plenty of sales jobs for ethical products.",
            })
    
    # ========================================================================
    # OTHER / DEFAULT
    # ========================================================================
    else:
        # General recommendations if job category unknown
        if predictions["stress_level"]["at_risk"]:
            recommendations.append({
                "category": "stress",
                "priority": "high",
                "advice": "Manage stress: practice daily relaxation (meditation, deep breathing, progressive muscle relaxation), identify stress triggers, seek therapy if persistent",
                "rationale": "Chronic stress damages physical and mental health. Early intervention prevents escalation to anxiety disorders or burnout.",
                "job_specific": None,
            })
        
        if avg["sleep_hours"] < 7:
            recommendations.append({
                "category": "sleep",
                "priority": "high",
                "advice": "Improve sleep: maintain consistent schedule, create bedtime routine, avoid screens 1h before bed, keep bedroom cool and dark, limit caffeine after 2pm",
                "rationale": "Sleep is foundation of mental and physical health. Most adults need 7-9h for optimal functioning.",
                "job_specific": None,
            })
        
        if avg["exercise_minutes"] < 30:
            recommendations.append({
                "category": "physical_health",
                "priority": "medium",
                "advice": "Increase activity: aim for 30min moderate exercise daily (brisk walk counts), take movement breaks every hour, use stairs, park farther away",
                "rationale": "Regular exercise reduces stress, improves mood, and enhances cognitive function. Small changes compound over time.",
                "job_specific": None,
            })
        
        if predictions["job_satisfaction"]["at_risk"]:
            recommendations.append({
                "category": "career",
                "priority": "high",
                "advice": "Address job dissatisfaction: identify specific dissatisfaction sources, discuss with manager, explore growth opportunities, consider career counseling",
                "rationale": "Job dissatisfaction affects all life domains. Early intervention can prevent burnout or misguided career changes.",
                "job_specific": None,
            })
    
    # ========================================================================
    # UNIVERSAL RECOMMENDATIONS (add to all categories if relevant)
    # ========================================================================
    
    # Depression - always critical
    if predictions["depression_score"]["at_risk"] and not any(r["category"] == "mental_health" for r in recommendations):
        recommendations.append({
            "category": "mental_health",
            "priority": "high",
            "advice": "URGENT: Seek professional help immediately. Contact therapist, psychiatrist, or Employee Assistance Program (EAP). Depression is treatable - you don't have to suffer.",
            "rationale": "Depression is a medical condition, not weakness. It affects work performance, relationships, and safety. Early treatment dramatically improves outcomes.",
            "job_specific": None,
        })
    
    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
    
    return recommendations


# ============================================================================
# MENTAL HEALTH HISTORY TRACKING
# ============================================================================

def load_user_history(user_id: str, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Load past assessment profiles for a user.
    
    Returns list of past profiles sorted by timestamp (oldest first).
    """
    history = []
    
    if not output_dir.exists():
        return history
    
    # Find all JSON files for this user
    pattern = f"profile_{user_id}_*.json"
    for file_path in output_dir.glob(pattern):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                history.append(data)
        except Exception as e:
            print(f"⚠️  Could not load {file_path}: {e}")
    
    # Sort by timestamp
    history.sort(key=lambda x: x.get("timestamp", ""))
    
    return history


def analyze_mental_health_trends(
    current_predictions: Dict[str, Dict[str, Any]],
    history: List[Dict[str, Any]],
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """
    Analyze mental health trends from historical assessments.
    
    Returns:
      - trends: Dict mapping target → {direction, change, assessments}
      - summary: Overall trajectory (improving/stable/declining/mixed)
      - alerts: List of concerning patterns
      - insights: List of positive patterns
    """
    if not history:
        return {
            "trends": {},
            "summary": "first_assessment",
            "alerts": [],
            "insights": ["This is your first assessment. Future assessments will track trends."],
            "assessment_count": 0,
        }
    
    # Filter history by lookback window
    cutoff = datetime.now() - timedelta(days=lookback_days)
    recent_history = [
        h for h in history
        if datetime.fromisoformat(h["timestamp"]) >= cutoff
    ]
    
    if not recent_history:
        return {
            "trends": {},
            "summary": "no_recent_history",
            "alerts": [],
            "insights": [f"No assessments found in last {lookback_days} days. This is effectively a new baseline."],
            "assessment_count": 0,
        }
    
    # Analyze each target's trend
    trends = {}
    improving_count = 0
    declining_count = 0
    stable_count = 0
    
    for target in ALL_TARGETS:
        # Get historical values
        historical_values = []
        for h in recent_history:
            if target in h.get("predictions", {}):
                val = h["predictions"][target].get("value")
                timestamp = datetime.fromisoformat(h["timestamp"])
                if val is not None:
                    historical_values.append((timestamp, val))
        
        if len(historical_values) < 1:
            continue
        
        # Current value
        current_value = current_predictions[target]["value"]
        
        # Calculate trend
        past_values = [v for _, v in historical_values]
        past_avg = np.mean(past_values)
        most_recent = past_values[-1] if past_values else past_avg
        
        # Determine if lower or higher is better
        inverted = target in INVERTED_TARGETS
        
        # Change from most recent
        change = current_value - most_recent
        pct_change = (change / most_recent * 100) if most_recent != 0 else 0
        
        # Change from average
        change_from_avg = current_value - past_avg
        
        # Classify trend direction
        # For inverted targets (mood, energy, etc.), higher is better
        # For normal targets (stress, anxiety, etc.), lower is better
        threshold = 0.5  # Minimum change to be considered significant
        
        if inverted:
            # Higher is better (mood, energy, focus, job_satisfaction)
            if change > threshold:
                direction = "improving"
                improving_count += 1
            elif change < -threshold:
                direction = "declining"
                declining_count += 1
            else:
                direction = "stable"
                stable_count += 1
        else:
            # Lower is better (stress, anxiety, depression, PSS)
            if change < -threshold:
                direction = "improving"
                improving_count += 1
            elif change > threshold:
                direction = "declining"
                declining_count += 1
            else:
                direction = "stable"
                stable_count += 1
        
        # Store trend data
        trends[target] = {
            "direction": direction,
            "current_value": round(current_value, 2),
            "previous_value": round(most_recent, 2),
            "change": round(change, 2),
            "percent_change": round(pct_change, 1),
            "historical_avg": round(past_avg, 2),
            "assessment_count": len(historical_values) + 1,  # +1 for current
            "at_risk_current": current_predictions[target]["at_risk"],
        }
    
    # Overall summary
    total_trends = improving_count + declining_count + stable_count
    if total_trends == 0:
        summary = "insufficient_data"
    elif declining_count > improving_count * 1.5:
        summary = "declining"
    elif improving_count > declining_count * 1.5:
        summary = "improving"
    elif stable_count > (improving_count + declining_count):
        summary = "stable"
    else:
        summary = "mixed"
    
    # Generate alerts for concerning patterns
    alerts = []
    
    # Check for declining critical indicators
    critical_targets = ["depression_score", "anxiety_score", "perceived_stress_scale"]
    for target in critical_targets:
        if target in trends:
            trend = trends[target]
            if trend["direction"] == "declining" and trend["at_risk_current"]:
                alerts.append({
                    "severity": "high",
                    "message": f"{target.replace('_', ' ').title()} is worsening: {trend['previous_value']:.1f} → {trend['current_value']:.1f} ({trend['percent_change']:+.0f}%) and now AT RISK",
                    "action": "Consider immediate professional support (therapist, psychiatrist, EAP)",
                })
            elif trend["direction"] == "declining" and abs(trend["percent_change"]) > 20:
                alerts.append({
                    "severity": "medium",
                    "message": f"{target.replace('_', ' ').title()} increased significantly: {trend['previous_value']:.1f} → {trend['current_value']:.1f} ({trend['percent_change']:+.0f}%)",
                    "action": "Monitor closely and implement stress reduction strategies",
                })
    
    # Check for declining mood/energy/focus
    wellbeing_targets = ["mood_score", "energy_level", "focus_score", "job_satisfaction"]
    declining_wellbeing = [t for t in wellbeing_targets if t in trends and trends[t]["direction"] == "declining"]
    if len(declining_wellbeing) >= 2:
        targets_str = ", ".join([t.replace("_", " ").title() for t in declining_wellbeing])
        alerts.append({
            "severity": "medium",
            "message": f"Multiple wellbeing indicators declining: {targets_str}",
            "action": "Review recent life changes, stressors, or behavioral patterns that may be contributing",
        })
    
    # Check for persistent at-risk status
    persistent_risks = []
    for target in ALL_TARGETS:
        if target in trends and trends[target]["at_risk_current"]:
            # Check if was also at-risk in past
            past_at_risk_count = 0
            for h in recent_history[-3:]:  # Last 3 assessments
                if target in h.get("predictions", {}):
                    if h["predictions"][target].get("at_risk"):
                        past_at_risk_count += 1
            
            if past_at_risk_count >= 2:  # At-risk in 2+ past assessments
                persistent_risks.append(target)
    
    if persistent_risks:
        targets_str = ", ".join([t.replace("_", " ").title() for t in persistent_risks])
        alerts.append({
            "severity": "high",
            "message": f"Persistent risk status across multiple assessments: {targets_str}",
            "action": "This indicates chronic issue requiring professional intervention, not just self-help strategies",
        })
    
    # Generate insights for positive patterns
    insights = []
    
    # Celebrate sustained improvement
    sustained_improvement = [t for t in trends if trends[t]["direction"] == "improving" and trends[t]["assessment_count"] >= 3]
    if sustained_improvement:
        targets_str = ", ".join([t.replace("_", " ").title() for t in sustained_improvement[:3]])
        insights.append(f"Sustained improvement across multiple assessments: {targets_str}. Your efforts are working!")
    
    # Highlight exiting at-risk status
    for target in ALL_TARGETS:
        if target in trends and not trends[target]["at_risk_current"]:
            # Check if was at-risk before
            if recent_history:
                last_assessment = recent_history[-1]
                if target in last_assessment.get("predictions", {}):
                    was_at_risk = last_assessment["predictions"][target].get("at_risk", False)
                    if was_at_risk:
                        insights.append(f"🎉 {target.replace('_', ' ').title()} no longer at-risk! Previous: {trends[target]['previous_value']:.1f}, Current: {trends[target]['current_value']:.1f}")
    
    # Highlight stable good health
    stable_healthy = [
        t for t in trends
        if trends[t]["direction"] == "stable" 
        and not trends[t]["at_risk_current"]
        and trends[t]["assessment_count"] >= 3
    ]
    if len(stable_healthy) >= 3:
        insights.append(f"Maintaining healthy levels across {len(stable_healthy)} indicators over {trends[stable_healthy[0]]['assessment_count']} assessments. Consistency is key!")
    
    # Days since first assessment
    first_assessment_date = datetime.fromisoformat(history[0]["timestamp"])
    days_tracking = (datetime.now() - first_assessment_date).days
    insights.append(f"Tracking mental health for {days_tracking} days across {len(history) + 1} assessments")
    
    return {
        "trends": trends,
        "summary": summary,
        "alerts": alerts,
        "insights": insights,
        "assessment_count": len(history) + 1,
        "tracking_days": days_tracking,
        "lookback_days": lookback_days,
    }


def generate_trend_recommendations(history_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate recommendations based on historical trends."""
    recommendations = []
    
    if history_analysis["summary"] == "first_assessment":
        recommendations.append({
            "category": "tracking",
            "priority": "low",
            "advice": "Schedule your next mental health check-in 2-4 weeks from now to establish trend data. Consistent tracking reveals patterns that single assessments miss.",
            "rationale": "Mental health fluctuates. Tracking trends helps differentiate temporary stress from chronic issues, and validates that interventions are working.",
            "job_specific": None,
        })
        return recommendations
    
    summary = history_analysis["summary"]
    alerts = history_analysis["alerts"]
    trends = history_analysis["trends"]
    
    # Declining trajectory
    if summary == "declining":
        recommendations.append({
            "category": "intervention",
            "priority": "high",
            "advice": "Your mental health shows declining trends across multiple indicators. This requires immediate attention: (1) Schedule therapy/counseling this week, (2) Review what changed (new stressor, stopped healthy habits), (3) Inform trusted person about struggles",
            "rationale": "Declining trends indicate current strategies aren't working. Early intervention prevents crisis. Waiting 'to see if it gets better' often leads to worse outcomes.",
            "job_specific": None,
        })
    
    # Improving trajectory
    if summary == "improving":
        recommendations.append({
            "category": "reinforcement",
            "priority": "medium",
            "advice": "Your mental health is improving! Identify what's working: What changed in your life? Which recommendations did you implement? Double down on effective strategies and maintain consistency.",
            "rationale": "Improvement validates your efforts. Explicitly naming what's working helps sustain progress and prevents backsliding into old patterns.",
            "job_specific": None,
        })
    
    # Mixed/volatile patterns
    if summary == "mixed":
        # Check for volatile patterns (large swings)
        volatile_targets = []
        for target, trend in trends.items():
            if abs(trend.get("percent_change", 0)) > 30:
                volatile_targets.append(target)
        
        if volatile_targets:
            recommendations.append({
                "category": "stability",
                "priority": "medium",
                "advice": "Your mental health shows high volatility (large swings between assessments). Focus on consistency: regular sleep schedule, steady routines, avoid boom-bust patterns (overwork → crash).",
                "rationale": "Volatility indicates unstable foundations. While ups feel good, the swings themselves are exhausting. Stability provides sustainable wellbeing.",
                "job_specific": None,
            })
    
    # Persistent risks
    high_alerts = [a for a in alerts if a["severity"] == "high"]
    if high_alerts:
        recommendations.append({
            "category": "chronic_care",
            "priority": "high",
            "advice": "You have persistent at-risk indicators across multiple assessments. This suggests chronic issues requiring professional treatment, not just self-help. Consider: (1) Regular therapy, (2) Psychiatric evaluation for medication, (3) Disability accommodations at work if needed",
            "rationale": "Chronic mental health issues are like chronic physical conditions - they require ongoing professional management. Self-care alone is insufficient.",
            "job_specific": None,
        })
    
    # Assessment frequency recommendations
    if history_analysis["assessment_count"] >= 4:
        avg_days_between = history_analysis["tracking_days"] / (history_analysis["assessment_count"] - 1)
        
        if avg_days_between > 45:
            recommendations.append({
                "category": "tracking",
                "priority": "low",
                "advice": f"You're averaging {avg_days_between:.0f} days between assessments. Consider more frequent check-ins (every 2-4 weeks) for better trend detection and intervention timing.",
                "rationale": "Monthly check-ins catch deterioration earlier and provide more data points to validate what's helping vs. hurting.",
                "job_specific": None,
            })
        elif avg_days_between < 10:
            recommendations.append({
                "category": "tracking",
                "priority": "low",
                "advice": f"You're assessing every {avg_days_between:.0f} days. Mental health changes take time - consider spacing to every 2-3 weeks unless in crisis or actively adjusting interventions.",
                "rationale": "Too-frequent assessments can create anxiety about metrics rather than lived experience. Allow time for interventions to work before re-measuring.",
                "job_specific": None,
            })
    
    return recommendations


# ============================================================================
# PROFILE GENERATION
# ============================================================================

def generate_profile(
    user_id: str,
    user_data: pd.DataFrame,
    job: Optional[str],
    model: MentalHealthPredictor,
    checkpoint: Dict[str, Any],
    output_dir: Path = OUTPUT_DIR,
    enable_history: bool = True,
) -> UserProfile:
    """Generate complete user profile with predictions and analysis."""
    
    # Classify job category
    job_category = classify_job_category(job)
    
    # Get predictions
    predictions = predict_user(
        model,
        user_data,
        checkpoint["scaler_mean"],
        checkpoint["scaler_std"],
        checkpoint["risk_thresholds"],
    )
    
    # Analyze
    contradictions = detect_contradictions(user_data, predictions)
    risk_factors = identify_risk_factors(predictions, user_data)
    positive_factors = identify_positive_factors(predictions, user_data)
    data_quality, missing = calculate_data_quality(user_data)
    
    # Generate job-specific advice
    recommendations = generate_job_specific_advice(
        job_category,
        predictions,
        user_data,
        risk_factors,
    )
    
    # Mental health history tracking
    history_analysis = None
    if enable_history:
        history = load_user_history(user_id, output_dir)
        history_analysis = analyze_mental_health_trends(predictions, history)
        
        # Add trend-based recommendations
        trend_recommendations = generate_trend_recommendations(history_analysis)
        recommendations.extend(trend_recommendations)
        
        # Re-sort by priority after adding trend recommendations
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
    
    # Build profile
    profile = UserProfile(
        user_id=user_id,
        timestamp=datetime.now(),
        job_category=job_category,
        behavioral_data=user_data,
        predictions=predictions,
        contradictions=contradictions,
        risk_factors=risk_factors,
        positive_factors=positive_factors,
        recommendations=recommendations,
        history_analysis=history_analysis,
        data_quality_score=data_quality,
        missing_features=missing,
    )
    
    return profile


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def print_profile_summary(profile: UserProfile) -> None:
    """Print profile summary to console."""
    print(f"\n{'='*80}")
    print(f"MENTAL HEALTH PROFILE - User {profile.user_id}")
    print(f"{'='*80}")
    print(f"Generated: {profile.timestamp.strftime('%Y-%m-%d %H:%M')}")
    print(f"Job Category: {profile.job_category.replace('_', ' ').title()}")
    print(f"Data Quality: {profile.data_quality_score:.1%}")
    
    # Mental health history summary
    if profile.history_analysis:
        hist = profile.history_analysis
        print(f"\n📈 MENTAL HEALTH HISTORY")
        print(f"{'-'*80}")
        print(f"Assessment #{hist['assessment_count']} • Tracking: {hist['tracking_days']} days • Trajectory: {hist['summary'].upper()}")
        
        # Trajectory indicator
        trajectory_icons = {
            "improving": "📈 IMPROVING - Keep up the great work!",
            "declining": "📉 DECLINING - Immediate attention needed",
            "stable": "➡️  STABLE - Maintaining consistent levels",
            "mixed": "📊 MIXED - Some improving, some declining",
            "first_assessment": "🆕 FIRST ASSESSMENT - Establishing baseline",
            "no_recent_history": "⏳ NO RECENT DATA - Starting fresh",
        }
        print(f"    {trajectory_icons.get(hist['summary'], hist['summary'])}")
        
        # Alerts (if any)
        if hist["alerts"]:
            print(f"\n  ⚠️  TREND ALERTS:")
            for i, alert in enumerate(hist["alerts"], 1):
                severity_icon = "🔴" if alert["severity"] == "high" else "🟡"
                print(f"    {i}. {severity_icon} {alert['message']}")
                print(f"       → Action: {alert['action']}")
        
        # Insights
        if hist["insights"]:
            print(f"\n  💡 INSIGHTS:")
            for i, insight in enumerate(hist["insights"], 1):
                print(f"    {i}. {insight}")
        
        # Detailed trends (top 5 most significant)
        if hist["trends"]:
            print(f"\n  📊 KEY TRENDS (last {hist['lookback_days']} days):")
            
            # Sort by absolute percent change
            sorted_trends = sorted(
                hist["trends"].items(),
                key=lambda x: abs(x[1].get("percent_change", 0)),
                reverse=True
            )[:5]
            
            for target, trend in sorted_trends:
                direction_icon = {
                    "improving": "✓",
                    "declining": "⚠️",
                    "stable": "→",
                }.get(trend["direction"], "?")
                
                risk_badge = " [AT RISK]" if trend["at_risk_current"] else ""
                
                print(f"    {direction_icon} {target.replace('_', ' ').title():30s}: "
                      f"{trend['previous_value']:>5.1f} → {trend['current_value']:>5.1f} "
                      f"({trend['percent_change']:+.0f}%){risk_badge}")
    
    # Predictions
    print(f"\n📊 PREDICTIONS (Next Day / End of Week)")
    print(f"{'-'*80}")
    
    # Daily targets
    print("\n🔹 DAILY PREDICTIONS (Next Day):")
    for target in DAILY_TARGETS:
        pred = profile.predictions[target]
        risk_icon = "⚠️ " if pred["at_risk"] else "✓ "
        print(f"  {risk_icon}{target.replace('_', ' ').title():20s}: {pred['value']:>5.1f} (confidence: {pred['confidence']:.0%})")
    
    # Weekly targets
    print("\n🔹 WEEKLY PREDICTIONS (End of Week):")
    for target in WEEKLY_TARGETS:
        pred = profile.predictions[target]
        risk_icon = "⚠️ " if pred["at_risk"] else "✓ "
        print(f"  {risk_icon}{target.replace('_', ' ').title():30s}: {pred['value']:>5.1f} (confidence: {pred['confidence']:.0%})")
    
    # Risk factors
    if profile.risk_factors:
        print(f"\n⚠️  RISK FACTORS ({len(profile.risk_factors)}):")
        for i, risk in enumerate(profile.risk_factors[:5], 1):
            print(f"  {i}. {risk}")
        if len(profile.risk_factors) > 5:
            print(f"  ... and {len(profile.risk_factors) - 5} more")
    
    # Positive factors
    if profile.positive_factors:
        print(f"\n✓ POSITIVE FACTORS ({len(profile.positive_factors)}):")
        for i, positive in enumerate(profile.positive_factors[:5], 1):
            print(f"  {i}. {positive}")
    
    # Contradictions
    if profile.contradictions:
        print(f"\n🔍 CONTRADICTIONS DETECTED ({len(profile.contradictions)}):")
        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_contradictions = sorted(profile.contradictions, key=lambda x: severity_order.get(x["severity"], 3))
        
        for i, contra in enumerate(sorted_contradictions, 1):
            severity_icon = "🔴" if contra["severity"] == "high" else "⚠️" if contra["severity"] == "medium" else "ℹ️"
            print(f"\n  {i}. {severity_icon} {contra['type'].replace('_', ' ').upper()} ({contra['severity'].upper()})")
            print(f"     {contra['description']}")
            if contra.get('explanation'):
                print(f"     💭 {contra['explanation']}")
            if contra.get('recommendation'):
                print(f"     💡 {contra['recommendation']}")
    
    # Job-specific recommendations
    if profile.recommendations:
        print(f"\n💡 RECOMMENDATIONS ({profile.job_category.replace('_', ' ').title()}):")
        print(f"{'-'*80}")
        
        for i, rec in enumerate(profile.recommendations[:10], 1):  # Show top 10
            priority_icon = "🔴" if rec["priority"] == "high" else "🟡" if rec["priority"] == "medium" else "🟢"
            print(f"\n  {i}. {priority_icon} {rec['category'].replace('_', ' ').upper()} - {rec['priority'].upper()} PRIORITY")
            print(f"     ▸ {rec['advice']}")
            print(f"     ▸ WHY: {rec['rationale']}")
            if rec.get("job_specific"):
                print(f"     ▸ JOB-SPECIFIC: {rec['job_specific']}")
        
        if len(profile.recommendations) > 10:
            print(f"\n  ... and {len(profile.recommendations) - 10} more recommendations")
    
    print(f"\n{'='*80}\n")


def save_profile_json(profile: UserProfile, output_dir: Path) -> Path:
    """Save profile as JSON for programmatic access."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"profile_{profile.user_id}_{profile.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    
    profile_dict = {
        "user_id": profile.user_id,
        "timestamp": profile.timestamp.isoformat(),
        "job_category": profile.job_category,
        "data_quality_score": profile.data_quality_score,
        "missing_features": profile.missing_features,
        "predictions": profile.predictions,
        "risk_factors": profile.risk_factors,
        "positive_factors": profile.positive_factors,
        "contradictions": profile.contradictions,
        "recommendations": profile.recommendations,
        "history_analysis": profile.history_analysis,
    }
    
    with open(output_path, "w") as f:
        json.dump(profile_dict, f, indent=2)
    
    print(f"✓ Saved JSON profile: {output_path}")
    return output_path


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def generate_html_report(profile: UserProfile, output_dir: Path) -> Path:
    """Generate comprehensive HTML report with charts and styling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"report_{profile.user_id}_{profile.timestamp.strftime('%Y%m%d_%H%M%S')}.html"
    
    # Helper functions for HTML generation
    def get_status_color(at_risk: bool) -> str:
        return "#dc3545" if at_risk else "#28a745"
    
    def get_priority_color(priority: str) -> str:
        return {"high": "#dc3545", "medium": "#ffc107", "low": "#17a2b8"}.get(priority, "#6c757d")
    
    def get_trajectory_color(summary: str) -> str:
        colors = {
            "improving": "#28a745",
            "declining": "#dc3545",
            "stable": "#17a2b8",
            "mixed": "#ffc107",
        }
        return colors.get(summary, "#6c757d")
    
    def get_trajectory_icon(summary: str) -> str:
        icons = {
            "improving": "📈",
            "declining": "📉",
            "stable": "➡️",
            "mixed": "📊",
            "first_assessment": "🆕",
        }
        return icons.get(summary, "📊")
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Profile - User {profile.user_id}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px 40px;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
        }}
        
        .metadata-item {{
            text-align: center;
        }}
        
        .metadata-label {{
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        
        .metadata-value {{
            font-size: 1.5em;
            font-weight: 600;
            color: #495057;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #495057;
            border-left: 5px solid #667eea;
            padding-left: 15px;
        }}
        
        .trajectory-banner {{
            background: linear-gradient(135deg, {get_trajectory_color(profile.history_analysis['summary'] if profile.history_analysis else 'stable')} 0%, 
                        {get_trajectory_color(profile.history_analysis['summary'] if profile.history_analysis else 'stable')}dd 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        .trajectory-banner h2 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        .trajectory-banner p {{
            font-size: 1.2em;
            opacity: 0.95;
        }}
        
        .predictions-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .prediction-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            border: 3px solid;
        }}
        
        .prediction-card.at-risk {{
            border-color: #dc3545;
            background: #fff5f5;
        }}
        
        .prediction-card.healthy {{
            border-color: #28a745;
            background: #f0fff4;
        }}
        
        .prediction-name {{
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .prediction-value {{
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .prediction-status {{
            font-size: 0.95em;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: 20px;
            display: inline-block;
        }}
        
        .status-at-risk {{
            background: #dc3545;
            color: white;
        }}
        
        .status-healthy {{
            background: #28a745;
            color: white;
        }}
        
        .prediction-confidence {{
            font-size: 0.85em;
            color: #6c757d;
            margin-top: 8px;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .alert-box {{
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 5px solid;
        }}
        
        .alert-high {{
            background: #fff5f5;
            border-color: #dc3545;
        }}
        
        .alert-medium {{
            background: #fffbf0;
            border-color: #ffc107;
        }}
        
        .alert-low {{
            background: #f0f8ff;
            border-color: #17a2b8;
        }}
        
        .alert-title {{
            font-weight: 700;
            font-size: 1.1em;
            margin-bottom: 8px;
        }}
        
        .alert-message {{
            margin-bottom: 8px;
            line-height: 1.5;
        }}
        
        .alert-action {{
            font-style: italic;
            color: #495057;
            margin-top: 5px;
        }}
        
        .recommendation-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 5px solid;
        }}
        
        .recommendation-card.priority-high {{
            border-color: #dc3545;
        }}
        
        .recommendation-card.priority-medium {{
            border-color: #ffc107;
        }}
        
        .recommendation-card.priority-low {{
            border-color: #17a2b8;
        }}
        
        .rec-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .rec-category {{
            font-size: 1.2em;
            font-weight: 700;
            color: #495057;
            text-transform: capitalize;
        }}
        
        .rec-priority {{
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            color: white;
        }}
        
        .rec-advice {{
            font-size: 1.05em;
            line-height: 1.7;
            margin-bottom: 15px;
            color: #212529;
        }}
        
        .rec-rationale {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #667eea;
        }}
        
        .rec-rationale-label {{
            font-weight: 600;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .rec-job-specific {{
            background: #e7f3ff;
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid #17a2b8;
        }}
        
        .rec-job-label {{
            font-weight: 600;
            color: #17a2b8;
            margin-bottom: 5px;
        }}
        
        .list-item {{
            padding: 12px;
            margin-bottom: 8px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 3px solid #667eea;
        }}
        
        .contradiction-item {{
            padding: 15px;
            margin-bottom: 12px;
            background: #fffbf0;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }}
        
        .trend-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin-bottom: 10px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        .trend-name {{
            font-weight: 600;
            color: #495057;
        }}
        
        .trend-values {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .trend-arrow {{
            font-size: 1.5em;
        }}
        
        .trend-change {{
            font-weight: 700;
            font-size: 1.1em;
        }}
        
        .insight-box {{
            background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            border-left: 5px solid #00acc1;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #6c757d;
            border-top: 2px solid #dee2e6;
        }}
        
        .footer-note {{
            font-size: 0.9em;
            line-height: 1.8;
            max-width: 800px;
            margin: 0 auto;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🧠 Mental Health Profile</h1>
            <div class="subtitle">Comprehensive Assessment & Recommendations</div>
        </div>
        
        <!-- Metadata -->
        <div class="metadata">
            <div class="metadata-item">
                <div class="metadata-label">User ID</div>
                <div class="metadata-value">{profile.user_id}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Generated</div>
                <div class="metadata-value">{profile.timestamp.strftime('%b %d, %Y')}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Job Category</div>
                <div class="metadata-value">{profile.job_category.replace('_', ' ').title()}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Data Quality</div>
                <div class="metadata-value">{profile.data_quality_score:.0%}</div>
            </div>
        </div>
        
        <div class="content">
"""
    
    # Mental Health History Section
    if profile.history_analysis and profile.history_analysis["summary"] != "first_assessment":
        hist = profile.history_analysis
        html += f"""
            <!-- Mental Health History -->
            <div class="section">
                <div class="trajectory-banner">
                    <h2>{get_trajectory_icon(hist['summary'])} Mental Health Trajectory: {hist['summary'].upper()}</h2>
                    <p>Assessment #{hist['assessment_count']} • Tracking for {hist['tracking_days']} days</p>
                </div>
"""
        
        # Alerts
        if hist["alerts"]:
            html += """
                <div style="margin-bottom: 30px;">
                    <h3 style="font-size: 1.5em; margin-bottom: 15px; color: #dc3545;">⚠️ Trend Alerts</h3>
"""
            for alert in hist["alerts"]:
                alert_class = f"alert-{alert['severity']}"
                html += f"""
                    <div class="alert-box {alert_class}">
                        <div class="alert-title">{'🔴' if alert['severity'] == 'high' else '🟡'} {alert['message']}</div>
                        <div class="alert-action">→ Action Required: {alert['action']}</div>
                    </div>
"""
            html += """
                </div>
"""
        
        # Insights
        if hist["insights"]:
            html += """
                <div style="margin-bottom: 30px;">
                    <h3 style="font-size: 1.5em; margin-bottom: 15px; color: #00acc1;">💡 Insights</h3>
"""
            for insight in hist["insights"]:
                html += f"""
                    <div class="insight-box">{insight}</div>
"""
            html += """
                </div>
"""
        
        # Key Trends
        if hist["trends"]:
            sorted_trends = sorted(
                hist["trends"].items(),
                key=lambda x: abs(x[1].get("percent_change", 0)),
                reverse=True
            )[:8]
            
            html += f"""
                <div style="margin-bottom: 30px;">
                    <h3 style="font-size: 1.5em; margin-bottom: 15px; color: #667eea;">📊 Key Trends (Last {hist['lookback_days']} Days)</h3>
"""
            for target, trend in sorted_trends:
                direction_icon = {
                    "improving": "✅",
                    "declining": "⚠️",
                    "stable": "➡️",
                }.get(trend["direction"], "?")
                
                change_color = "#28a745" if trend["direction"] == "improving" else "#dc3545" if trend["direction"] == "declining" else "#6c757d"
                risk_badge = ' <span style="background: #dc3545; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em;">AT RISK</span>' if trend["at_risk_current"] else ""
                
                html += f"""
                    <div class="trend-item">
                        <div class="trend-name">{direction_icon} {target.replace('_', ' ').title()}{risk_badge}</div>
                        <div class="trend-values">
                            <span>{trend['previous_value']:.1f}</span>
                            <span class="trend-arrow" style="color: {change_color};">→</span>
                            <span class="trend-change" style="color: {change_color};">{trend['current_value']:.1f}</span>
                            <span style="color: {change_color}; font-weight: 600;">({trend['percent_change']:+.0f}%)</span>
                        </div>
                    </div>
"""
            html += """
                </div>
"""
        
        html += """
            </div>
"""
    
    # Predictions Section
    html += """
            <!-- Predictions -->
            <div class="section">
                <h2 class="section-title">📊 Mental Health Predictions</h2>
                
                <h3 style="font-size: 1.3em; margin: 30px 0 15px 0; color: #495057;">Daily Predictions (Next Day)</h3>
                <div class="predictions-grid">
"""
    
    for target in DAILY_TARGETS:
        pred = profile.predictions[target]
        status_class = "at-risk" if pred["at_risk"] else "healthy"
        status_text = "AT RISK" if pred["at_risk"] else "HEALTHY"
        status_badge_class = "status-at-risk" if pred["at_risk"] else "status-healthy"
        
        html += f"""
                    <div class="prediction-card {status_class}">
                        <div class="prediction-name">{target.replace('_', ' ').title()}</div>
                        <div class="prediction-value" style="color: {get_status_color(pred['at_risk'])};">{pred['value']:.1f}</div>
                        <div class="prediction-status {status_badge_class}">{status_text}</div>
                        <div class="prediction-confidence">Confidence: {pred['confidence']:.0%}</div>
                    </div>
"""
    
    html += """
                </div>
                
                <h3 style="font-size: 1.3em; margin: 30px 0 15px 0; color: #495057;">Weekly Predictions (End of Week)</h3>
                <div class="predictions-grid">
"""
    
    for target in WEEKLY_TARGETS:
        pred = profile.predictions[target]
        status_class = "at-risk" if pred["at_risk"] else "healthy"
        status_text = "AT RISK" if pred["at_risk"] else "HEALTHY"
        status_badge_class = "status-at-risk" if pred["at_risk"] else "status-healthy"
        
        html += f"""
                    <div class="prediction-card {status_class}">
                        <div class="prediction-name">{target.replace('_', ' ').title()}</div>
                        <div class="prediction-value" style="color: {get_status_color(pred['at_risk'])};">{pred['value']:.1f}</div>
                        <div class="prediction-status {status_badge_class}">{status_text}</div>
                        <div class="prediction-confidence">Confidence: {pred['confidence']:.0%}</div>
                    </div>
"""
    
    html += """
                </div>
                
                <!-- Predictions Chart -->
                <div class="chart-container">
                    <canvas id="predictionsChart"></canvas>
                </div>
            </div>
"""
    
    # Behavioral Patterns Chart
    avg_behavior = profile.behavioral_data.mean()
    html += """
            <!-- Behavioral Patterns -->
            <div class="section">
                <h2 class="section-title">📈 Behavioral Patterns (7-Day Average)</h2>
                <div class="chart-container" style="height: 400px;">
                    <canvas id="behaviorChart"></canvas>
                </div>
            </div>
"""
    
    # Risk Factors Section
    if profile.risk_factors:
        html += """
            <!-- Risk Factors -->
            <div class="section">
                <h2 class="section-title">⚠️ Risk Factors</h2>
"""
        for risk in profile.risk_factors:
            html += f"""
                <div class="list-item">⚠️ {risk}</div>
"""
        html += """
            </div>
"""
    
    # Positive Factors Section
    if profile.positive_factors:
        html += """
            <!-- Positive Factors -->
            <div class="section">
                <h2 class="section-title">✅ Positive Factors</h2>
"""
        for positive in profile.positive_factors:
            html += f"""
                <div class="list-item" style="border-color: #28a745;">✅ {positive}</div>
"""
        html += """
            </div>
"""
    
    # Contradictions Section
    if profile.contradictions:
        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_contradictions = sorted(profile.contradictions, key=lambda x: severity_order.get(x.get("severity", "low"), 3))
        
        html += """
            <!-- Contradictions -->
            <div class="section">
                <h2 class="section-title">🔍 Contradictions Detected</h2>
                <p style="margin-bottom: 20px; color: #6c757d;">These patterns suggest inconsistencies between reported behaviors and predicted outcomes. They may indicate measurement errors, unique circumstances, or areas requiring closer attention.</p>
"""
        for contra in sorted_contradictions:
            severity_icon = "🔴" if contra['severity'] == 'high' else "⚠️" if contra['severity'] == 'medium' else "ℹ️"
            severity_color = "#dc3545" if contra['severity'] == 'high' else "#ffc107" if contra['severity'] == 'medium' else "#17a2b8"
            
            html += f"""
                <div class="contradiction-item" style="border-color: {severity_color};">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                        <strong style="font-size: 1.1em;">{severity_icon} {contra['type'].replace('_', ' ').title()}</strong>
                        <span style="background: {severity_color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">
                            {contra['severity'].upper()}
                        </span>
                    </div>
                    <div style="margin-bottom: 10px; line-height: 1.6;">
                        {contra['description']}
                    </div>
"""
            if contra.get('explanation'):
                html += f"""
                    <div style="background: rgba(102, 126, 234, 0.1); padding: 12px; border-radius: 6px; margin-bottom: 10px; border-left: 3px solid #667eea;">
                        <strong style="color: #667eea;">💭 Explanation:</strong><br>
                        {contra['explanation']}
                    </div>
"""
            if contra.get('recommendation'):
                html += f"""
                    <div style="background: rgba(40, 167, 69, 0.1); padding: 12px; border-radius: 6px; border-left: 3px solid #28a745;">
                        <strong style="color: #28a745;">💡 Recommendation:</strong><br>
                        {contra['recommendation']}
                    </div>
"""
            html += """
                </div>
"""
        html += """
            </div>
"""
    
    # Recommendations Section
    if profile.recommendations:
        html += f"""
            <!-- Recommendations -->
            <div class="section">
                <h2 class="section-title">💡 Personalized Recommendations</h2>
                <p style="margin-bottom: 25px; color: #6c757d; font-size: 1.05em;">
                    Tailored advice for <strong>{profile.job_category.replace('_', ' ').title()}</strong> professionals based on your predictions and behavioral patterns.
                </p>
"""
        
        for i, rec in enumerate(profile.recommendations[:15], 1):  # Show top 15
            priority_class = f"priority-{rec['priority']}"
            priority_color = get_priority_color(rec['priority'])
            
            html += f"""
                <div class="recommendation-card {priority_class}">
                    <div class="rec-header">
                        <div class="rec-category">{i}. {rec['category'].replace('_', ' ').title()}</div>
                        <div class="rec-priority" style="background: {priority_color};">{rec['priority']} Priority</div>
                    </div>
                    <div class="rec-advice">{rec['advice']}</div>
                    <div class="rec-rationale">
                        <div class="rec-rationale-label">💭 Why This Matters:</div>
                        {rec['rationale']}
                    </div>
"""
            if rec.get('job_specific'):
                html += f"""
                    <div class="rec-job-specific">
                        <div class="rec-job-label">👔 Job-Specific Advice:</div>
                        {rec['job_specific']}
                    </div>
"""
            html += """
                </div>
"""
        
        if len(profile.recommendations) > 15:
            html += f"""
                <div style="text-align: center; padding: 20px; color: #6c757d; font-style: italic;">
                    + {len(profile.recommendations) - 15} additional recommendations available in JSON export
                </div>
"""
        
        html += """
            </div>
"""
    
    # Footer
    html += f"""
        </div>
        
        <div class="footer">
            <div class="footer-note">
                <strong>⚠️ Important Disclaimer:</strong> This report is generated by a predictive model for informational purposes only. 
                It is NOT a substitute for professional medical advice, diagnosis, or treatment. If you're experiencing mental health 
                difficulties, please consult with a qualified healthcare provider, therapist, or counselor. In crisis, contact emergency 
                services or a crisis hotline immediately.
                <br><br>
                <strong>Report Generated:</strong> {profile.timestamp.strftime('%B %d, %Y at %I:%M %p')}<br>
                <strong>Model:</strong> Multi-Target Mental Health Predictor (LSTM/Transformer)<br>
                <strong>Data Quality Score:</strong> {profile.data_quality_score:.1%}
            </div>
        </div>
    </div>
    
    <script>
        // Predictions Chart
        const predictionsCtx = document.getElementById('predictionsChart').getContext('2d');
        new Chart(predictionsCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps([t.replace('_', ' ').title() for t in ALL_TARGETS])},
                datasets: [{{
                    label: 'Predicted Values',
                    data: {json.dumps([profile.predictions[t]['value'] for t in ALL_TARGETS])},
                    backgroundColor: {json.dumps([get_status_color(profile.predictions[t]['at_risk']) + '80' for t in ALL_TARGETS])},
                    borderColor: {json.dumps([get_status_color(profile.predictions[t]['at_risk']) for t in ALL_TARGETS])},
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    title: {{
                        display: true,
                        text: 'Mental Health Predictions Overview',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Predicted Score'
                        }}
                    }}
                }}
            }}
        }});
        
        // Behavioral Patterns Chart
        const behaviorCtx = document.getElementById('behaviorChart').getContext('2d');
        new Chart(behaviorCtx, {{
            type: 'radar',
            data: {{
                labels: {json.dumps([
                    'Sleep Hours', 'Sleep Quality', 'Work Hours', 'Exercise', 
                    'Social Interactions', 'Outdoor Time', 'Diet Quality', 'Work Pressure'
                ])},
                datasets: [{{
                    label: '7-Day Average',
                    data: [
                        {avg_behavior['sleep_hours']:.1f},
                        {avg_behavior['sleep_quality']:.1f},
                        {avg_behavior['work_hours']:.1f},
                        {avg_behavior['exercise_minutes'] / 10:.1f},  // Scale down for visibility
                        {avg_behavior['social_interactions']:.1f},
                        {avg_behavior['outdoor_time_minutes'] / 10:.1f},  // Scale down
                        {avg_behavior['diet_quality']:.1f},
                        {avg_behavior['work_pressure']:.1f}
                    ],
                    fill: true,
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgb(102, 126, 234)',
                    pointBackgroundColor: 'rgb(102, 126, 234)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgb(102, 126, 234)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    title: {{
                        display: true,
                        text: 'Behavioral Pattern Profile',
                        font: {{ size: 16, weight: 'bold' }}
                    }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true,
                        ticks: {{
                            stepSize: 2
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Saved HTML report: {output_path}")
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comprehensive mental health profiles")
    parser.add_argument("--csv", type=Path, required=True, help="Path to Google Form CSV")
    parser.add_argument("--user-id", type=str, help="Specific user ID to profile")
    parser.add_argument("--all-users", action="store_true", help="Generate profiles for all users")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="Path to trained model")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory for reports")
    parser.add_argument("--window", type=int, default=7, help="Days of input data")
    parser.add_argument("--no-history", action="store_true", help="Disable history tracking")
    parser.add_argument("--lookback-days", type=int, default=90, help="Days to look back for trend analysis")
    parser.add_argument("--html", action="store_true", help="Generate HTML reports")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print("MENTAL HEALTH PROFILING GENERATOR")
    print(f"{'='*80}\n")
    
    # Load model
    model, checkpoint = load_model(args.model_path)
    
    # Parse CSV
    user_data = parse_google_form_csv(args.csv, window=args.window)
    
    # Determine which users to process
    if args.user_id:
        if args.user_id not in user_data:
            print(f"❌ User ID '{args.user_id}' not found in CSV")
            return
        user_ids = [args.user_id]
    elif args.all_users:
        user_ids = list(user_data.keys())
    else:
        print("❌ Specify --user-id or --all-users")
        return
    
    print(f"\n🔄 Generating profiles for {len(user_ids)} user(s)...\n")
    
    # Generate profiles
    for user_id in user_ids:
        data_dict = user_data[user_id]
        
        profile = generate_profile(
            user_id=user_id,
            user_data=data_dict["data"],
            job=data_dict["job"],
            model=model,
            checkpoint=checkpoint,
            output_dir=args.output_dir,
            enable_history=not args.no_history,
        )
        
        # Output
        print_profile_summary(profile)
        save_profile_json(profile, args.output_dir)
        
        # Generate HTML if requested
        if args.html:
            generate_html_report(profile, args.output_dir)
    
    print(f"\n✓ Completed {len(user_ids)} profile(s)")
    print(f"✓ Reports saved to: {args.output_dir.resolve()}")
    if args.html:
        print(f"✓ HTML reports generated with interactive charts")


if __name__ == "__main__":
    main()
