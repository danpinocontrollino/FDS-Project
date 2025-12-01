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
   python scripts/predict_burnout.py --csv my_responses.csv

2. Interactive mode (answer questions):
   python scripts/predict_burnout.py --interactive

3. Single prediction with command line:
   python scripts/predict_burnout.py --stress 7 --sleep 6 --work 9 --mood 4

4. Specify custom model path:
   python scripts/predict_burnout.py --model-path ./downloads/lstm_sequence.pt

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
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models."""
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Transformer classifier matching the training architecture."""
    def __init__(
        self,
        input_dim: int = 17,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


class MAEClassifier(nn.Module):
    """
    Burnout classifier using pre-trained MAE encoder (65% accuracy).
    
    This model was pre-trained on 1.4M behavioral sequences using
    masked autoencoding, then fine-tuned for burnout classification.
    """
    
    def __init__(
        self,
        input_dim: int = 17,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Encoder components
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.input_proj(x)
        embeddings = self.pos_encoder(embeddings)
        encoded = self.transformer_encoder(embeddings)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


class CVAESmartAdvisor(nn.Module):
    """
    Conditional Variational Autoencoder for generating "what-if" behavioral suggestions.
    
    This model was trained to understand the latent patterns that distinguish
    low, medium, and high burnout behaviors. It can suggest behavioral changes
    by encoding your current week, then decoding with a "low burnout" condition.
    """
    
    def __init__(
        self,
        input_dim: int = 17,
        seq_len: int = 7,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_classes: int = 3,
        condition_dim: int = 16,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.flat_dim = seq_len * input_dim
        
        # Condition embedding: burnout level → vector
        self.condition_embedding = nn.Embedding(num_classes, condition_dim)
        
        # ENCODER
        encoder_input_dim = self.flat_dim + condition_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # DECODER
        decoder_input_dim = latent_dim + condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.flat_dim),
        )
    
    def encode(self, x: torch.Tensor, condition: torch.Tensor):
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        cond_emb = self.condition_embedding(condition)
        encoder_input = torch.cat([x_flat, cond_emb], dim=-1)
        hidden = self.encoder(encoder_input)
        return self.fc_mu(hidden), self.fc_logvar(hidden)
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor):
        batch_size = z.size(0)
        cond_emb = self.condition_embedding(condition)
        decoder_input = torch.cat([z, cond_emb], dim=-1)
        x_flat = self.decoder(decoder_input)
        return x_flat.reshape(batch_size, self.seq_len, self.input_dim)
    
    def suggest_changes(
        self,
        original_sequence: torch.Tensor,
        original_label: torch.Tensor,
        target_label: int = 0,
    ) -> torch.Tensor:
        """
        THE "SMART ADVISOR" FUNCTION
        
        Generate a counterfactual schedule: "What would this week look like
        if you had low burnout?"
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(original_sequence, original_label)
            target_cond = torch.tensor([target_label], device=mu.device)
            suggested = self.decode(mu, target_cond)
        return suggested


def load_cvae_advisor(model_dir: Path = None) -> Optional[Tuple[CVAESmartAdvisor, dict]]:
    """
    Load the CVAE Smart Advisor model if available.
    
    Returns:
        (model, stats_dict) or None if not available
    """
    if model_dir is None:
        model_dir = MODEL_DIR
    
    cvae_path = model_dir / "cvae_advisor.pt"
    
    if not cvae_path.exists():
        return None
    
    try:
        checkpoint = torch.load(cvae_path, map_location="cpu", weights_only=False)
        
        # Get model parameters from checkpoint
        input_dim = checkpoint.get("input_dim", 17)
        seq_len = checkpoint.get("seq_len", 7)
        latent_dim = checkpoint.get("latent_dim", 32)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        
        # Build model
        model = CVAESmartAdvisor(
            input_dim=input_dim,
            seq_len=seq_len,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        
        # Load weights
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        # Get normalization stats (may be in 'normalization' dict or directly in checkpoint)
        norm_dict = checkpoint.get("normalization", {})
        mean_val = norm_dict.get("mean") if norm_dict else checkpoint.get("mean")
        std_val = norm_dict.get("std") if norm_dict else checkpoint.get("std")
        
        stats = {
            "mean": mean_val,
            "std": std_val,
            "feature_cols": checkpoint.get("feature_cols", FEATURE_COLS),
        }
        
        print(f"✅ Loaded CVAE Smart Advisor from {cvae_path}")
        return model, stats
        
    except Exception as e:
        print(f"⚠️  Could not load CVAE advisor: {e}")
        return None


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
    elif "mae" in str(model_path).lower():
        model_type = "mae"
    
    # Build appropriate model with CORRECT input dimension
    state_dict = checkpoint["model_state"]
    
    if model_type == "lstm":
        model = LSTMClassifier(input_dim=input_dim)
    elif model_type == "gru":
        model = GRUClassifier(input_dim=input_dim)
    elif model_type == "transformer":
        d_model = checkpoint.get("d_model", 64)
        nhead = checkpoint.get("nhead", 4)
        num_layers = checkpoint.get("num_layers", 2)
        dim_feedforward = checkpoint.get("dim_feedforward", 256)
        model = TransformerClassifier(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )
        # Handle positional encoding buffer size mismatch (checkpoint has seq_len, model has 100)
        if "pos_encoder.pe" in state_dict:
            saved_pe = state_dict["pos_encoder.pe"]  # (1, seq_len, d_model)
            model_pe = model.pos_encoder.pe  # (1, 100, d_model)
            # Copy saved PE values to model's PE buffer
            seq_len = saved_pe.size(1)
            model_pe[:, :seq_len, :] = saved_pe
            # Remove from state_dict to avoid error
            state_dict = {k: v for k, v in state_dict.items() if k != "pos_encoder.pe"}
    elif model_type == "mae":
        # MAE classifier with its full architecture
        d_model = checkpoint.get("d_model", 64)
        nhead = checkpoint.get("nhead", 4)
        num_layers = checkpoint.get("num_layers", 2)
        dim_feedforward = checkpoint.get("dim_feedforward", 256)
        model = MAEClassifier(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )
    else:
        # Default to LSTM
        model = LSTMClassifier(input_dim=input_dim)
    
    # Load weights (use strict=False to handle any remaining mismatches)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"✅ Loaded {model_type.upper()} model from {model_path}")
    print(f"   Features: {len(feature_cols)} ({', '.join(feature_cols[:3])}...)")
    
    return model, model_type, feature_cols


# ============================================================================
# DATA PROCESSING
# ============================================================================

def parse_google_form_csv(csv_path: str) -> Tuple[pd.DataFrame, bool]:
    """
    Parse a Google Form CSV export into model-ready format.
    
    Supports two formats:
    1. Weekly averages: One row per person
    2. Daily data: 7 rows per person (grouped by name/email)
    
    Returns:
        (dataframe, is_daily): Processed data and whether it's daily format
    """
    df = pd.read_csv(csv_path)
    
    # Column name mapping (Google Form question -> feature name)
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
        
        # Identifiers
        "name": "_name",
        "email": "_email",
        "your name": "_name",
        "day": "_day",
        
        # === NEW V2 FEATURES ===
        # Social
        "quality of social": "social_quality",
        "social interaction": "social_quality",
        "lonely": "loneliness_level",
        "isolated": "loneliness_level",
        
        # Work Environment
        "work arrangement": "work_arrangement",
        "workspace type": "workspace_type",
        "distracting": "environment_distractions",
        "distraction": "environment_distractions",
        
        # Job Context
        "job type": "job_type",
        "job require": "job_requires_screen",
        "screen time required": "job_requires_screen",
        "take breaks": "break_flexibility",
        "break flexibility": "break_flexibility",
        
        # Boundaries & Recovery
        "work-life boundary": "work_life_boundary",
        "boundary": "work_life_boundary",
        "check work outside": "after_hours_checking",
        "outside hours": "after_hours_checking",
        "recover": "recovery_ability",
        "recovery": "recovery_ability",
        
        # === DATASET-ALIGNED FEATURES (High Priority) ===
        "meaningful social": "social_interactions",
        "conversations": "social_interactions",
        "social interactions": "social_interactions",
        "outdoor": "outdoor_time_minutes",
        "outside time": "outdoor_time_minutes",
        "time outdoors": "outdoor_time_minutes",
        "diet quality": "diet_quality",
        "nutrition": "diet_quality",
        "job satisfaction": "job_satisfaction",
        "satisfied with job": "job_satisfaction",
        "chronotype": "chronotype",
        "morning person": "chronotype",
        "night owl": "chronotype",
    }
    
    # V2 feature defaults
    V2_DEFAULTS = {
        "social_quality": 6,
        "loneliness_level": "Sometimes",
        "work_arrangement": "Hybrid",
        "workspace_type": "Shared office",
        "environment_distractions": 5,
        "job_type": "Other",
        "job_requires_screen": "Moderate, 2-4 hours",
        "break_flexibility": "Mostly, with some constraints",
        "work_life_boundary": "Somewhat blurred",
        "after_hours_checking": "Sometimes",
        "recovery_ability": 5,
        # Dataset-aligned features
        "social_interactions": 3,
        "outdoor_time_minutes": 30,
        "diet_quality": 5,
        "job_satisfaction": 6,
        "chronotype": "intermediate",
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
    
    # Fill V2 features with defaults if missing
    for feature, default in V2_DEFAULTS.items():
        if feature not in mapped_df.columns:
            mapped_df[feature] = default
    
    # Clean numeric columns - handle messy user input like "Abbastanza", "4h", "10/15"
    def clean_numeric(value, default):
        """Extract numeric value from messy input."""
        if pd.isna(value):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        
        # Convert to string and clean
        s = str(value).strip().lower()
        
        # Handle common text responses
        text_to_num = {
            "abbastanza": 7,  # "enough" in Italian
            "molto": 8,      # "a lot"
            "poco": 3,       # "little"
            "normale": 5,    # "normal"
            "high": 2,       # for work_pressure
            "medium": 1,
            "low": 0,
        }
        for text, num in text_to_num.items():
            if text in s:
                return num
        
        # Handle European number format (15.000 or 15,000 = 15000)
        # First, try to parse the whole string as a number
        import re
        
        # Remove common units and text
        s_cleaned = re.sub(r'[a-zA-Z%°]+', '', s).strip()
        
        # Handle European format: "15.000" or "15,000" meaning 15000
        # If it looks like "X.000" or "X,000", it's likely thousands
        european_match = re.match(r'^(\d{1,3})[.,](\d{3})$', s_cleaned)
        if european_match:
            # This is European thousands format (e.g., 15.000 = 15000)
            return float(european_match.group(1) + european_match.group(2))
        
        # Handle decimal numbers like "7.5" or "7,5"
        decimal_match = re.match(r'^(\d+)[.,](\d{1,2})$', s_cleaned)
        if decimal_match:
            # This is a decimal (e.g., 7.5 or 7,5)
            return float(decimal_match.group(1) + '.' + decimal_match.group(2))
        
        # Try to extract first number from string (handles "4h", "10/15", "120 mg", etc.)
        numbers = re.findall(r'\d+', s_cleaned)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        
        return default
    
    # Apply cleaning to all feature columns
    for feature in FEATURE_COLS:
        if feature in mapped_df.columns:
            default = DEFAULTS.get(feature, 5)
            mapped_df[feature] = mapped_df[feature].apply(lambda x: clean_numeric(x, default))
    
    # Convert work_pressure text to numeric
    if mapped_df["work_pressure"].dtype == object:
        pressure_map = {"low": 0, "medium": 1, "high": 2}
        mapped_df["work_pressure"] = mapped_df["work_pressure"].str.lower().map(pressure_map).fillna(1)
    
    # Detect if this is daily data (check for name/email column and multiple rows)
    is_daily = False
    group_col = None
    
    if "_name" in mapped_df.columns:
        group_col = "_name"
    elif "_email" in mapped_df.columns:
        group_col = "_email"
    
    if group_col and len(mapped_df) >= 7:
        # Check if any user has ~7 entries
        counts = mapped_df[group_col].value_counts()
        if counts.max() >= 7:
            is_daily = True
            mapped_df["_group"] = mapped_df[group_col]
    
    return mapped_df, is_daily


def get_interactive_input() -> Union[Dict[str, float], List[Dict[str, float]]]:
    """Get input interactively from the user (average or daily mode)."""
    print("\n" + "=" * 60)
    print("🧠 BURNOUT RISK ASSESSMENT")
    print("=" * 60)
    
    # Ask user which input mode they prefer
    print("\nHow would you like to enter your data?")
    print("  [1] Weekly AVERAGES (quick - ~10 questions)")
    print("  [2] DAILY data for each of the past 7 days (detailed - more accurate)")
    
    while True:
        mode_choice = input("\nChoose mode [1]: ").strip()
        if mode_choice in ["", "1"]:
            return _get_average_input()
        elif mode_choice == "2":
            return _get_daily_input()
        else:
            print("  ⚠️  Please enter 1 or 2")


def _get_average_input() -> Dict[str, float]:
    """Get weekly average input from user."""
    print("\n📊 WEEKLY AVERAGES MODE")
    print("Answer these questions about your PAST WEEK (average per day).")
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


def _get_daily_input() -> List[Dict[str, float]]:
    """Get daily data for each of the past 7 days."""
    print("\n📅 DAILY DATA MODE")
    print("Enter your data for each of the past 7 days.")
    print("Press Enter to use default value shown in [brackets].")
    print("Tip: Start with the oldest day (7 days ago) and work to today.\n")
    
    day_names = ["Day 1 (7 days ago)", "Day 2 (6 days ago)", "Day 3 (5 days ago)", 
                 "Day 4 (4 days ago)", "Day 5 (3 days ago)", "Day 6 (2 days ago)", 
                 "Day 7 (yesterday)"]
    
    # Simplified questions for daily entry (fewer questions to avoid tedium)
    daily_questions = [
        ("stress_level", "Stress level (1-10)", 1, 10),
        ("sleep_hours", "Sleep hours", 0, 12),
        ("work_hours", "Work hours", 0, 16),
        ("mood_score", "Mood (1-10)", 1, 10),
        ("exercise_minutes", "Exercise (minutes)", 0, 180),
    ]
    
    daily_data = []
    
    for day_idx, day_name in enumerate(day_names):
        print(f"\n{'─' * 40}")
        print(f"📆 {day_name}")
        print(f"{'─' * 40}")
        
        day_record = DEFAULTS.copy()  # Start with defaults
        
        for feature, question, min_val, max_val in daily_questions:
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
                    day_record[feature] = value
                    break
                except ValueError:
                    print("    ⚠️  Please enter a number")
        
        daily_data.append(day_record)
        
        # Allow user to copy previous day for similar days
        if day_idx < 6:
            copy_prev = input("\n  Copy these values to next day? [y/N]: ").strip().lower()
            if copy_prev == 'y':
                # Pre-fill defaults for next iteration
                for feature, _, _, _ in daily_questions:
                    DEFAULTS[feature] = day_record[feature]
    
    print(f"\n✅ Collected data for all 7 days!")
    return daily_data


def create_weekly_sequence(daily_data: Union[Dict[str, float], List[Dict[str, float]]], feature_cols: List[str], days: int = 7) -> np.ndarray:
    """
    Create a 7-day sequence from input data.
    
    Supports two input formats:
    1. Single dict (averages) - simulates daily variation
    2. List of 7 dicts (actual daily data) - uses real values
    
    Args:
        daily_data: Either a dict of averages or list of 7 daily dicts
        feature_cols: List of features expected by the model (in order!)
        days: Number of days in sequence
    """
    # If it's already a list of daily records, use them directly
    if isinstance(daily_data, list):
        if len(daily_data) != days:
            raise ValueError(f"Expected {days} days of data, got {len(daily_data)}")
        
        sequence = []
        for day_record in daily_data:
            day_data = []
            for feature in feature_cols:
                value = day_record.get(feature, DEFAULTS.get(feature, 5))
                day_data.append(float(value))
            sequence.append(day_data)
        
        return np.array(sequence, dtype=np.float32)
    
    # Otherwise, simulate daily variation from averages
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


def detect_contradictions(data: Dict[str, float]) -> List[str]:
    """
    Detect contradictions in self-reported data that might indicate
    denial, unawareness, or data entry errors.
    
    Returns a list of warning messages.
    """
    warnings = []
    
    # Extract all relevant metrics
    stress = data.get("stress_level", 5)
    work_hours = data.get("work_hours", 8)
    work_pressure = data.get("work_pressure", 1)
    sleep_hours = data.get("sleep_hours", 7)
    sleep_quality = data.get("sleep_quality", 6)
    mood = data.get("mood_score", 6)
    energy = data.get("energy_level", 6)
    caffeine = data.get("caffeine_mg", 100)
    exercise = data.get("exercise_minutes", 30)
    screen_time = data.get("screen_time_hours", 4)
    focus = data.get("focus_score", 6)
    alcohol = data.get("alcohol_units", 0)
    meetings = data.get("meetings_count", 3)
    
    # =====================================================================
    # STRESS & PRESSURE CONTRADICTIONS
    # =====================================================================
    
    # High stress but claims low work pressure
    if stress >= 7 and work_pressure == 0:
        warnings.append(
            "⚠️ You report HIGH stress ({}), but LOW work pressure. "
            "This contradiction may indicate you're underestimating workplace demands, "
            "or stress is coming from non-work sources.".format(int(stress))
        )
    
    # Long hours but claims low pressure
    if work_hours >= 10 and work_pressure == 0:
        warnings.append(
            "⚠️ You work {} hours/day but report LOW pressure. "
            "Long hours often lead to burnout even if they feel 'manageable' now. "
            "Consider if this is sustainable long-term.".format(int(work_hours))
        )
    
    # Many meetings but claims low pressure
    if meetings >= 6 and work_pressure == 0:
        warnings.append(
            "⚠️ {} meetings/day but LOW pressure? "
            "Meeting overload is a hidden stressor. Consider if they're all necessary.".format(int(meetings))
        )
    
    # =====================================================================
    # SLEEP CONTRADICTIONS
    # =====================================================================
    
    # Poor sleep but high energy (might be running on adrenaline/caffeine)
    if sleep_hours <= 5 and energy >= 7:
        warnings.append(
            "⚠️ Only {} hours of sleep but high energy ({}). "
            "You might be running on adrenaline or caffeine. "
            "This pattern often precedes a crash.".format(int(sleep_hours), int(energy))
        )
    
    # Poor sleep quality but long sleep hours
    if sleep_quality <= 3 and sleep_hours >= 8:
        warnings.append(
            "⚠️ You sleep {} hours but quality is only {}/10. "
            "Quantity without quality doesn't restore you. "
            "Consider sleep hygiene improvements.".format(int(sleep_hours), int(sleep_quality))
        )
    
    # High caffeine but claims good sleep (contradiction)
    if caffeine >= 300 and sleep_quality >= 7:
        warnings.append(
            "⚠️ High caffeine ({}mg) but good sleep quality ({}/10)? "
            "Caffeine has a 6-hour half-life and affects sleep architecture even if you fall asleep easily. "
            "You might not be getting the deep sleep you need.".format(int(caffeine), int(sleep_quality))
        )
    
    # High caffeine AND poor sleep (likely causal relationship)
    if caffeine >= 250 and sleep_quality <= 4:
        warnings.append(
            "⚠️ High caffeine ({}mg) with poor sleep quality ({}/10). "
            "This is likely not a coincidence! Caffeine blocks adenosine (your sleep signal) "
            "for 6+ hours. Try cutting caffeine after noon.".format(int(caffeine), int(sleep_quality))
        )
    
    # High caffeine AND low sleep hours
    if caffeine >= 250 and sleep_hours <= 5:
        warnings.append(
            "⚠️ High caffeine ({}mg) and only {} hours of sleep. "
            "You might be using caffeine to compensate for sleep deprivation. "
            "This creates a vicious cycle - caffeine disrupts sleep, poor sleep needs more caffeine.".format(int(caffeine), int(sleep_hours))
        )
    
    # High screen time + poor sleep quality
    if screen_time >= 6 and sleep_quality <= 4:
        warnings.append(
            "⚠️ High screen time ({}h) and poor sleep quality ({}/10). "
            "Blue light from screens suppresses melatonin. "
            "Try reducing screens 1-2 hours before bed.".format(int(screen_time), int(sleep_quality))
        )
    
    # Alcohol + claims good sleep
    if alcohol >= 3 and sleep_quality >= 7:
        warnings.append(
            "⚠️ {} alcohol units but good sleep quality ({}/10)? "
            "Alcohol disrupts REM sleep even if you feel you slept well. "
            "You may be missing restorative sleep stages.".format(int(alcohol), int(sleep_quality))
        )
    
    # =====================================================================
    # MOOD & ENERGY CONTRADICTIONS
    # =====================================================================
    
    # Low mood but claims low stress
    if mood <= 4 and stress <= 3:
        warnings.append(
            "⚠️ Low mood ({}) but low stress ({}). "
            "Low mood without recognized stress could indicate the early stages of burnout "
            "or depression. Consider checking in with how you're really feeling.".format(int(mood), int(stress))
        )
    
    # High mood but poor sleep (mood often crashes after sleep deprivation)
    if mood >= 8 and sleep_hours <= 5:
        warnings.append(
            "⚠️ Great mood ({}/10) but only {} hours of sleep. "
            "Mood often stays positive temporarily during sleep deprivation (adrenaline), "
            "then crashes. Make sure this isn't the calm before the storm.".format(int(mood), int(sleep_hours))
        )
    
    # High mood but very poor sleep quality
    if mood >= 8 and sleep_quality <= 3:
        warnings.append(
            "⚠️ Great mood ({}/10) but terrible sleep quality ({}/10). "
            "Poor sleep quality affects mood over time. "
            "You might be resilient now, but chronic poor sleep accumulates.".format(int(mood), int(sleep_quality))
        )
    
    # Low exercise but high energy
    if exercise <= 10 and energy >= 8:
        warnings.append(
            "⚠️ Minimal exercise ({}min) but high energy ({}/10). "
            "This energy might be from stimulants or hypervigilance rather than true vitality. "
            "Regular exercise builds sustainable energy.".format(int(exercise), int(energy))
        )
    
    # Low mood but high focus (hyperfocus can mask depression)
    if mood <= 3 and focus >= 8:
        warnings.append(
            "⚠️ Low mood ({}) but high focus ({}/10). "
            "Hyperfocus can be a way of avoiding difficult emotions. "
            "Make sure you're not using work to escape how you feel.".format(int(mood), int(focus))
        )
    
    # High energy but low mood (manic pattern or masking)
    if energy >= 8 and mood <= 4:
        warnings.append(
            "⚠️ High energy ({}) but low mood ({}/10). "
            "This disconnect between energy and mood can indicate anxiety, "
            "restlessness, or pushing through when you need rest.".format(int(energy), int(mood))
        )
    
    # =====================================================================
    # WORK PATTERN CONTRADICTIONS
    # =====================================================================
    
    # Very long hours + high focus (potential workaholism)
    if work_hours >= 12 and focus >= 8:
        warnings.append(
            "⚠️ Working {}h/day with high focus ({}/10). "
            "While productivity feels good, this pattern can mask workaholism. "
            "Are you working to avoid something else in life?".format(int(work_hours), int(focus))
        )
    
    # Short work hours but high pressure
    if work_hours <= 5 and work_pressure == 2:
        warnings.append(
            "⚠️ Only {} work hours but HIGH pressure? "
            "This suggests intense, demanding work. Quality matters as much as quantity. "
            "Even short bursts of high-pressure work can burn you out.".format(int(work_hours))
        )
    
    # =====================================================================
    # LIFESTYLE CONTRADICTIONS
    # =====================================================================
    
    # High caffeine + claims low stress
    if caffeine >= 400 and stress <= 3:
        warnings.append(
            "⚠️ Very high caffeine ({}mg) but low stress ({}/10)? "
            "Caffeine triggers cortisol (stress hormone). "
            "You might be chemically stressed without feeling it mentally.".format(int(caffeine), int(stress))
        )
    
    # No exercise but no stress
    if exercise <= 5 and stress <= 2 and work_hours >= 8:
        warnings.append(
            "⚠️ No exercise, full work hours, but minimal stress? "
            "Physical activity is a key stress regulator. "
            "Sedentary lifestyles accumulate stress in the body even when the mind feels calm.".format()
        )
    
    # Very high screen time + high focus
    if screen_time >= 8 and focus >= 7:
        warnings.append(
            "⚠️ {}h screen time with high focus ({}/10). "
            "Extended screen focus can feel productive but leads to digital fatigue. "
            "Your eyes and brain need breaks.".format(int(screen_time), int(focus))
        )
    
    # =====================================================================
    # CRITICAL WARNINGS (🚨)
    # =====================================================================
    
    # High stress + long hours + low sleep = classic burnout recipe
    if stress >= 7 and work_hours >= 10 and sleep_hours <= 6:
        warnings.append(
            "🚨 BURNOUT WARNING: High stress ({}), long hours ({}h), poor sleep ({}h) "
            "is a classic burnout recipe regardless of what the model predicts. "
            "Please take this seriously.".format(int(stress), int(work_hours), int(sleep_hours))
        )
    
    # Everything looks perfect (might be in denial or not being honest)
    perfect_count = sum([
        stress <= 2,
        mood >= 9,
        energy >= 9,
        sleep_quality >= 9,
        focus >= 9,
        work_pressure == 0,
    ])
    if perfect_count >= 5:
        warnings.append(
            "🤔 Your responses seem almost perfect across the board. "
            "If this truly reflects your life, that's wonderful! "
            "But if you're being optimistic, consider answering as your 'worst day this week' "
            "for a more realistic assessment."
        )
    
    # Low everything (possible depression or severe burnout)
    low_count = sum([
        mood <= 3,
        energy <= 3,
        focus <= 3,
        sleep_quality <= 3,
    ])
    if low_count >= 3:
        warnings.append(
            "🚨 Multiple very low scores (mood, energy, focus, or sleep). "
            "This pattern can indicate depression or severe burnout. "
            "Please consider talking to a mental health professional. "
            "You don't have to feel this way."
        )
    
    return warnings


def print_contradictions(data: Dict[str, float]) -> None:
    """Print any detected contradictions in the data."""
    warnings = detect_contradictions(data)
    
    if warnings:
        print("\n" + "─" * 50)
        print("🔍 DATA CONSISTENCY CHECK")
        print("─" * 50)
        print("\n  We noticed some patterns in your responses:\n")
        
        for warning in warnings:
            # Wrap text nicely
            print(f"  {warning}\n")


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
    """
    Generate specific, actionable advice for each feature.
    
    Based on:
    - Behavioral science research
    - Intervention data from our dataset (meditation, therapy, exercise_plan, 
      diet_coaching, workload_cap, vacation patterns)
    - Evidence-based wellness practices
    
    Each feature has multiple tips organized by severity/context.
    """
    import random
    
    # Comprehensive advice database - multiple tips per feature
    # Organized by: quick wins, lifestyle changes, and professional interventions
    advice_database = {
        # =====================================================================
        # SLEEP
        # =====================================================================
        "sleep_hours": {
            "quick": [
                f"Set a phone alarm labeled 'Wind Down' for {int(diff)}h before your target wake time.",
                f"Tonight: lights out {int(diff * 60)} minutes earlier. Just one night to start.",
                "Put your phone charger across the room. You'll have to get up to snooze.",
            ],
            "lifestyle": [
                f"Create a 'sleep runway': dim lights, no screens for {max(30, int(diff * 30))} min before bed.",
                "Weekends: don't sleep in more than 1h. It disrupts your circadian rhythm.",
                f"Your body needs ~{target:.0f}h. Sleep debt accumulates - you can't 'catch up' on weekends.",
            ],
            "intervention": [
                "📊 Data shows: people who improved sleep hours had 23% lower burnout risk after 2 weeks.",
                "Consider a sleep tracking app (Sleep Cycle, Pillow) to find your optimal bedtime.",
                "If insomnia persists >3 weeks, consider cognitive behavioral therapy for insomnia (CBT-I).",
            ],
        },
        "sleep_quality": {
            "quick": [
                "Tonight: no screens 1h before bed. Blue light blocks melatonin by up to 50%.",
                "Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s. Repeat 4x. Activates parasympathetic system.",
                "Keep bedroom at 65-68°F (18-20°C). Your body needs to cool down to sleep deeply.",
            ],
            "lifestyle": [
                "No caffeine after 2pm - it has a 6-hour half-life and fragments REM sleep.",
                "Alcohol may help you fall asleep, but it destroys sleep quality. Last drink 3h before bed.",
                "Exercise improves deep sleep, but finish workouts 3+ hours before bedtime.",
                "Same wake time every day (even weekends) is the #1 sleep quality factor.",
            ],
            "intervention": [
                "📊 Our meditation intervention data shows improved sleep quality in 6-8 weeks.",
                "Try magnesium glycinate (200-400mg) before bed - it supports GABA and relaxation.",
                f"Quality of {current:.0f}/10 is concerning. If ongoing, consider a sleep study for apnea.",
                "White noise or sleep sounds can improve quality by masking disruptive noises.",
            ],
        },
        # =====================================================================
        # STRESS
        # =====================================================================
        "stress_level": {
            "quick": [
                "Right now: Box breathing. 4s inhale → 4s hold → 4s exhale → 4s hold. Do 5 rounds.",
                "Stress hack: Hold ice cubes for 30 seconds. Physical sensation interrupts stress spiral.",
                f"Write down 3 things stressing you right now. Naming reduces amygdala activation by 30%.",
            ],
            "lifestyle": [
                "Daily 10-min meditation reduces cortisol by 23% within 8 weeks (Harvard study).",
                f"Your stress is {current:.0f}/10. Ask: 'What's one thing I can delegate or drop this week?'",
                "Nature exposure: 20 min in green space lowers cortisol more than urban walking.",
                "Set 'worry time': 15 min daily to write worries. Outside that, redirect thoughts.",
            ],
            "intervention": [
                "📊 Our therapy intervention data: participants showed stress reduction in 4-6 sessions.",
                "Consider an EAP (Employee Assistance Program) - most employers offer free counseling.",
                "Chronic stress >7/10 warrants professional support. Therapy isn't just for 'problems'.",
                "📊 Meditation interventions in our data: 48 people completed 2-4 week programs.",
            ],
        },
        "work_pressure": {
            "quick": [
                "Brain dump: Write every task on paper. Seeing it all reduces mental load.",
                "Ask your manager: 'I have A, B, C. Which is the real priority for this week?'",
                "One 'no' today. Pick the lowest-value request and decline or defer it.",
            ],
            "lifestyle": [
                "Batch similar tasks. Context-switching costs 23 minutes of refocus time each switch.",
                "Set 'office hours' for interruptions. Protect deep work blocks.",
                "End each day writing tomorrow's top 3 priorities. Morning clarity reduces pressure.",
            ],
            "intervention": [
                "📊 Our workload_cap intervention: 35 cases of negotiated reduced hours/expectations.",
                "Talk to your manager about workload. 'I want to do great work, but need to prioritize.'",
                "High pressure + long hours is a burnout recipe. Consider a workload conversation.",
            ],
        },
        # =====================================================================
        # WORK
        # =====================================================================
        "work_hours": {
            "quick": [
                f"Today: Set a hard stop. Leave {diff:.1f}h earlier. Put it in your calendar as a meeting.",
                "Close laptop at a set time. Create a 'shutdown ritual' - it signals work is done.",
                f"Working {current:.0f}h/day? That's {current*5:.0f}h/week. Productivity drops sharply after 50h.",
            ],
            "lifestyle": [
                f"Block your calendar after {18 - int(diff/2)}:00 this week. Treat it as non-negotiable.",
                "Track actual productive time vs. hours worked. Most people are only productive 4-5h/day.",
                "Parkinson's Law: Work expands to fill time. Set shorter deadlines for yourself.",
            ],
            "intervention": [
                "📊 Our workload_cap intervention: 35 participants had hours/expectations formally reduced.",
                "If >50h/week is expected, this is a systemic issue. Consider discussing with HR/manager.",
                "Long hours correlate with lower productivity per hour. Rest is a performance strategy.",
            ],
        },
        "meetings_count": {
            "quick": [
                f"Decline {max(1, int(diff))} meeting(s) this week. 'I'm focused on [project] this week.'",
                "For your next meeting: propose 25 min instead of 30. You'll get a 5-min buffer.",
                "Ask: 'Could this meeting be an email?' before accepting.",
            ],
            "lifestyle": [
                "Block 'focus time' on your calendar 9-11am daily. Protect it from meetings.",
                "No-meeting days: Try Maker Schedule - one day/week with zero meetings.",
                "Batch meetings to 2-3 days. Leave other days for deep work.",
            ],
            "intervention": [
                "Meeting overload is often cultural. Propose a team 'meeting audit' to reduce waste.",
                "If you're double-booked constantly, you're saying yes too much. Practice graceful nos.",
            ],
        },
        "emails_received": {
            "quick": [
                "Turn off email notifications. Check 3x/day at set times instead.",
                "Use 2-minute rule: if <2 min to respond, do it now. Otherwise, schedule it.",
                "Unsubscribe from 5 newsletters today. Reduce incoming volume permanently.",
            ],
            "lifestyle": [
                "No email before 10am. Start with deep work, not reactive inbox management.",
                "Set up filters: auto-sort newsletters, CCs, and low-priority senders.",
                "Write shorter emails to get shorter responses. Model the behavior you want.",
            ],
            "intervention": [
                "Email overload is a symptom. The root cause is often unclear priorities or boundaries.",
                "Consider discussing email expectations with your team - async doesn't mean instant.",
            ],
        },
        "commute_minutes": {
            "quick": [
                "Use commute for podcasts, audiobooks, or language learning. Make it productive.",
                "If possible, shift hours to avoid peak traffic. Even 30 min earlier can save time.",
            ],
            "lifestyle": [
                "Ask about hybrid/remote options. Even 2 WFH days saves significant commute stress.",
                "Longer commutes correlate with lower happiness. Consider housing closer to work.",
                "If driving, try audiobooks or call a friend. Social connection offsets commute stress.",
            ],
            "intervention": [
                "Commute >45 min/day is associated with higher stress and lower life satisfaction.",
                "This may warrant a larger life decision: move, change jobs, or negotiate remote.",
            ],
        },
        # =====================================================================
        # PHYSICAL HEALTH
        # =====================================================================
        "exercise_minutes": {
            "quick": [
                f"Today: {max(10, int(diff/2))}-min walk after lunch. Movement immediately boosts mood.",
                "Micro-workouts: 3x10 squats, push-ups, or stairs spread through the day.",
                "Take calls while walking. 30-min call = ~3,000 steps.",
            ],
            "lifestyle": [
                f"Add {int(diff)} min/day = {int(diff * 7)} extra minutes/week. Compounds to real change.",
                "Morning exercise is most consistent - willpower depletes throughout the day.",
                "Find exercise you enjoy. Consistency beats intensity every time.",
                "Exercise is the most effective antidepressant with zero side effects.",
            ],
            "intervention": [
                "📊 Our exercise_plan intervention: 51 participants did 2-8 week structured programs.",
                "Consider a fitness class, personal trainer, or exercise buddy for accountability.",
                "WHO recommends 150 min moderate activity/week. That's just 22 min/day.",
            ],
        },
        "steps_count": {
            "quick": [
                f"Add {int(diff)} steps today. A 15-min walk = ~1,500-2,000 steps.",
                "Walking meeting: Suggest your next 1:1 be a walk instead of sitting.",
                "Park farther away, take stairs, walk to a farther bathroom. Steps add up.",
            ],
            "lifestyle": [
                "Aim for 8,000-10,000 steps. Above 7,500, mortality risk drops significantly.",
                "Get a standing desk or treadmill desk. Movement throughout day > gym 1x.",
                "Post-meal walks improve blood sugar regulation. 10 min after eating helps.",
            ],
            "intervention": [
                "📊 Our exercise_plan data: step counts are a reliable predictor of overall health.",
                "Consider a step-tracking challenge with colleagues or friends for motivation.",
            ],
        },
        "caffeine_mg": {
            "quick": [
                f"You're at {current:.0f}mg. Skip one coffee today → save {min(100, diff):.0f}mg.",
                "Switch your afternoon coffee to decaf or half-caff. Same ritual, less stimulant.",
                "Caffeine has a 6-hour half-life. That 3pm coffee is still 50% active at 9pm.",
            ],
            "lifestyle": [
                "Max 400mg/day (FDA). Above that: anxiety, insomnia, and dependence risk.",
                "Try matcha or green tea: L-theanine smooths the caffeine edge.",
                f"At {current:.0f}mg, you may have tolerance. Consider a 1-week caffeine reset.",
            ],
            "intervention": [
                "📊 Our diet_coaching intervention: 51 participants worked on nutrition including caffeine.",
                "High caffeine often masks sleep debt. Fix sleep first, caffeine need drops naturally.",
                "Caffeine is a drug. If >500mg/day, consider gradual reduction to avoid withdrawal.",
            ],
        },
        "alcohol_units": {
            "quick": [
                "Alternate: one alcoholic drink, one water. Halves intake, prevents dehydration.",
                "Alcohol disrupts REM sleep. Even 2 drinks significantly reduce sleep quality.",
                "Try a mocktail or NA beer. You keep the ritual without the impact.",
            ],
            "lifestyle": [
                "2+ dry days per week. Your liver needs 48h to fully recover.",
                f"At {current:.0f} units, consider tracking with an app like Drinks Meter.",
                "Alcohol is a depressant. If using to cope with stress, that's a warning sign.",
            ],
            "intervention": [
                "📊 Regular alcohol use correlates with higher burnout risk in our data.",
                "If drinking to manage stress, consider speaking with a counselor about alternatives.",
                "More than 14 units/week (men) or 7 (women) is considered risky by health guidelines.",
            ],
        },
        # =====================================================================
        # MOOD & ENERGY
        # =====================================================================
        "mood_score": {
            "quick": [
                "Mood follows behavior. Plan ONE enjoyable activity in the next 2 hours.",
                "Text or call one friend right now. Social connection is the fastest mood boost.",
                "Go outside for 15 min. Sunlight regulates serotonin and improves mood.",
            ],
            "lifestyle": [
                "Gratitude practice: Write 3 good things each night. Rewires brain for positivity.",
                "Morning sunlight (10-30 min) sets circadian rhythm and improves mood all day.",
                "Exercise is as effective as antidepressants for mild-moderate depression.",
            ],
            "intervention": [
                "📊 Our therapy intervention: 56 participants, many for mood and anxiety support.",
                f"Mood at {current:.0f}/10 for multiple weeks warrants professional conversation.",
                "Consider talking to a therapist. Low mood is treatable, not a character flaw.",
                "📊 Our meditation intervention: 48 participants, meditation helps mood regulation.",
            ],
        },
        "energy_level": {
            "quick": [
                "Feeling tired? Try a 10-min walk instead of reaching for caffeine.",
                "Drink water now. Dehydration causes fatigue before you feel thirsty.",
                "Open a window or step outside. Fresh air and light boost alertness.",
            ],
            "lifestyle": [
                "Eat protein at lunch. Carb-heavy meals cause afternoon energy crashes.",
                "Energy follows sleep, exercise, and nutrition. Which is your weakest link?",
                "Iron, B12, or vitamin D deficiency? Low energy can have medical causes.",
            ],
            "intervention": [
                f"Energy at {current:.0f}/10 persistently? Consider bloodwork for thyroid, iron, B12, D.",
                "📊 Our exercise_plan intervention: exercise is the #1 natural energy booster.",
                "Chronic fatigue lasting >6 months should be evaluated by a doctor.",
            ],
        },
        "focus_score": {
            "quick": [
                "Pomodoro technique: 25 min focused work → 5 min break. Repeat. Use a timer.",
                "Put your phone in another room. Physical distance removes temptation.",
                "Close all browser tabs except what you need. Each tab is a distraction.",
            ],
            "lifestyle": [
                "Single-tasking > multitasking. Your brain can only truly focus on one thing.",
                "Block 'deep work' time on your calendar. Protect it like a meeting.",
                "Morning is typically best for focus. Do hard cognitive tasks before noon.",
                "Sleep impacts focus more than anything. Poor sleep = poor focus next day.",
            ],
            "intervention": [
                "📊 Our meditation intervention: 48 people trained attention through mindfulness.",
                "Consider apps like Freedom, Cold Turkey, or Screen Time to block distractions.",
                "Persistent focus issues may indicate ADHD or other conditions worth evaluating.",
            ],
        },
        # =====================================================================
        # SCREEN & TASKS
        # =====================================================================
        "screen_time_hours": {
            "quick": [
                f"Set a {target:.0f}h daily screen limit on your phone. iOS/Android have this built in.",
                "Grayscale mode after 8pm. Removing color reduces engagement by ~40%.",
                f"Replace {min(1, diff):.0f}h of scrolling with a walk, book, or conversation.",
            ],
            "lifestyle": [
                "No phones during meals. Reconnect with food and people.",
                "Delete social media apps. Access only via browser to add friction.",
                "Charge phone outside bedroom. Reduces nighttime scrolling.",
            ],
            "intervention": [
                "📊 Excessive screen time correlates with lower mood and higher stress in our data.",
                "Screen addiction is real. If you can't control it, consider digital detox programs.",
            ],
        },
        "tasks_completed": {
            "quick": [
                f"Aim for {target:.0f} tasks today. Break big tasks into smaller, completable chunks.",
                "Start with the hardest task (eat the frog). Momentum builds from there.",
                "If stuck, use the 2-minute rule: if <2 min, do it immediately.",
            ],
            "lifestyle": [
                "Plan tomorrow's 3 most important tasks tonight. Wake up with clarity.",
                "Batch similar tasks. Context-switching wastes 23 minutes each switch.",
                "Don't confuse busy with productive. Focus on high-impact tasks.",
            ],
            "intervention": [
                "Low task completion may indicate overwhelm, unclear priorities, or burnout.",
                "📊 Consider our workload_cap intervention pattern: reduce scope to increase output.",
            ],
        },
        # =====================================================================
        # NEW V2 FEATURES - SOCIAL, LIFESTYLE, & ENVIRONMENT
        # =====================================================================
        "social_interactions": {
            "quick": [
                f"Aim for {target:.0f} meaningful conversations today. Quality > quantity.",
                "Text a friend right now. 'Hey, how are you?' is enough to start.",
                "Take a coffee break WITH someone. Turn solo time into connection time.",
            ],
            "lifestyle": [
                "Schedule weekly calls with friends/family. Put them in your calendar like meetings.",
                "Join a club, class, or group activity. Regular structure builds relationships.",
                "Loneliness is as harmful as smoking 15 cigarettes/day. Prioritize connection.",
            ],
            "intervention": [
                "📊 Our data shows social isolation strongly correlates with burnout risk.",
                "If you're isolated, consider: therapy groups, meetups, volunteer work.",
                "Remote work without social effort leads to disconnection. Be intentional.",
            ],
        },
        "outdoor_time_minutes": {
            "quick": [
                f"Get outside for {max(15, int(diff))} minutes today. Even standing outside helps.",
                "Take your next call outdoors. Walk and talk or just stand in the sun.",
                "Eat lunch outside. Combine nourishment with nature.",
            ],
            "lifestyle": [
                "Morning sunlight (15-30 min) sets your circadian rhythm for the whole day.",
                "Nature exposure reduces cortisol more than urban environments.",
                "20 min in green space improves mood for 4+ hours. It's like a dose of medicine.",
            ],
            "intervention": [
                "📊 Our dataset tracks outdoor_time_minutes - it's a reliable stress predictor.",
                "Consider outdoor hobbies: walking, gardening, sports, photography.",
                "Vitamin D from sunlight affects mood. If indoors all day, consider supplements.",
            ],
        },
        "diet_quality": {
            "quick": [
                "Add one vegetable to your next meal. Small wins compound.",
                "Drink water before reaching for snacks. Thirst often feels like hunger.",
                "Skip the afternoon candy. Try nuts, fruit, or cheese instead.",
            ],
            "lifestyle": [
                "Meal prep on Sundays. Decisions are hard when you're tired and hungry.",
                "Protein at every meal stabilizes blood sugar and energy.",
                f"Your diet score is {current:.0f}/10. The goal is progress, not perfection.",
            ],
            "intervention": [
                "📊 Our diet_coaching intervention: 51 participants worked with nutritionists.",
                "Poor diet → low energy → poor performance → stress → worse diet. Break the cycle.",
                "Consider consulting a dietitian for personalized advice.",
            ],
        },
        "job_satisfaction": {
            "quick": [
                "Identify ONE thing that would make today better at work. Ask for it.",
                "Write down what you enjoyed about work this week. Even small things count.",
                "Have a conversation with someone who energizes you at work.",
            ],
            "lifestyle": [
                f"Job satisfaction at {current:.0f}/10 is concerning. What would need to change?",
                "Purpose matters more than perks. Are you doing meaningful work?",
                "Consider: Is this the right role, wrong company, or both?",
            ],
            "intervention": [
                "📊 Low job_satisfaction is one of the strongest burnout predictors in our data.",
                "Talk to your manager about what would improve your experience.",
                "If satisfaction is low for >6 months, it may be time to explore other options.",
            ],
        },
        "social_quality": {
            "quick": [
                "Quality beats quantity. Have ONE deep conversation this week.",
                "Put your phone away when with people. Full presence = quality.",
                "Ask deeper questions. 'How are you really doing?' vs 'How are you?'",
            ],
            "lifestyle": [
                "Schedule regular 1:1 time with people who energize you.",
                "Energy vampires exist. Reduce time with people who drain you.",
                "Vulnerability builds connection. Share something real.",
            ],
            "intervention": [
                f"Social quality at {current:.0f}/10 suggests loneliness risk.",
                "📊 Loneliness strongly predicts burnout in our data.",
                "Consider therapy if you struggle to connect. Attachment patterns are changeable.",
            ],
        },
        "loneliness_score": {
            "quick": [
                "Reach out to one person today. Even a text breaks isolation.",
                "Join an online community around a hobby or interest.",
                "Feeling lonely doesn't mean you're broken. It means you need connection.",
            ],
            "lifestyle": [
                "Regular, repeated contact builds relationships. Consistency > intensity.",
                "Volunteer. Helping others creates connection AND purpose.",
                "Consider a pet. Companionship from animals is real and measurable.",
            ],
            "intervention": [
                "📊 Loneliness is as harmful as 15 cigarettes/day. It's a health crisis.",
                "Therapy can help with isolation. It's a safe place to practice connection.",
                "If chronically lonely, consider group therapy or support groups.",
            ],
        },
        "environment_distractions": {
            "quick": [
                "Noise-canceling headphones are an investment in productivity.",
                "Put a 'focus time' sign on your desk. Train people to respect it.",
                "Move to a quiet spot for 2 hours. Environment shapes behavior.",
            ],
            "lifestyle": [
                "Open offices are productivity killers. Advocate for quiet spaces.",
                "If WFH, create a dedicated workspace. Separate work from living.",
                "Use 'do not disturb' modes on all devices during focus time.",
            ],
            "intervention": [
                "📊 Distractions correlate with stress and lower productivity in our data.",
                "If your environment is uncontrollable, consider negotiating WFH time.",
                "Talk to your manager about workspace improvements.",
            ],
        },
        "break_flexibility": {
            "quick": [
                "Take a 5-min break NOW. You're allowed to step away.",
                "Micro-breaks prevent burnout. 90 min focused, then 15 min rest.",
                "A walk to get water counts as a break. Build them into your routine.",
            ],
            "lifestyle": [
                "If breaks feel impossible, that's a red flag about your workload.",
                "Breaks improve productivity. Rest is not laziness, it's strategy.",
                "Ultradian rhythms: our brains work in 90-min cycles. Honor them.",
            ],
            "intervention": [
                "📊 No break flexibility correlates with higher burnout in our data.",
                "If you can't take breaks, this is a systemic issue to raise with management.",
                "Your body and brain need rest. Advocate for humane working conditions.",
            ],
        },
        "work_life_boundary": {
            "quick": [
                "Set a hard stop time today. Put it in your calendar.",
                "Create a 'shutdown ritual': close laptop, say 'done for today' out loud.",
                "Change clothes after work. Physical signals help mental transitions.",
            ],
            "lifestyle": [
                "Work in a dedicated space. Avoid working from bed or couch.",
                "Separate devices: work laptop closed = work is over.",
                "Clear boundaries protect relationships. Your family needs you present.",
            ],
            "intervention": [
                "📊 Blurred boundaries strongly predict burnout in our data.",
                "If work invades all hours, this is a culture problem to address.",
                "Consider discussing expectations with your manager.",
            ],
        },
        "after_hours_checking": {
            "quick": [
                "Turn off email notifications on your phone after 6pm.",
                "Delete Slack/Teams from your phone. Check only on laptop during work hours.",
                "'Urgent' emails can wait until morning 99% of the time.",
            ],
            "lifestyle": [
                "Set 'do not disturb' schedules on all devices automatically.",
                "Model the behavior you want. If you email at 10pm, others will too.",
                "Response time expectations: discuss with your team explicitly.",
            ],
            "intervention": [
                "📊 Constant availability destroys recovery and predicts burnout.",
                "If required to be on-call constantly, negotiate limits or compensation.",
                "This is a boundary issue. You are not your job.",
            ],
        },
        "recovery_ability": {
            "quick": [
                "Tonight: do something purely for enjoyment. Not productivity, not improvement.",
                "Recovery activity ideas: reading, bath, music, cooking, games, art.",
                "Rest is not earned. You need it regardless of productivity.",
            ],
            "lifestyle": [
                f"Recovery ability at {current:.0f}/10 is concerning. What's blocking rest?",
                "Sleep, exercise, social connection, and nature are the recovery pillars.",
                "Burnout = chronic recovery deficit. You're spending more than you're replenishing.",
            ],
            "intervention": [
                "📊 Poor recovery ability is a leading indicator of burnout in our data.",
                "📊 Our vacation intervention: 44 people took extended time off.",
                "If you can't recover, consider sick leave or vacation. It's not weakness.",
            ],
        },
    }
    
    # Select advice based on severity and randomness
    advice_list = advice_database.get(feature, {})
    
    if not advice_list:
        return f"Aim to {'reduce' if current > target else 'increase'} this to {target:.1f}."
    
    # Determine severity
    severity_ratio = abs(diff) / max(abs(current), abs(target), 1)
    
    # Higher severity → more likely to get intervention advice
    if severity_ratio > 0.5 and "intervention" in advice_list:
        pool = advice_list.get("intervention", []) + advice_list.get("lifestyle", [])
    elif severity_ratio > 0.25 and "lifestyle" in advice_list:
        pool = advice_list.get("lifestyle", []) + advice_list.get("quick", [])
    else:
        pool = advice_list.get("quick", []) + advice_list.get("lifestyle", [])
    
    return random.choice(pool) if pool else f"Aim to {'reduce' if current > target else 'increase'} to {target:.1f}."


# Global CVAE advisor (loaded once)
_cvae_advisor = None
_cvae_stats = None


def get_cvae_advisor():
    """Load CVAE advisor on first use (lazy loading)."""
    global _cvae_advisor, _cvae_stats
    
    if _cvae_advisor is None:
        result = load_cvae_advisor()
        if result is not None:
            _cvae_advisor, _cvae_stats = result
    
    return _cvae_advisor, _cvae_stats


def print_cvae_suggestions(data: Dict[str, float], pred_class: int, feature_cols: List[str]) -> None:
    """
    Print AI-generated behavioral suggestions from the CVAE Smart Advisor.
    
    This uses a generative model trained on 1.4M behavioral sequences to
    suggest what your week would look like with low burnout.
    
    IMPORTANT: We filter suggestions to only show BENEFICIAL changes that align
    with health science. The CVAE learns correlations, not causation - so we
    must filter out spurious suggestions like "receive more emails" or "walk less".
    """
    cvae, stats = get_cvae_advisor()
    
    if cvae is None:
        return  # CVAE not available
    
    if pred_class == 0:
        # Already low risk, no need for suggestions
        return
    
    # Define what direction is ACTUALLY healthy for each feature
    # True = higher is better, False = lower is better
    HEALTHY_DIRECTION = {
        "stress_level": False,      # Lower stress is better
        "sleep_hours": True,        # More sleep is better
        "sleep_quality": True,      # Better sleep quality is better
        "work_hours": False,        # Fewer work hours is better (up to a point)
        "exercise_minutes": True,   # More exercise is better
        "mood_score": True,         # Better mood is better
        "energy_level": True,       # More energy is better
        "focus_score": True,        # Better focus is better
        "caffeine_mg": False,       # Less caffeine is better
        "screen_time_hours": False, # Less screen time is better
        "meetings_count": False,    # Fewer meetings is better
        "alcohol_units": False,     # Less alcohol is better
        "steps_count": True,        # More steps is better
        "tasks_completed": True,    # More tasks completed is better
        "work_pressure": False,     # Lower pressure is better
        "commute_minutes": False,   # Shorter commute is better
        "emails_received": False,   # Fewer emails is better (less overload)
        # === NEW V2 FEATURES ===
        "social_interactions": True,    # More social connection is better
        "outdoor_time_minutes": True,   # More outdoor time is better
        "diet_quality": True,           # Better diet is better
        "job_satisfaction": True,       # Higher job satisfaction is better
        "social_quality": True,         # Better social quality is better
        "loneliness_score": False,      # Lower loneliness is better
        "environment_distractions": False,  # Fewer distractions is better
        "break_flexibility": True,      # More flexibility is better
        "work_life_boundary": True,     # Clearer boundaries is better
        "after_hours_checking": False,  # Less after-hours work is better
        "recovery_ability": True,       # Better recovery is better
    }
    
    try:
        # Use CVAE's feature columns (may differ from classification model)
        cvae_feature_cols = stats.get("feature_cols", FEATURE_COLS)
        
        # Create sequence from data using CVAE's feature columns
        sequence = create_weekly_sequence(data, cvae_feature_cols)
        
        # Convert to tensor
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        current_label = torch.tensor([pred_class])
        
        # Normalize if stats available
        if stats and stats.get("mean") is not None and stats.get("std") is not None:
            mean = torch.tensor(stats["mean"], dtype=torch.float32)
            std = torch.tensor(stats["std"], dtype=torch.float32)
            x_norm = (x - mean) / (std + 1e-8)
        else:
            x_norm = x
        
        # Generate "low burnout" counterfactual
        suggested = cvae.suggest_changes(x_norm, current_label, target_label=0)
        
        # Denormalize
        if stats and stats.get("mean") is not None and stats.get("std") is not None:
            suggested = suggested * (std + 1e-8) + mean
        
        # Average over the week
        suggested_avg = suggested.squeeze(0).mean(dim=0).numpy()
        
        # Compare with current and FILTER to only beneficial changes
        changes = []
        for i, feature in enumerate(cvae_feature_cols):
            current = data.get(feature, DEFAULTS.get(feature, 5))
            suggested_val = suggested_avg[i] if i < len(suggested_avg) else current
            
            diff = suggested_val - current
            
            # Skip tiny changes
            if abs(diff) < 0.5 and (abs(diff) / max(abs(current), 1)) < 0.1:
                continue
            
            # CRITICAL: Only show suggestions that align with health science!
            # The CVAE learns correlations, not causation - we must filter out
            # spurious suggestions like "receive more emails" or "walk less"
            higher_is_better = HEALTHY_DIRECTION.get(feature, True)
            
            # Check if the suggestion is in the healthy direction
            is_beneficial = (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better)
            
            if is_beneficial:
                changes.append({
                    "feature": feature,
                    "current": current,
                    "suggested": suggested_val,
                    "diff": diff,
                    "higher_is_better": higher_is_better,
                })
        
        # Sort by magnitude of change
        changes.sort(key=lambda x: abs(x["diff"]), reverse=True)
        
        if not changes:
            # No beneficial suggestions found
            return
        
        print("\n" + "─" * 50)
        print("🤖 AI-POWERED BEHAVIORAL SUGGESTIONS")
        print("\n  📊 Changes that could help:\n")
        
        # Show top 5 beneficial changes
        for change in changes[:5]:
            feature = change["feature"]
            current = change["current"]
            suggested = change["suggested"]
            diff = change["diff"]
            
            feature_label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
            arrow = "↗️" if diff > 0 else "↘️"
            
            print(f"    {arrow} {feature_label}: {current:.1f} → {suggested:.1f} ({diff:+.1f})")
        
        print("\n  💡 These suggestions are based on patterns from people")
        print("     with similar profiles who had LOW burnout risk.")
        
    except Exception as e:
        print(f"\n  ⚠️  Could not generate AI suggestions: {e}")


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
    
    # Check for contradictions in self-reported data
    print_contradictions(data)
    
    # Data-driven recommendations (if model available)
    if model is not None and feature_cols is not None:
        print_recommendations(data, pred_class, model, feature_cols)
    
    # AI-powered suggestions from CVAE (if available and not already low risk)
    if feature_cols is not None and pred_class > 0:
        print_cvae_suggestions(data, pred_class, feature_cols)
    
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
        mapped_df, is_daily = parse_google_form_csv(args.csv)
        
        if is_daily and "_group" in mapped_df.columns:
            # Process as daily data grouped by user
            print(f"📅 Detected DAILY data format (grouped by user)")
            for user_id, group in mapped_df.groupby("_group"):
                if len(group) < 7:
                    print(f"\n⚠️  User '{user_id}' has only {len(group)} days (need 7). Skipping...")
                    continue
                
                # Take the first 7 days
                group = group.head(7)
                daily_records = [row[FEATURE_COLS].to_dict() for _, row in group.iterrows()]
                
                sequence = create_weekly_sequence(daily_records, feature_cols)
                pred_class, probs = predict(model, sequence, feature_cols)
                
                # Calculate averages for display
                avg_data = {f: np.mean([d[f] for d in daily_records]) for f in feature_cols}
                
                print(f"\n{'='*60}")
                print(f"📊 Results for: {user_id}")
                print_prediction_result(avg_data, pred_class, probs, model, feature_cols)
        else:
            # Process as weekly averages (one row per person)
            print(f"📊 Detected WEEKLY AVERAGES format ({len(mapped_df)} responses)")
            for idx, row in mapped_df.iterrows():
                data = {f: row[f] for f in FEATURE_COLS if f in row}
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
    
    # For what-if analysis, we need average data (convert if daily)
    if isinstance(data, list):
        # Calculate averages from daily data for display and what-if
        avg_data = {}
        for feature in feature_cols:
            avg_data[feature] = np.mean([d.get(feature, DEFAULTS.get(feature, 5)) for d in data])
        display_data = avg_data
    else:
        display_data = data
    
    # Print result with model for what-if analysis
    print_prediction_result(display_data, pred_class, probs, model, feature_cols)


if __name__ == "__main__":
    main()
