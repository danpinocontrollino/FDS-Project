"""
================================================================================
MULTI-TARGET MENTAL HEALTH PREDICTION MODEL
================================================================================
Predicts ALL 8 real mental health outcomes from 7 days of behavioral data.

TARGETS (all real self-reported data, no made-up scores!):
─────────────────────────────────────────────────────────────────────────────────
DAILY (predict next-day):
  - stress_level (1-9): Daily self-reported stress
  - mood_score (1-10): Daily mood rating
  - energy_level (1-10): Daily energy rating  
  - focus_score (1-10): Daily focus/concentration

WEEKLY (predict end-of-week survey):
  - perceived_stress_scale (10-40): PSS - Validated clinical instrument
  - anxiety_score (0-21): GAD-7 style anxiety measure
  - depression_score (0-27): PHQ-9 style depression measure
  - job_satisfaction (1-10): Work satisfaction rating

ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────────────
  Input: 7 days × 17 behavioral features
         ↓
    Shared Encoder (LSTM/Transformer)
         ↓
    ┌────┴────┐
    │ Shared  │
    │ Repr.   │
    └────┬────┘
         ↓
  ┌──────┼──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
Stress  Mood  Energy Focus  PSS   Anx   Dep   Job_Sat
Head   Head   Head   Head   Head  Head  Head   Head

Each head outputs: regression value + binary classification (at-risk)

USAGE:
    python scripts/train_mental_health.py --model lstm --epochs 30
    python scripts/train_mental_health.py --model transformer --epochs 50

Author: University Project - Mental Health Prediction
================================================================================
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
MODEL_DIR = Path("models/saved")

DAILY_PATH = PROCESSED_DIR / "daily_with_burnout.parquet"

# Behavioral features (INPUT) - what we observe
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

# Mental health targets (OUTPUT) - what we predict
DAILY_TARGETS = ["stress_level", "mood_score", "energy_level", "focus_score"]
WEEKLY_TARGETS = ["perceived_stress_scale", "anxiety_score", "depression_score", "job_satisfaction"]
ALL_TARGETS = DAILY_TARGETS + WEEKLY_TARGETS

# Binary thresholds for "at-risk" classification
# Based on clinical cutoffs where available
RISK_THRESHOLDS = {
    "stress_level": 6,           # High daily stress (≥6 out of 9)
    "mood_score": 4,             # Low mood (≤4 out of 10) - inverted
    "energy_level": 4,           # Low energy (≤4 out of 10) - inverted
    "focus_score": 4,            # Poor focus (≤4 out of 10) - inverted
    "perceived_stress_scale": 20, # PSS ≥20 = high stress (clinical cutoff)
    "anxiety_score": 6,          # Moderate anxiety (≥6)
    "depression_score": 15,      # Moderate depression (≥15)
    "job_satisfaction": 4,       # Low satisfaction (≤4) - inverted
}

# Which targets are "lower is worse" (need inverted threshold logic)
INVERTED_TARGETS = {"mood_score", "energy_level", "focus_score", "job_satisfaction"}


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-target mental health model")
    parser.add_argument("--model", choices=["lstm", "gru", "transformer"], default="lstm")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=7, help="Days of input sequence")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--sample-users", type=float, default=1.0, help="Fraction of users (for testing)")
    parser.add_argument("--task", choices=["regression", "classification", "both"], default="both",
                        help="Predict continuous values, binary risk, or both")
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(window: int = 7, sample_users: float = 1.0) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load behavioral sequences and ALL mental health targets.
    
    Returns:
        X: Behavioral sequences (N × window × features)
        targets: Dict mapping target name → labels array (N,)
    """
    # Load daily data
    if not DAILY_PATH.exists():
        raise FileNotFoundError("Run scripts/create_burnout_labels.py first")
    
    daily = pd.read_parquet(DAILY_PATH)
    
    # Load weekly data for weekly targets
    weekly = pd.read_csv(RAW_DIR / "weekly_summaries.csv")
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    
    # Add week number to daily for merging
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["year"] = daily["date"].dt.year
    
    weekly["week"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["year"] = weekly["week_start"].dt.year
    
    # Merge weekly targets into daily
    daily = daily.merge(
        weekly[["user_id", "week", "year"] + WEEKLY_TARGETS],
        on=["user_id", "week", "year"],
        how="left"
    )
    
    # Sort by user and date
    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)
    
    # Convert work_pressure to numeric
    if daily["work_pressure"].dtype == object:
        pressure_map = {"low": 0, "medium": 1, "high": 2}
        daily["work_pressure"] = daily["work_pressure"].map(pressure_map).fillna(1).astype(np.float32)
    
    # Sample users if needed
    if sample_users < 1.0:
        all_users = daily["user_id"].unique()
        n_sample = max(10, int(len(all_users) * sample_users))
        selected = np.random.choice(all_users, n_sample, replace=False)
        daily = daily[daily["user_id"].isin(selected)]
    
    # Build sequences
    sequences = []
    target_values = {t: [] for t in ALL_TARGETS}
    
    for user_id, group in daily.groupby("user_id"):
        group = group.sort_values("date").reset_index(drop=True)
        
        # Need at least window + 1 days (window for input, +1 for target)
        if len(group) < window + 1:
            continue
        
        feats = group[FEATURE_COLS].values.astype(np.float32)
        
        # Create sliding windows
        for i in range(len(group) - window):
            seq = feats[i:i + window]
            sequences.append(seq)
            
            # Get targets from the day AFTER the window (next-day prediction)
            target_row = group.iloc[i + window]
            
            for target in ALL_TARGETS:
                val = target_row.get(target, np.nan)
                target_values[target].append(val if pd.notna(val) else np.nan)
    
    X = np.array(sequences, dtype=np.float32)
    targets = {t: np.array(v, dtype=np.float32) for t, v in target_values.items()}
    
    print(f"Loaded {len(X)} sequences from {daily['user_id'].nunique()} users")
    print(f"Shape: X={X.shape}")
    for t in ALL_TARGETS:
        valid = ~np.isnan(targets[t])
        print(f"  {t}: {valid.sum()} valid samples, mean={np.nanmean(targets[t]):.2f}")
    
    return X, targets


def normalize_features(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Normalize features using training set statistics."""
    n_train, seq_len, n_features = X_train.shape
    n_val = X_val.shape[0]
    
    # Flatten for scaling
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_train, seq_len, n_features)
    X_val_scaled = scaler.transform(X_val_flat).reshape(n_val, seq_len, n_features)
    
    return X_train_scaled, X_val_scaled, scaler


def create_binary_targets(targets: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert continuous targets to binary at-risk labels."""
    binary = {}
    for target, values in targets.items():
        threshold = RISK_THRESHOLDS[target]
        if target in INVERTED_TARGETS:
            # Lower is worse
            binary[target] = (values <= threshold).astype(np.float32)
        else:
            # Higher is worse
            binary[target] = (values >= threshold).astype(np.float32)
        # Preserve NaN
        binary[target][np.isnan(values)] = np.nan
    return binary


# ============================================================================
# MODEL ARCHITECTURE - Multi-Head Mental Health Predictor
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
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
    """Single prediction head for one target (regression + classification)."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.regression = nn.Linear(hidden_dim, 1)  # Continuous value
        self.classification = nn.Linear(hidden_dim, 1)  # Binary logit
    
    def forward(self, x):
        h = self.shared(x)
        reg_out = self.regression(h).squeeze(-1)
        cls_out = self.classification(h).squeeze(-1)
        return reg_out, cls_out


class MentalHealthPredictor(nn.Module):
    """
    Multi-target mental health prediction model.
    
    Shared encoder + 8 prediction heads (one per mental health outcome).
    Each head predicts both regression value and binary at-risk probability.
    """
    
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
        
        # Shared encoder
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
        
        # Shared representation layer
        self.shared_repr = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # One prediction head per target
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
    
    def forward(self, x: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            Dict mapping target name → (regression_pred, classification_logit)
        """
        # Encode
        h = self.encode(x)
        h = self.shared_repr(h)
        
        # Predict all targets
        outputs = {}
        for target in self.targets:
            reg, cls = self.heads[target](h)
            outputs[target] = (reg, cls)
        
        return outputs


# ============================================================================
# LOSS FUNCTION - Multi-Task with Uncertainty Weighting
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable uncertainty weighting.
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)
    Automatically balances regression and classification losses across targets.
    """
    
    def __init__(self, targets: List[str], task: str = "both"):
        super().__init__()
        self.targets = targets
        self.task = task
        
        # Learnable log-variance for each task (uncertainty weighting)
        # Higher uncertainty = lower weight
        if task in ["regression", "both"]:
            self.log_vars_reg = nn.ParameterDict({
                t: nn.Parameter(torch.zeros(1)) for t in targets
            })
        if task in ["classification", "both"]:
            self.log_vars_cls = nn.ParameterDict({
                t: nn.Parameter(torch.zeros(1)) for t in targets
            })
    
    def forward(
        self,
        predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        reg_targets: Dict[str, torch.Tensor],
        cls_targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted multi-task loss.
        
        Returns:
            total_loss: Combined loss for backprop
            loss_dict: Individual losses for logging
        """
        total_loss = 0
        loss_dict = {}
        
        for target in self.targets:
            reg_pred, cls_pred = predictions[target]
            
            # Get targets and mask NaN
            reg_true = reg_targets[target]
            cls_true = cls_targets[target]
            
            valid_mask = ~torch.isnan(reg_true)
            if valid_mask.sum() == 0:
                continue
            
            # Regression loss (MSE)
            if self.task in ["regression", "both"]:
                reg_loss = F.mse_loss(reg_pred[valid_mask], reg_true[valid_mask])
                # Uncertainty weighting: loss / (2 * var) + log(var)
                precision = torch.exp(-self.log_vars_reg[target])
                weighted_reg = precision * reg_loss + self.log_vars_reg[target]
                total_loss = total_loss + weighted_reg
                loss_dict[f"{target}_reg"] = reg_loss.item()
            
            # Classification loss (BCE)
            if self.task in ["classification", "both"]:
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_pred[valid_mask], cls_true[valid_mask]
                )
                precision = torch.exp(-self.log_vars_cls[target])
                weighted_cls = precision * cls_loss + self.log_vars_cls[target]
                total_loss = total_loss + weighted_cls
                loss_dict[f"{target}_cls"] = cls_loss.item()
        
        return total_loss, loss_dict


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {args.model.upper()}, Task: {args.task}")
    print(f"Targets: {ALL_TARGETS}")
    
    # Load data
    X, targets = load_data(window=args.window, sample_users=args.sample_users)
    binary_targets = create_binary_targets(targets)
    
    # Train/val split (80/20)
    n = len(X)
    split = int(0.8 * n)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    
    # Normalize features
    X_train, X_val, scaler = normalize_features(X_train, X_val)
    
    # Prepare target tensors
    def make_target_tensors(idx):
        reg = {t: torch.tensor(targets[t][idx], dtype=torch.float32) for t in ALL_TARGETS}
        cls = {t: torch.tensor(binary_targets[t][idx], dtype=torch.float32) for t in ALL_TARGETS}
        return reg, cls
    
    reg_train, cls_train = make_target_tensors(train_idx)
    reg_val, cls_val = make_target_tensors(val_idx)
    
    # Create data loaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, *[reg_train[t] for t in ALL_TARGETS], *[cls_train[t] for t in ALL_TARGETS]),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, *[reg_val[t] for t in ALL_TARGETS], *[cls_val[t] for t in ALL_TARGETS]),
        batch_size=args.batch_size * 2, shuffle=False
    )
    
    # Build model
    model = MentalHealthPredictor(
        input_dim=len(FEATURE_COLS),
        hidden_dim=args.hidden_dim,
        num_layers=2,
        encoder_type=args.model,
        targets=ALL_TARGETS,
    ).to(device)
    
    # Loss and optimizer
    criterion = MultiTaskLoss(ALL_TARGETS, task=args.task).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float("inf")
    model_path = MODEL_DIR / f"mental_health_{args.model}.pt"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_losses = []
        
        for batch in train_loader:
            x = batch[0].to(device)
            reg_targets_batch = {t: batch[1 + i].to(device) for i, t in enumerate(ALL_TARGETS)}
            cls_targets_batch = {t: batch[1 + len(ALL_TARGETS) + i].to(device) for i, t in enumerate(ALL_TARGETS)}
            
            optimizer.zero_grad()
            predictions = model(x)
            loss, _ = criterion(predictions, reg_targets_batch, cls_targets_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_losses = []
        all_preds = {t: {"reg": [], "cls": []} for t in ALL_TARGETS}
        all_true = {t: {"reg": [], "cls": []} for t in ALL_TARGETS}
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                reg_targets_batch = {t: batch[1 + i].to(device) for i, t in enumerate(ALL_TARGETS)}
                cls_targets_batch = {t: batch[1 + len(ALL_TARGETS) + i].to(device) for i, t in enumerate(ALL_TARGETS)}
                
                predictions = model(x)
                loss, loss_dict = criterion(predictions, reg_targets_batch, cls_targets_batch)
                val_losses.append(loss.item())
                
                # Collect predictions
                for t in ALL_TARGETS:
                    reg_pred, cls_pred = predictions[t]
                    all_preds[t]["reg"].extend(reg_pred.cpu().numpy())
                    all_preds[t]["cls"].extend(torch.sigmoid(cls_pred).cpu().numpy())
                    all_true[t]["reg"].extend(reg_targets_batch[t].cpu().numpy())
                    all_true[t]["cls"].extend(cls_targets_batch[t].cpu().numpy())
        
        val_loss = np.mean(val_losses)
        
        # Log progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{args.epochs} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {val_loss:.4f}")
            
            # Show metrics for each target
            for t in ALL_TARGETS:
                reg_true = np.array(all_true[t]["reg"])
                reg_pred = np.array(all_preds[t]["reg"])
                cls_true = np.array(all_true[t]["cls"])
                cls_pred = np.array(all_preds[t]["cls"])
                
                valid = ~np.isnan(reg_true)
                if valid.sum() > 0:
                    mae = mean_absolute_error(reg_true[valid], reg_pred[valid])
                    acc = accuracy_score(cls_true[valid] > 0.5, cls_pred[valid] > 0.5)
                    print(f"  {t:25s}: MAE={mae:.2f}, Acc={acc:.1%}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "model_type": args.model,
                "targets": ALL_TARGETS,
                "feature_cols": FEATURE_COLS,
                "risk_thresholds": RISK_THRESHOLDS,
                "scaler_mean": scaler.mean_,
                "scaler_std": scaler.scale_,
                "hidden_dim": args.hidden_dim,
                "window": args.window,
                "task": args.task,
            }, model_path)
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_path}")
    
    print(f"\n📊 Per-Target Performance:")
    print(f"{'Target':<25} {'MAE':>8} {'R²':>8} {'Accuracy':>10} {'Threshold':>10}")
    print("-" * 65)
    
    for t in ALL_TARGETS:
        reg_true = np.array(all_true[t]["reg"])
        reg_pred = np.array(all_preds[t]["reg"])
        cls_true = np.array(all_true[t]["cls"])
        cls_pred = np.array(all_preds[t]["cls"])
        
        valid = ~np.isnan(reg_true)
        if valid.sum() > 0:
            mae = mean_absolute_error(reg_true[valid], reg_pred[valid])
            r2 = r2_score(reg_true[valid], reg_pred[valid])
            acc = accuracy_score(cls_true[valid] > 0.5, cls_pred[valid] > 0.5)
            threshold = RISK_THRESHOLDS[t]
            direction = "≤" if t in INVERTED_TARGETS else "≥"
            print(f"{t:<25} {mae:>8.2f} {r2:>8.2f} {acc:>10.1%} {direction}{threshold:>9}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
