"""
================================================================================
MAE-PRETRAINED CLASSIFIER FOR BURNOUT PREDICTION
================================================================================
This script fine-tunes a pre-trained MAE encoder for burnout classification.

TRANSFER LEARNING APPROACH:
---------------------------
1. Load pre-trained encoder from MAE (train_mae.py)
2. Freeze or fine-tune encoder weights
3. Add classification head on top
4. Train on labeled burnout data

WHY THIS WORKS:
---------------
The MAE encoder has already learned behavioral patterns like:
- "Poor sleep → low energy next day"
- "High stress + long commute → mood decline"

These learned representations make burnout classification easier because
the encoder already "understands" the temporal dynamics of behavior.

FINE-TUNING STRATEGIES:
-----------------------
1. Frozen encoder (--freeze-encoder): Only train classification head
   - Faster training, less overfitting
   - Good when labeled data is limited

2. Full fine-tuning (default): Train everything end-to-end
   - Better performance with enough data
   - Encoder adapts to burnout-specific patterns

Usage:
    # First, run MAE pre-training:
    python scripts/train_mae.py --epochs 50

    # Then fine-tune for classification:
    python scripts/train_mae_classifier.py --epochs 30
    python scripts/train_mae_classifier.py --freeze-encoder --epochs 50

Input:  models/saved/mae_encoder.pt (pre-trained encoder)
        data/processed/daily_with_burnout.parquet
Output: models/saved/mae_classifier.pt

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DIR = Path("data/processed")
DAILY_PATH = PROCESSED_DIR / "daily_with_burnout.parquet"
MODEL_DIR = Path("models/saved")
MAE_PATH = MODEL_DIR / "mae_encoder.pt"

# Same features as MAE pre-training
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


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune MAE Encoder for Burnout Classification")
    parser.add_argument("--window", type=int, default=7,
                        help="Sequence window size in days (default: 7)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of fine-tuning epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (lower for fine-tuning, default: 1e-4)")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder weights, only train classification head")
    parser.add_argument("--sample-users", type=float, default=0.1,
                        help="Fraction of users to include (default: 0.1 for CPU)")
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_sequences_with_labels(window: int, sample_users: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load sequences with burnout labels for supervised fine-tuning.
    """
    if not DAILY_PATH.exists():
        raise FileNotFoundError(
            "Run scripts/create_burnout_labels.py first"
        )
    
    daily = pd.read_parquet(DAILY_PATH)
    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)

    if daily["work_pressure"].dtype == object:
        pressure_map = {"low": 0, "medium": 1, "high": 2}
        daily["work_pressure"] = daily["work_pressure"].map(pressure_map).fillna(1).astype(np.float32)

    # User sampling
    if sample_users < 1.0:
        user_ids = daily["user_id"].unique()
        n_sample = max(3, int(len(user_ids) * sample_users))
        selected = np.random.choice(user_ids, n_sample, replace=False)
        daily = daily[daily["user_id"].isin(selected)]
        print(f"Sampled {len(selected):,} users ({sample_users*100:.0f}%)")

    sequences, labels = [], []
    for uid, group in daily.groupby("user_id"):
        feats = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        labs = group["burnout_level"].to_numpy(dtype=np.int64)
        
        if len(group) < window:
            continue
        
        for idx in range(window, len(group) + 1):
            seq = feats[idx - window: idx]
            label = labs[idx - 1]
            if not np.isnan(seq).any():
                sequences.append(seq)
                labels.append(label)
    
    return np.stack(sequences), np.array(labels)


# ============================================================================
# POSITIONAL ENCODING (same as MAE)
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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ============================================================================
# MAE CLASSIFIER MODEL
# ============================================================================

class MAEClassifier(nn.Module):
    """
    Burnout classifier using pre-trained MAE encoder.
    
    Architecture:
    1. Pre-trained encoder (from MAE)
    2. Mean pooling over sequence
    3. Classification head → 3 burnout classes
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Encoder components (will be loaded from pre-trained weights)
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
        """
        Forward pass for classification.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
            
        Returns:
            logits: Class logits (batch, num_classes)
        """
        # Encode sequence
        embeddings = self.input_proj(x)
        embeddings = self.pos_encoder(embeddings)
        encoded = self.transformer_encoder(embeddings)
        
        # Mean pooling over sequence
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
    
    def load_pretrained_encoder(self, mae_checkpoint: dict) -> None:
        """
        Load pre-trained encoder weights from MAE checkpoint.
        """
        # Map MAE encoder weights to this model
        mae_state = mae_checkpoint.get("model_state", mae_checkpoint)
        
        # Load input projection
        self.input_proj.load_state_dict({
            "weight": mae_state["input_proj.weight"],
            "bias": mae_state["input_proj.bias"],
        })
        
        # Load transformer encoder
        encoder_state = {
            k.replace("transformer_encoder.", ""): v 
            for k, v in mae_state.items() 
            if k.startswith("transformer_encoder.")
        }
        self.transformer_encoder.load_state_dict(encoder_state)
        
        print("✓ Loaded pre-trained encoder weights from MAE")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args: argparse.Namespace) -> None:
    """Main fine-tuning function."""
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Check for pre-trained MAE
    if not MAE_PATH.exists():
        raise FileNotFoundError(
            f"Pre-trained MAE not found at {MAE_PATH}\n"
            "Run: python scripts/train_mae.py --epochs 50"
        )
    
    # Load MAE checkpoint
    print(f"Loading pre-trained MAE from {MAE_PATH}...")
    mae_checkpoint = torch.load(MAE_PATH, map_location=DEVICE)
    
    # Get normalization from MAE
    norm_stats = mae_checkpoint.get("normalization", {})
    mean = np.array(norm_stats.get("mean", [0] * len(FEATURE_COLS)))
    std = np.array(norm_stats.get("std", [1] * len(FEATURE_COLS)))
    
    # Load sequences with labels
    print("Loading sequences with labels...")
    X, y = load_sequences_with_labels(args.window, args.sample_users)
    print(f"Total sequences: {len(X):,}")
    
    # Normalize using MAE's statistics (important for transfer learning!)
    X = (X - mean) / std
    print("Features normalized using MAE statistics")
    
    # Train/val split
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_val = X[idx[:split]], X[idx[split:]]
    y_train, y_val = y[idx[:split]], y[idx[split:]]
    
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    
    # Build model
    model = MAEClassifier(
        input_dim=mae_checkpoint["input_dim"],
        d_model=mae_checkpoint["d_model"],
        nhead=mae_checkpoint["nhead"],
        num_layers=mae_checkpoint["num_layers"],
        num_classes=3,
    ).to(DEVICE)
    
    # Load pre-trained encoder
    model.load_pretrained_encoder(mae_checkpoint)
    
    # Optionally freeze encoder
    if args.freeze_encoder:
        print("Freezing encoder weights (only training classifier head)")
        for param in model.input_proj.parameters():
            param.requires_grad = False
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.1%})")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0.0
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_losses = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(yb.numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro")
        
        # Log
        print(f"Epoch {epoch:3d}: loss={np.mean(train_losses):.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": mae_checkpoint["input_dim"],
                "d_model": mae_checkpoint["d_model"],
                "nhead": mae_checkpoint["nhead"],
                "num_layers": mae_checkpoint["num_layers"],
                "feature_cols": FEATURE_COLS,
                "normalization": norm_stats,
                "best_accuracy": best_acc,
                "best_f1": val_f1,
            }, MODEL_DIR / "mae_classifier.pt")
    
    print("=" * 60)
    print(f"Fine-tuning complete!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Saved model to: {MODEL_DIR / 'mae_classifier.pt'}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
