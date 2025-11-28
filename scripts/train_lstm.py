"""
================================================================================
TRAIN SEQUENCE MODELS (LSTM / GRU / CNN)
================================================================================
This script trains recurrent and convolutional neural networks to classify
burnout levels from 7-day behavioral time series windows.

ARCHITECTURE OPTIONS:
---------------------
1. LSTM (Long Short-Term Memory):
   - Learns long-term dependencies in sequential data
   - Has "gates" to control information flow (forget, input, output)
   - Best when: temporal patterns span multiple days

2. GRU (Gated Recurrent Unit):
   - Simplified LSTM with fewer parameters
   - Often performs similarly to LSTM but trains faster
   - Best when: dataset is smaller or faster training is needed

3. CNN (1D Convolutional Network):
   - Learns local patterns with sliding filters
   - Parallelizable (faster than RNNs on GPU)
   - Best when: local patterns (2-3 day trends) are important

WHY SEQUENCE MODELS?
-------------------
Burnout develops over time. A single day's data might not capture the trend.
By looking at 7 consecutive days, we can detect:
- Declining sleep quality over the week
- Increasing stress accumulation
- Work-life balance deterioration patterns

STRATIFIED USER SAMPLING:
------------------------
The full dataset has 1000 users × 365 days ≈ 1.47M sequences.
For CPU training, we support sampling a fraction of USERS (not sequences)
to preserve complete behavioral patterns per user.

Usage:
    python scripts/train_lstm.py --model lstm --window 7 --epochs 40
    python scripts/train_lstm.py --model gru --window 7 --epochs 40
    python scripts/train_lstm.py --model cnn --window 7 --epochs 40
    python scripts/train_lstm.py --model lstm --sample-users 0.1  # 10% of users

For Transformer models, use scripts/train_transformer.py instead.

Input:  data/processed/daily_with_burnout.parquet (from create_burnout_labels.py)
Output: models/saved/{model}_sequence.pt

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DIR = Path("data/processed")
DAILY_PATH = PROCESSED_DIR / "daily_with_burnout.parquet"
MODEL_DIR = Path("models/saved")

# Daily behavioral features used for sequence modeling
# These capture the key dimensions of work-life balance:
FEATURE_COLS = [
    # Sleep metrics (recovery indicators)
    "sleep_hours",           # Total sleep duration
    "sleep_quality",         # Subjective sleep quality (1-10)
    
    # Work metrics (stress sources)
    "work_hours",            # Hours worked that day
    "meetings_count",        # Number of meetings (interruption proxy)
    "tasks_completed",       # Productivity metric
    
    # Physical health metrics
    "exercise_minutes",      # Physical activity
    "steps_count",           # Daily movement
    
    # Consumption metrics (coping behaviors)
    "caffeine_mg",           # Stimulant consumption
    "alcohol_units",         # Depressant consumption
    "screen_time_hours",     # Digital exposure
    
    # Psychological metrics (outcome indicators)
    "stress_level",          # Self-reported stress (1-10)
    "mood_score",            # Emotional state (1-10)
    "energy_level",          # Fatigue indicator (1-10)
    "focus_score",           # Cognitive performance (1-10)
    
    # Work environment (categorical -> numeric)
    "work_pressure",         # Converted: low=0, medium=1, high=2
]


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for model training configuration.
    
    Key arguments:
    - --model: Architecture type (lstm, gru, cnn)
    - --window: Sequence length in days (default 7 = weekly patterns)
    - --sample-users: Fraction of users for CPU optimization
    """
    parser = argparse.ArgumentParser(description="Train burnout sequence models")
    parser.add_argument("--model", choices=["lstm", "gru", "cnn"], default="lstm",
                        help="Model architecture: lstm, gru, or cnn")
    parser.add_argument("--window", type=int, default=7,
                        help="Sequence window size in days (default: 7)")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=512, 
                        help="Mini-batch size (larger = faster on CPU)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--sample-users", type=float, default=1.0,
                        help="Fraction of USERS to include (preserves complete user patterns)")
    return parser.parse_args()


# ============================================================================
# DATA LOADING & SEQUENCE CREATION
# ============================================================================

def load_daily(window: int, sample_users: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load daily data and create sliding window sequences.
    
    Process:
    1. Load daily records (1000 users × 365 days)
    2. Optional: Stratified user sampling for CPU optimization
    3. Create sliding windows: each 7-day window → 1 sequence
    4. Label = burnout level of the LAST day in the window
    
    Args:
        window: Number of days in each sequence
        sample_users: Fraction of users to include (for CPU training)
        
    Returns:
        X: Sequences array (N × window × features)
        y: Labels array (N,)
        
    Example:
        For user with 365 days and window=7:
        - Sequence 1: days 1-7, label = day 7's burnout
        - Sequence 2: days 2-8, label = day 8's burnout
        - ...
        - Sequence 359: days 359-365, label = day 365's burnout
    """
    if not DAILY_PATH.exists():
        raise FileNotFoundError(
            "Run scripts/create_burnout_labels.py first to generate daily_with_burnout.parquet"
        )
    
    # Load and sort by user and date for proper sequence creation
    daily = pd.read_parquet(DAILY_PATH)
    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)

    # Convert categorical work_pressure to numeric
    # Original values: "low", "medium", "high"
    # Numeric encoding: 0, 1, 2 (ordinal - preserves order)
    if daily["work_pressure"].dtype == object:
        pressure_map = {"low": 0, "medium": 1, "high": 2}
        daily["work_pressure"] = daily["work_pressure"].map(pressure_map).fillna(1).astype(np.float32)

    # ========== STRATIFIED USER SAMPLING ==========
    # For CPU training optimization: sample a fraction of users while
    # preserving the population's burnout class distribution.
    if sample_users < 1.0:
        # Step 1: Determine each user's "dominant" burnout level (mode)
        # This represents the user's typical state
        user_burnout = daily.groupby("user_id")["burnout_level"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median()
        )
        user_burnout = user_burnout.reset_index()
        user_burnout.columns = ["user_id", "dominant_burnout"]
        
        n_users_total = len(user_burnout)
        n_users_sample = max(3, int(n_users_total * sample_users))  # at least 3 users
        
        # Step 2: Sample users proportionally from each burnout stratum
        # This ensures we don't accidentally sample only "Low burnout" users
        selected_users = []
        for level in sorted(user_burnout["dominant_burnout"].unique()):
            level_users = user_burnout[user_burnout["dominant_burnout"] == level]["user_id"].values
            # Proportion of this level in population
            level_proportion = len(level_users) / n_users_total
            n_from_level = max(1, int(n_users_sample * level_proportion))
            sampled = np.random.choice(level_users, min(n_from_level, len(level_users)), replace=False)
            selected_users.extend(sampled)
        
        daily = daily[daily["user_id"].isin(selected_users)]
        print(f"Stratified sampling: {len(selected_users):,} users ({sample_users*100:.0f}% of {n_users_total:,})")
        print(f"  Class distribution preserved: {daily['burnout_level'].value_counts(normalize=True).round(3).to_dict()}")

    # ========== SLIDING WINDOW SEQUENCE CREATION ==========
    sequences, labels = [], []
    
    for uid, group in daily.groupby("user_id"):
        feats = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        labs = group["burnout_level"].to_numpy(dtype=np.int64)
        
        # Skip users with insufficient data for even one window
        if len(group) < window:
            continue
        
        # Create sliding windows
        for idx in range(window, len(group) + 1):
            seq = feats[idx - window: idx]  # 7-day feature window
            label = labs[idx - 1]            # Last day's burnout level
            
            # Skip windows with missing values
            if np.isnan(seq).any():
                continue
                
            sequences.append(seq)
            labels.append(label)
    
    return np.stack(sequences), np.array(labels)


# ============================================================================
# DATALOADER CONSTRUCTION
# ============================================================================

def build_loaders(X: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[DataLoader, DataLoader, np.ndarray]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Uses random 80/20 split with shuffling (unlike MLP's temporal split).
    Shuffling is OK here because sequences are independent after creation.
    
    Args:
        X: Sequences array (N × window × features)
        y: Labels array (N,)
        batch_size: Mini-batch size
        
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        y_val: Validation labels for metric computation
    """
    # Random permutation for train/val split
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Wrap in TensorDataset
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    # Create DataLoaders
    # num_workers=0 avoids multiprocessing issues on some systems
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, y_val


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def build_model(model_type: str, input_dim: int, seq_len: int, num_classes: int) -> nn.Module:
    """
    Build the specified sequence model architecture.
    
    Args:
        model_type: "lstm", "gru", or "cnn"
        input_dim: Number of features per timestep
        seq_len: Sequence length (window size)
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    
    # ========== LSTM / GRU ==========
    if model_type in {"lstm", "gru"}:
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.GRU
        
        class SequenceNet(nn.Module):
            """
            Recurrent neural network for sequence classification.
            
            Architecture:
                Input (batch, seq_len, features)
                    -> 2-layer LSTM/GRU (128 hidden units)
                    -> Take last hidden state
                    -> LayerNorm -> ReLU -> Dropout
                    -> Linear(num_classes)
            """
            def __init__(self) -> None:
                super().__init__()
                # 2-layer RNN with dropout between layers
                self.rnn = rnn_cls(
                    input_size=input_dim,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True,  # Input shape: (batch, seq, features)
                    dropout=0.2        # Dropout between RNN layers
                )
                # Classification head
                self.head = nn.Sequential(
                    nn.LayerNorm(128),    # Normalize RNN output
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, seq_len, input_dim)
                out, _ = self.rnn(x)       # out: (batch, seq_len, 128)
                last_hidden = out[:, -1, :]  # Take last timestep: (batch, 128)
                return self.head(last_hidden)
        
        return SequenceNet()
    
    # ========== 1D CNN ==========
    if model_type == "cnn":
        
        class CNN1D(nn.Module):
            """
            1D Convolutional network for sequence classification.
            
            Architecture:
                Input (batch, seq_len, features) -> transpose -> (batch, features, seq_len)
                    -> Conv1D(64 filters, kernel=3)
                    -> ReLU -> BatchNorm
                    -> Conv1D(128 filters, kernel=3)
                    -> ReLU -> Global Average Pooling
                    -> Dropout -> Linear(num_classes)
            
            The CNN learns local temporal patterns (2-3 day windows)
            through the convolutional filters.
            """
            def __init__(self) -> None:
                super().__init__()
                # Convolutional layers
                # Kernel size 3 = looks at 3 consecutive days
                self.conv = nn.Sequential(
                    nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),  # Global average pooling
                )
                # Classification head
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, seq_len, features)
                # Conv1d expects: (batch, channels, seq_len)
                x = x.transpose(1, 2)  # -> (batch, features, seq_len)
                feats = self.conv(x)   # -> (batch, 128, 1)
                return self.head(feats)
        
        return CNN1D()
    
    raise ValueError(
        f"Unknown model type: {model_type}. "
        "Use 'lstm', 'gru', or 'cnn'. For transformer, use train_transformer.py"
    )


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args: argparse.Namespace) -> None:
    """
    Main training function.
    
    Implements standard PyTorch training loop with:
    - CrossEntropy loss for multi-class classification
    - Adam optimizer
    - Model checkpointing (save best validation loss)
    """
    # Select device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Load sequences
    seq_X, seq_y = load_daily(args.window, args.sample_users)
    print(f"Total sequences: {len(seq_X):,}")
    
    # Build data loaders
    train_loader, val_loader, y_val = build_loaders(seq_X, seq_y, args.batch_size)
    num_classes = len(np.unique(seq_y))
    
    # Build model
    model = build_model(args.model, seq_X.shape[2], seq_X.shape[1], num_classes).to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Tracking
    best_val = float("inf")
    history = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{args.model}_sequence.pt"

    # ========== TRAINING LOOP ==========
    for epoch in range(1, args.epochs + 1):
        
        # ----- Training phase -----
        model.train()
        train_losses = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()           # Clear gradients
            loss = criterion(model(xb), yb)  # Forward + loss
            loss.backward()                  # Backward
            optimizer.step()                 # Update weights
            
            train_losses.append(loss.item())
        
        # ----- Validation phase -----
        model.eval()
        val_losses = []
        preds = []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        
        val_loss = float(np.mean(val_losses))
        history.append(val_loss)
        
        # Log progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: val_loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "model_type": args.model,
                "feature_cols": FEATURE_COLS,
                "window": args.window,
            }, model_path)

    # ========== FINAL METRICS ==========
    accuracy = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    
    print("Finished training.")
    print({"val_loss": best_val, "accuracy": accuracy, "f1_macro": f1})
    print("Saved best model to", model_path)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Main function - parse args and start training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
