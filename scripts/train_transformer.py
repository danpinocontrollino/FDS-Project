"""
================================================================================
TRAIN TRANSFORMER CLASSIFIER
================================================================================
This script trains a Transformer-based neural network to classify burnout levels
from 7-day behavioral time series windows.

WHAT IS A TRANSFORMER?
----------------------
The Transformer architecture (Vaswani et al., "Attention Is All You Need", 2017)
revolutionized NLP and has since been applied to many domains including:
- Language models (GPT, BERT)
- Vision (ViT)
- Time series (this project!)

Key innovation: SELF-ATTENTION mechanism allows the model to weigh the importance
of different positions in the sequence when making predictions.

For our burnout prediction task:
- The model can learn that "Tuesday's stress spike" matters more if followed by
  "Wednesday's sleep deficit" for predicting Friday's burnout level.
- Unlike LSTM (sequential), Transformer processes all days in parallel.

ARCHITECTURE OVERVIEW:
---------------------
1. Linear Projection: 15 raw features → d_model (64) embedding dimensions
2. Positional Encoding: Adds sinusoidal signals to encode day positions
3. Transformer Encoder: 2 layers of multi-head self-attention
4. Mean Pooling: Aggregate all 7 day representations into one vector
5. Classification Head: Linear layer → 3 burnout classes

WHY TRANSFORMER OVER LSTM?
--------------------------
- Parallel computation: All timesteps processed simultaneously (faster on GPU)
- Long-range dependencies: Self-attention can directly connect any two days
- Interpretability: Attention weights show which days influenced the prediction
- Modern baseline: State-of-the-art architecture for sequence modeling

HYPERPARAMETERS:
---------------
- d_model: Embedding dimension (64 for efficiency on CPU)
- nhead: Number of attention heads (4 - each learns different patterns)
- num_layers: Depth of encoder (2 - sufficient for 7-day sequences)
- dim_feedforward: Hidden size in feed-forward layers (256)

Usage:
    python scripts/train_transformer.py --window 7 --epochs 40
    python scripts/train_transformer.py --sample-users 0.1  # CPU optimization

Input:  data/processed/daily_with_burnout.parquet (from create_burnout_labels.py)
Output: models/saved/transformer_sequence.pt

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

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DIR = Path("data/processed")
DAILY_PATH = PROCESSED_DIR / "daily_with_burnout.parquet"
MODEL_DIR = Path("models/saved")

# Daily behavioral features (same as LSTM/GRU/CNN for fair comparison)
FEATURE_COLS = [
    # Highest correlation features
    "stress_level",          # Self-reported stress (1-10) - corr: 0.345
    "commute_minutes",       # Daily commute time - corr: 0.340
    "exercise_minutes",      # Physical activity - corr: 0.222
    "work_hours",            # Hours worked that day - corr: 0.201
    "mood_score",            # Emotional state (1-10) - corr: 0.175
    "sleep_quality",         # Subjective sleep quality (1-10) - corr: 0.142
    "emails_received",       # Email volume (workload proxy) - corr: 0.123
    "caffeine_mg",           # Stimulant consumption - corr: 0.116
    "energy_level",          # Fatigue indicator (1-10) - corr: 0.112
    "sleep_hours",           # Total sleep duration - corr: 0.094
    "focus_score",           # Cognitive performance (1-10) - corr: 0.090
    
    # Secondary features
    "meetings_count",        # Number of meetings (interruption proxy)
    "tasks_completed",       # Productivity metric
    "steps_count",           # Daily movement
    "alcohol_units",         # Depressant consumption
    "screen_time_hours",     # Digital exposure
    
    # Social & lifestyle (NEW - high value for burnout)
    "social_interactions",   # Social connection indicator
    "outdoor_time_minutes",  # Recovery/nature exposure
    "diet_quality",          # Nutrition quality score
    
    # Work environment (categorical → numeric)
    "work_pressure",
]


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Transformer training configuration.
    
    Transformer-specific args:
    - --d-model: Embedding dimension (smaller = faster, larger = more capacity)
    - --nhead: Number of attention heads (must divide d_model evenly)
    - --num-layers: Depth of the encoder stack
    """
    parser = argparse.ArgumentParser(description="Train Transformer burnout classifier")
    parser.add_argument("--window", type=int, default=7,
                        help="Sequence window size in days (default: 7)")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=512, 
                        help="Mini-batch size (larger = faster on CPU)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for AdamW optimizer")
    parser.add_argument("--d-model", type=int, default=64, 
                        help="Transformer embedding dimension (smaller = faster)")
    parser.add_argument("--nhead", type=int, default=4, 
                        help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, 
                        help="Number of encoder layers")
    parser.add_argument("--sample-users", type=float, default=0.1, 
                        help="Fraction of USERS to include (default 0.1 = 10%% for CPU training)")
    parser.add_argument("--forecast-horizon", type=int, default=7,
                        help="Days ahead to predict burnout (default 7 = next week)")
    return parser.parse_args()


# ============================================================================
# DATA LOADING & SEQUENCE CREATION
# ============================================================================

def load_daily(window: int, sample_users: float = 1.0, forecast_horizon: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load daily data and create sliding window sequences for FORECASTING.
    
    CRITICAL: We predict FUTURE burnout, not same-period burnout.
    This prevents data leakage and creates a realistic early warning system.
    
    Args:
        window: Number of days in each sequence
        sample_users: Fraction of users to include (for CPU training)
        forecast_horizon: Days into the future to predict (default 7 = next week)
        
    Returns:
        X: Sequences array (N × window × features)
        y: Labels array (N,)
    """
    if not DAILY_PATH.exists():
        raise FileNotFoundError(
            "Run scripts/create_burnout_labels.py first to generate daily_with_burnout.parquet"
        )
    
    # Load and sort data
    daily = pd.read_parquet(DAILY_PATH)
    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)

    # Convert categorical work_pressure to numeric (ordinal encoding)
    if daily["work_pressure"].dtype == object:
        pressure_map = {"low": 0, "medium": 1, "high": 2}
        daily["work_pressure"] = daily["work_pressure"].map(pressure_map).fillna(1).astype(np.float32)

    # ========== STRATIFIED USER SAMPLING ==========
    # Sample users proportionally by their dominant burnout level
    # This preserves population-level class distribution
    if sample_users < 1.0:
        # Compute dominant burnout level per user (mode)
        user_burnout = daily.groupby("user_id")["burnout_level"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median()
        )
        user_burnout = user_burnout.reset_index()
        user_burnout.columns = ["user_id", "dominant_burnout"]
        
        n_users_total = len(user_burnout)
        n_users_sample = max(3, int(n_users_total * sample_users))
        
        # Stratified sampling from each burnout stratum
        selected_users = []
        for level in sorted(user_burnout["dominant_burnout"].unique()):
            level_users = user_burnout[user_burnout["dominant_burnout"] == level]["user_id"].values
            level_proportion = len(level_users) / n_users_total
            n_from_level = max(1, int(n_users_sample * level_proportion))
            sampled = np.random.choice(level_users, min(n_from_level, len(level_users)), replace=False)
            selected_users.extend(sampled)
        
        daily = daily[daily["user_id"].isin(selected_users)]
        print(f"Stratified sampling: {len(selected_users):,} users ({sample_users*100:.0f}% of {n_users_total:,})")
        print(f"  Class distribution preserved: {daily['burnout_level'].value_counts(normalize=True).round(3).to_dict()}")

    # ========== SLIDING WINDOW CREATION (FORECASTING) ==========
    sequences, labels = [], []
    
    for uid, group in daily.groupby("user_id"):
        feats = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        labs = group["burnout_level"].to_numpy(dtype=np.int64)
        
        # Need enough data for window + forecast horizon
        if len(group) < window + forecast_horizon:
            continue
            
        # Create windows with FUTURE labels
        for idx in range(window, len(group) - forecast_horizon + 1):
            seq = feats[idx - window: idx]          # 7-day window (days t-6 to t)
            label = labs[idx + forecast_horizon - 1]  # Burnout at day t+7
            
            if np.isnan(seq).any():
                continue
                
            sequences.append(seq)
            labels.append(label)
    
    print(f"Forecasting mode: predicting burnout {forecast_horizon} days ahead")
    return np.stack(sequences), np.array(labels)


# ============================================================================
# DATALOADER CONSTRUCTION
# ============================================================================

def build_loaders(X: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[DataLoader, DataLoader, np.ndarray]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Random 80/20 split with shuffling for training.
    
    Args:
        X: Sequences array (N × window × features)
        y: Labels array (N,)
        batch_size: Mini-batch size
        
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        y_val: Validation labels for metric computation
    """
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    # pin_memory=False for CPU training (avoids warnings)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, 
                            num_workers=0, pin_memory=False)
    
    return train_loader, val_loader, y_val


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need" (Vaswani et al., 2017).
    
    WHY POSITIONAL ENCODING?
    Transformers have no inherent notion of sequence order (unlike RNNs).
    Self-attention treats the input as a SET, not a SEQUENCE.
    Positional encoding injects position information into the embeddings.
    
    FORMULA:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Where:
        - pos: position in sequence (0 to seq_len-1, i.e., day index)
        - i: dimension index
        - d_model: embedding dimension
    
    The sinusoidal functions create unique patterns for each position that
    the model can learn to interpret as "Day 1", "Day 2", etc.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length supported
            dropout: Dropout rate applied after adding positional encoding
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # Compute division term for sinusoidal frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model) for batch broadcasting
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for sequence data.
    
    ARCHITECTURE:
    ------------
    Input: (batch, seq_len=7, features=15)
        │
        ▼
    [Linear Projection] → (batch, 7, d_model=64)
        │
        ▼
    [Positional Encoding] → Add position information
        │
        ▼
    [Transformer Encoder] × 2 layers
        │  └─ Multi-Head Self-Attention (4 heads)
        │  └─ Feed-Forward Network (256 hidden)
        │  └─ LayerNorm + Residual connections
        │
        ▼
    [Mean Pooling] → (batch, d_model) - aggregate sequence
        │
        ▼
    [Classification Head] → (batch, 3) - burnout probabilities
    
    SELF-ATTENTION MECHANISM:
    ------------------------
    For each day in the sequence, attention computes:
    - Query (Q): "What am I looking for?"
    - Key (K): "What do I contain?"
    - Value (V): "What information do I provide?"
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) × V
    
    This allows any day to "attend to" any other day, learning
    patterns like "high stress on day 3 + poor sleep on day 5 → burnout".
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize Transformer classifier.
        
        Args:
            input_dim: Number of input features per timestep (15)
            seq_len: Sequence length (7 days)
            num_classes: Number of output classes (3 burnout levels)
            d_model: Embedding dimension
            nhead: Number of attention heads (must divide d_model)
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Hidden size in feed-forward networks
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.d_model = d_model

        # Project raw features to embedding space
        # 15 features → 64 dimensions
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding (adds day position information)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',      # GELU works well for transformers
            batch_first=True,        # Input shape: (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),   # Normalize before classification
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Step 1: Project input features to d_model dimensions
        # (batch, 7, 15) → (batch, 7, 64)
        x = self.input_projection(x)

        # Step 2: Add positional encoding
        # The model now knows "this is day 1, this is day 2, ..."
        x = self.pos_encoder(x)

        # Step 3: Pass through Transformer encoder
        # Self-attention allows each day to attend to all other days
        x = self.transformer_encoder(x)  # (batch, 7, 64)

        # Step 4: Mean pooling over sequence dimension
        # Aggregate all 7 day representations into one vector
        # Alternative: take last hidden state (like LSTM) or use [CLS] token
        x = x.mean(dim=1)  # (batch, 64)

        # Step 5: Classification head
        return self.head(x)  # (batch, 3)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args: argparse.Namespace) -> None:
    """
    Main training function for Transformer classifier.
    
    Training features:
    - AdamW optimizer (Adam with proper weight decay)
    - Cosine annealing learning rate schedule
    - Gradient clipping (max norm = 1.0) for stability
    - Model checkpointing (save best validation loss)
    """
    # Select device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load data with forecasting
    print("Loading data...")
    seq_X, seq_y = load_daily(args.window, args.sample_users, args.forecast_horizon)
    print(f"Total sequences: {len(seq_X):,} with shape {seq_X.shape}")

    # Build data loaders
    train_loader, val_loader, y_val = build_loaders(seq_X, seq_y, args.batch_size)
    num_classes = len(np.unique(seq_y))
    print(f"Number of classes: {num_classes}")

    # Build model
    model = TransformerClassifier(
        input_dim=seq_X.shape[2],
        seq_len=seq_X.shape[1],
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
    ).to(DEVICE)

    # Report model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss function with label smoothing (reduces overconfidence)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW optimizer (better than Adam for transformers)
    # Weight decay = L2 regularization, properly decoupled in AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Cosine annealing schedule: LR decreases smoothly over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Tracking
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "transformer_sequence.pt"

    # ========== TRAINING LOOP ==========
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        
        # ----- Training phase -----
        model.train()
        train_losses = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()           # Clear gradients
            loss = criterion(model(xb), yb)  # Forward + loss
            loss.backward()                  # Backward
            
            # Gradient clipping: prevent exploding gradients
            # Common issue with Transformers on small datasets
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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

        # Compute epoch metrics
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Update learning rate
        scheduler.step()

        # Log progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "model_type": "transformer",
                "feature_cols": FEATURE_COLS,
                "window": args.window,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
            }, model_path)

    # ========== FINAL METRICS ==========
    accuracy = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    cm = confusion_matrix(y_val, preds)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Loss: {best_val:.4f}")
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(f"Final F1 (macro): {f1*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"        Pred→  LOW    MED   HIGH")
    for i, row in enumerate(cm):
        label = ['LOW ', 'MED ', 'HIGH'][i]
        print(f"  {label}:  {row}")
    print(f"\nPer-class report:")
    print(classification_report(y_val, preds, target_names=['LOW', 'MEDIUM', 'HIGH']))
    
    # Update saved model with final metrics
    checkpoint = torch.load(model_path, weights_only=False)
    checkpoint['metrics'] = {
        'val_loss': best_val,
        'val_acc': accuracy,
        'val_f1': f1,
        'confusion_matrix': cm.tolist(),
    }
    torch.save(checkpoint, model_path)
    print(f"\nModel saved to: {model_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Main function - parse args and start training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
