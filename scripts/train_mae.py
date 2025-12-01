"""
================================================================================
MASKED AUTOENCODER (MAE) PRE-TRAINING SCRIPT
================================================================================
This script implements self-supervised pre-training for behavioral time-series
using a Masked Autoencoder approach, inspired by BERT and Vision Transformers.

WHAT IS MAE PRE-TRAINING?
-------------------------
Instead of learning to classify burnout directly, we first teach the model
to understand the "grammar" of human behavior by:
1. Taking a 7-day behavioral sequence
2. Randomly masking 2 days (hiding their features)
3. Training the model to reconstruct the hidden days

This forces the model to learn patterns like:
- "High stress on Monday often correlates with poor sleep on Tuesday"
- "Exercise deficit leads to lower energy levels"
- "Long commutes associate with higher caffeine consumption"

WHY PRE-TRAIN?
--------------
1. Learn from ALL data (not just labeled examples)
2. Discover latent behavioral patterns
3. Better generalization to downstream tasks (burnout prediction)
4. Especially useful when labeled data is limited

ARCHITECTURE:
------------
1. Input Projection: 18 raw features → d_model (64) embeddings
2. Mask Token: Learnable vector that replaces masked days
3. Positional Encoding: Sinusoidal encoding for day positions
4. Transformer Encoder: 2-layer self-attention
5. Reconstruction Head: Projects back to original feature space

MASKING STRATEGY:
-----------------
- Mask 2 out of 7 days (~28% masking ratio)
- Replace masked positions with learnable [MASK] token
- Only compute loss on masked positions (reconstruction task)

Usage:
    python scripts/train_mae.py --epochs 50
    python scripts/train_mae.py --sample-users 0.1 --epochs 30  # CPU optimization

Input:  data/processed/daily_with_burnout.parquet
Output: models/saved/mae_encoder.pt (pre-trained encoder weights)

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
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DIR = Path("data/processed")
DAILY_PATH = PROCESSED_DIR / "daily_with_burnout.parquet"
MODEL_DIR = Path("models/saved")

# Same features as train_lstm.py for consistency
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
    parser = argparse.ArgumentParser(description="MAE Pre-training for Behavioral Time-Series")
    parser.add_argument("--window", type=int, default=7,
                        help="Sequence window size in days (default: 7)")
    parser.add_argument("--mask-ratio", type=float, default=0.3,
                        help="Fraction of days to mask (default: 0.3 = ~2 out of 7)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of pre-training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--d-model", type=int, default=64,
                        help="Transformer embedding dimension (default: 64)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of Transformer encoder layers (default: 2)")
    parser.add_argument("--sample-users", type=float, default=0.1,
                        help="Fraction of users to include (default: 0.1 for CPU)")
    return parser.parse_args()


# ============================================================================
# DATA LOADING (Reused from train_lstm.py)
# ============================================================================

def load_sequences(window: int, sample_users: float = 1.0) -> np.ndarray:
    """
    Load daily data and create sliding window sequences (unsupervised - no labels needed).
    
    Returns:
        X: Sequences array (N × window × features)
    """
    if not DAILY_PATH.exists():
        raise FileNotFoundError(
            "Run scripts/create_burnout_labels.py first to generate daily_with_burnout.parquet"
        )
    
    daily = pd.read_parquet(DAILY_PATH)
    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)

    # Convert categorical work_pressure to numeric
    if daily["work_pressure"].dtype == object:
        pressure_map = {"low": 0, "medium": 1, "high": 2}
        daily["work_pressure"] = daily["work_pressure"].map(pressure_map).fillna(1).astype(np.float32)

    # Stratified user sampling for CPU optimization
    if sample_users < 1.0:
        user_ids = daily["user_id"].unique()
        n_sample = max(3, int(len(user_ids) * sample_users))
        selected = np.random.choice(user_ids, n_sample, replace=False)
        daily = daily[daily["user_id"].isin(selected)]
        print(f"Sampled {len(selected):,} users ({sample_users*100:.0f}%)")

    # Create sliding window sequences
    sequences = []
    for uid, group in daily.groupby("user_id"):
        feats = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        
        if len(group) < window:
            continue
        
        for idx in range(window, len(group) + 1):
            seq = feats[idx - window: idx]
            if not np.isnan(seq).any():
                sequences.append(seq)
    
    return np.stack(sequences)


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    Adds position information to embeddings so the model knows day order.
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


# ============================================================================
# MASKED AUTOENCODER MODEL
# ============================================================================

class MAETransformer(nn.Module):
    """
    Masked Autoencoder with Transformer Encoder.
    
    Architecture:
    1. Input projection: features → d_model
    2. Learnable [MASK] token for masked positions
    3. Positional encoding
    4. Transformer encoder (self-attention)
    5. Reconstruction head: d_model → features
    
    The encoder learns to predict masked days from context.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection: raw features → embedding space
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Learnable [MASK] token - replaces masked day features
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Reconstruction head: project back to original feature space
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, input_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with masking.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
            mask: Boolean mask (batch, seq_len) - True for masked positions
            
        Returns:
            reconstructed: Predicted features (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to embedding space
        embeddings = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Replace masked positions with [MASK] token
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.d_model)
        mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
        embeddings = torch.where(mask_expanded, mask_tokens, embeddings)
        
        # Add positional encoding
        embeddings = self.pos_encoder(embeddings)
        
        # Pass through Transformer encoder
        encoded = self.transformer_encoder(embeddings)
        
        # Reconstruct original features
        reconstructed = self.reconstruction_head(encoded)
        
        return reconstructed
    
    def get_encoder(self) -> nn.Module:
        """
        Extract the encoder components for downstream tasks.
        Returns a module containing input_proj, pos_encoder, and transformer_encoder.
        """
        class Encoder(nn.Module):
            def __init__(self, input_proj, pos_encoder, transformer_encoder, d_model):
                super().__init__()
                self.input_proj = input_proj
                self.pos_encoder = pos_encoder
                self.transformer_encoder = transformer_encoder
                self.d_model = d_model
            
            def forward(self, x):
                embeddings = self.input_proj(x)
                embeddings = self.pos_encoder(embeddings)
                return self.transformer_encoder(embeddings)
        
        return Encoder(
            self.input_proj,
            self.pos_encoder,
            self.transformer_encoder,
            self.d_model,
        )


# ============================================================================
# MASKING UTILITIES
# ============================================================================

def create_random_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Create random boolean mask for each sequence in the batch.
    
    Args:
        batch_size: Number of sequences
        seq_len: Length of each sequence (e.g., 7 days)
        mask_ratio: Fraction of positions to mask (e.g., 0.3)
        device: Torch device
        
    Returns:
        mask: Boolean tensor (batch_size, seq_len) - True for masked positions
    """
    num_mask = max(1, int(seq_len * mask_ratio))  # At least 1 masked position
    
    # Create mask for each sequence
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        # Randomly select positions to mask
        indices = torch.randperm(seq_len, device=device)[:num_mask]
        mask[i, indices] = True
    
    return mask


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args: argparse.Namespace) -> None:
    """Main pre-training function."""
    
    # Device selection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Load sequences (no labels needed for self-supervised learning)
    print("Loading sequences...")
    X = load_sequences(args.window, args.sample_users)
    print(f"Total sequences: {len(X):,}")
    print(f"Sequence shape: {X.shape} (N × window × features)")
    
    # Normalize features (important for reconstruction loss)
    X_flat = X.reshape(-1, X.shape[2])
    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0) + 1e-8
    X = (X - mean) / std
    print("Features normalized (z-score)")
    
    # Create DataLoader
    dataset = TensorDataset(torch.from_numpy(X).float())
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Build model
    input_dim = X.shape[2]
    model = MAETransformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(DEVICE)
    
    print(f"\nMAE Model:")
    print(f"  Input dim: {input_dim}")
    print(f"  d_model: {args.d_model}")
    print(f"  Attention heads: {args.nhead}")
    print(f"  Encoder layers: {args.num_layers}")
    print(f"  Mask ratio: {args.mask_ratio:.0%}")
    
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='none')  # Per-element loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    
    print(f"\nStarting MAE pre-training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_size, seq_len, _ = batch_x.shape
            
            # Create random mask
            mask = create_random_mask(batch_size, seq_len, args.mask_ratio, DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(batch_x, mask)
            
            # Compute MSE loss only on masked positions
            loss_per_element = criterion(reconstructed, batch_x)  # (batch, seq, features)
            loss_per_position = loss_per_element.mean(dim=-1)     # (batch, seq)
            
            # Only count loss from masked positions
            masked_loss = (loss_per_position * mask.float()).sum() / mask.float().sum()
            
            # Backward pass
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(masked_loss.item())
        
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        
        # Log progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: reconstruction_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Save full model and encoder separately
            save_dict = {
                "model_state": model.state_dict(),
                "encoder_state": model.get_encoder().state_dict(),
                "input_dim": input_dim,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "feature_cols": FEATURE_COLS,
                "normalization": {"mean": mean.tolist(), "std": std.tolist()},
            }
            torch.save(save_dict, MODEL_DIR / "mae_encoder.pt")
    
    print("=" * 60)
    print(f"Pre-training complete!")
    print(f"Best reconstruction loss: {best_loss:.4f}")
    print(f"Saved pre-trained encoder to: {MODEL_DIR / 'mae_encoder.pt'}")
    print("\nUse this encoder for downstream burnout classification!")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Main function."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
