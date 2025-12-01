"""
================================================================================
CONDITIONAL VARIATIONAL AUTOENCODER (CVAE) - SMART ADVISOR
================================================================================
This script trains a generative model that can suggest lifestyle changes to
reduce burnout by generating "counterfactual" schedules.

WHAT IS A CVAE?
---------------
A Conditional VAE learns the distribution of data conditioned on a label:
    P(Sequence | BurnoutLevel)

This allows us to:
1. Encode a real schedule into a latent space
2. Decode it back with a DIFFERENT burnout level condition
3. Get a "counterfactual" schedule: what would this week look like with low burnout?

THE "SMART ADVISOR" CONCEPT:
----------------------------
Instead of manually guessing "what if I sleep more?", you can ask:
    "Generate a schedule similar to mine, but where I don't burn out."

The model learns from thousands of low-burnout weeks what patterns are associated
with wellbeing, and applies those patterns while keeping your individual style.

ARCHITECTURE:
------------
1. Condition Embedding: BurnoutLevel (0,1,2) → embedding vector
2. Encoder: (Sequence + Label) → (mu, logvar) latent distribution
3. Reparameterization: z = mu + sigma * epsilon (differentiable sampling)
4. Decoder: (z + Label) → Reconstructed sequence

LOSS FUNCTION:
--------------
VAE Loss = Reconstruction Loss (MSE) + β * KL Divergence

- Reconstruction: How well can we rebuild the original sequence?
- KL Divergence: Force latent space to be close to N(0,1) for smooth interpolation

Usage:
    python scripts/train_cvae.py --epochs 100 --sample-users 1.0
    
    # Generate advice for a specific user:
    python scripts/train_cvae.py --suggest --input-sequence path/to/seq.npy

Input:  data/processed/daily_with_burnout.parquet
Output: models/saved/cvae_advisor.pt

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DIR = Path("data/processed")
DAILY_PATH = PROCESSED_DIR / "daily_with_burnout.parquet"
MODEL_DIR = Path("models/saved")

# Same features as other models
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

# Human-readable feature names for advice generation
FEATURE_NAMES = {
    "stress_level": "Stress Level",
    "commute_minutes": "Commute Time (min)",
    "exercise_minutes": "Exercise (min)",
    "work_hours": "Work Hours",
    "mood_score": "Mood Score",
    "sleep_quality": "Sleep Quality",
    "emails_received": "Emails Received",
    "caffeine_mg": "Caffeine (mg)",
    "energy_level": "Energy Level",
    "sleep_hours": "Sleep Hours",
    "focus_score": "Focus Score",
    "meetings_count": "Meetings",
    "tasks_completed": "Tasks Completed",
    "steps_count": "Steps",
    "alcohol_units": "Alcohol Units",
    "screen_time_hours": "Screen Time (hrs)",
    "work_pressure": "Work Pressure",
}


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CVAE Smart Advisor for Burnout Prevention")
    parser.add_argument("--window", type=int, default=7,
                        help="Sequence window size in days (default: 7)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="Latent space dimension (default: 32)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden layer dimension (default: 128)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="KL divergence weight (default: 1.0)")
    parser.add_argument("--sample-users", type=float, default=0.1,
                        help="Fraction of users to include (default: 0.1)")
    parser.add_argument("--suggest", action="store_true",
                        help="Run suggestion mode instead of training")
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_sequences_with_labels(window: int, sample_users: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load sequences with burnout labels."""
    if not DAILY_PATH.exists():
        raise FileNotFoundError("Run scripts/create_burnout_labels.py first")
    
    daily = pd.read_parquet(DAILY_PATH)
    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)

    if daily["work_pressure"].dtype == object:
        pressure_map = {"low": 0, "medium": 1, "high": 2}
        daily["work_pressure"] = daily["work_pressure"].map(pressure_map).fillna(1).astype(np.float32)

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
# CVAE MODEL
# ============================================================================

class BurnoutCVAE(nn.Module):
    """
    Conditional Variational Autoencoder for burnout schedule generation.
    
    Architecture:
    - Condition Embedding: Maps burnout level (0,1,2) to embedding
    - Encoder: (Sequence, Condition) → (mu, logvar)
    - Decoder: (z, Condition) → Reconstructed Sequence
    
    The conditioning allows generating schedules for specific burnout levels.
    """
    
    def __init__(
        self,
        input_dim: int,
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
        
        # Flatten sequence dimension
        self.flat_dim = seq_len * input_dim
        
        # Condition embedding: burnout level → vector
        self.condition_embedding = nn.Embedding(num_classes, condition_dim)
        
        # ========== ENCODER ==========
        # Input: flattened sequence + condition embedding
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
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # ========== DECODER ==========
        # Input: latent z + condition embedding
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
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence to latent distribution.
        
        Args:
            x: Input sequence (batch, seq_len, input_dim)
            condition: Burnout labels (batch,)
            
        Returns:
            mu: Latent mean (batch, latent_dim)
            logvar: Latent log-variance (batch, latent_dim)
        """
        batch_size = x.size(0)
        
        # Flatten sequence
        x_flat = x.view(batch_size, -1)  # (batch, seq_len * input_dim)
        
        # Get condition embedding
        cond_emb = self.condition_embedding(condition)  # (batch, condition_dim)
        
        # Concatenate and encode
        encoder_input = torch.cat([x_flat, cond_emb], dim=-1)
        hidden = self.encoder(encoder_input)
        
        # Get distribution parameters
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling.
        
        z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent vector to sequence.
        
        Args:
            z: Latent vector (batch, latent_dim)
            condition: Target burnout labels (batch,)
            
        Returns:
            reconstructed: Reconstructed sequence (batch, seq_len, input_dim)
        """
        batch_size = z.size(0)
        
        # Get condition embedding
        cond_emb = self.condition_embedding(condition)
        
        # Concatenate and decode
        decoder_input = torch.cat([z, cond_emb], dim=-1)
        x_flat = self.decoder(decoder_input)
        
        # Reshape to sequence
        reconstructed = x_flat.view(batch_size, self.seq_len, self.input_dim)
        
        return reconstructed
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, sample, decode.
        
        Returns:
            reconstructed: Reconstructed sequence
            mu: Latent mean
            logvar: Latent log-variance
        """
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, condition)
        
        return reconstructed, mu, logvar
    
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
        
        Args:
            original_sequence: Your actual week (1, seq_len, input_dim)
            original_label: Your actual burnout level (1,)
            target_label: Desired burnout level (default: 0 = Low)
            
        Returns:
            suggested_sequence: Counterfactual schedule (1, seq_len, input_dim)
        """
        self.eval()
        with torch.no_grad():
            # Encode the original sequence with its true label
            mu, logvar = self.encode(original_sequence, original_label)
            
            # Use the mean (no sampling for deterministic suggestion)
            z = mu
            
            # Decode with the TARGET label (e.g., Low Burnout)
            target_cond = torch.tensor([target_label], device=z.device)
            suggested = self.decode(z, target_cond)
            
        return suggested


# ============================================================================
# LOSS FUNCTION
# ============================================================================

def vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE Loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        reconstructed: Decoded sequence
        original: Original sequence
        mu: Latent mean
        logvar: Latent log-variance
        beta: KL weight (β-VAE)
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction component
        kl_loss: KL divergence component
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, original, reduction='mean')
    
    # KL Divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, 1)
    # = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("Loading sequences...")
    X, y = load_sequences_with_labels(args.window, args.sample_users)
    print(f"Total sequences: {len(X):,}")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Normalize features
    X_flat = X.reshape(-1, X.shape[2])
    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0) + 1e-8
    X = (X - mean) / std
    print("Features normalized")
    
    # Create DataLoader
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Build model
    input_dim = X.shape[2]
    model = BurnoutCVAE(
        input_dim=input_dim,
        seq_len=args.window,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(DEVICE)
    
    print(f"\nCVAE Model:")
    print(f"  Input dim: {input_dim}")
    print(f"  Sequence length: {args.window}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Beta (KL weight): {args.beta}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    
    print(f"\nStarting CVAE training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"total": [], "recon": [], "kl": []}
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, logvar = model(batch_x, batch_y)
            
            # Compute loss
            total_loss, recon_loss, kl_loss = vae_loss(
                reconstructed, batch_x, mu, logvar, beta=args.beta
            )
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses["total"].append(total_loss.item())
            epoch_losses["recon"].append(recon_loss.item())
            epoch_losses["kl"].append(kl_loss.item())
        
        scheduler.step()
        
        avg_total = np.mean(epoch_losses["total"])
        avg_recon = np.mean(epoch_losses["recon"])
        avg_kl = np.mean(epoch_losses["kl"])
        
        # Log progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: total={avg_total:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}")
        
        # Save best model
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "seq_len": args.window,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "feature_cols": FEATURE_COLS,
                "normalization": {"mean": mean.tolist(), "std": std.tolist()},
            }, MODEL_DIR / "cvae_advisor.pt")
    
    print("=" * 60)
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Saved model to: {MODEL_DIR / 'cvae_advisor.pt'}")
    
    # Demo: Generate advice for a sample sequence
    print("\n" + "=" * 60)
    print("DEMO: Smart Advisor in Action")
    print("=" * 60)
    demo_advice(model, X, y, mean, std, DEVICE)


# ============================================================================
# ADVICE GENERATION
# ============================================================================

def demo_advice(
    model: BurnoutCVAE,
    X: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> None:
    """
    Demonstrate the Smart Advisor by generating advice for a high-burnout user.
    """
    # Find a high-burnout sequence
    high_burnout_idx = np.where(y == 2)[0]
    if len(high_burnout_idx) == 0:
        print("No high-burnout sequences found for demo.")
        return
    
    sample_idx = high_burnout_idx[0]
    original_seq = X[sample_idx:sample_idx+1]  # (1, 7, features)
    original_label = y[sample_idx:sample_idx+1]
    
    # Convert to tensors
    original_seq_t = torch.from_numpy(original_seq).float().to(device)
    original_label_t = torch.from_numpy(original_label).long().to(device)
    
    # Generate counterfactual (target: Low Burnout)
    suggested_seq_t = model.suggest_changes(
        original_seq_t,
        original_label_t,
        target_label=0,  # Low Burnout
    )
    
    # Convert back and denormalize
    original_denorm = original_seq[0] * std + mean
    suggested_denorm = suggested_seq_t.cpu().numpy()[0] * std + mean
    
    # Compute weekly averages
    original_avg = original_denorm.mean(axis=0)
    suggested_avg = suggested_denorm.mean(axis=0)
    
    print("\nOriginal Schedule (High Burnout) vs Suggested (Low Burnout)")
    print("-" * 60)
    print(f"{'Feature':<25} {'Current':>12} {'Suggested':>12} {'Change':>12}")
    print("-" * 60)
    
    for i, col in enumerate(FEATURE_COLS):
        name = FEATURE_NAMES.get(col, col)
        orig_val = original_avg[i]
        sugg_val = suggested_avg[i]
        change = sugg_val - orig_val
        change_str = f"{change:+.1f}" if abs(change) > 0.1 else "~"
        
        # Highlight significant changes
        if abs(change) > 0.5:
            print(f"{name:<25} {orig_val:>12.1f} {sugg_val:>12.1f} {change_str:>12} ⬅️")
        else:
            print(f"{name:<25} {orig_val:>12.1f} {sugg_val:>12.1f} {change_str:>12}")
    
    print("-" * 60)
    print("\n💡 Key Suggestions (changes > 0.5):")
    
    for i, col in enumerate(FEATURE_COLS):
        change = suggested_avg[i] - original_avg[i]
        if abs(change) > 0.5:
            name = FEATURE_NAMES.get(col, col)
            direction = "increase" if change > 0 else "decrease"
            print(f"   • {direction.capitalize()} {name} by {abs(change):.1f}")


def generate_advice_from_checkpoint(
    checkpoint_path: Path,
    sequence: np.ndarray,
    current_label: int,
    target_label: int = 0,
) -> dict:
    """
    Load trained model and generate advice for a given sequence.
    
    Args:
        checkpoint_path: Path to cvae_advisor.pt
        sequence: Input sequence (7, features) or (1, 7, features)
        current_label: Current burnout level
        target_label: Target burnout level (default: 0 = Low)
        
    Returns:
        Dictionary with original, suggested, and changes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model
    model = BurnoutCVAE(
        input_dim=checkpoint["input_dim"],
        seq_len=checkpoint["seq_len"],
        latent_dim=checkpoint["latent_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # Get normalization stats
    mean = np.array(checkpoint["normalization"]["mean"])
    std = np.array(checkpoint["normalization"]["std"])
    
    # Prepare input
    if sequence.ndim == 2:
        sequence = sequence[np.newaxis, ...]  # Add batch dim
    
    # Normalize
    seq_norm = (sequence - mean) / std
    seq_t = torch.from_numpy(seq_norm).float().to(device)
    label_t = torch.tensor([current_label], device=device)
    
    # Generate suggestion
    suggested_t = model.suggest_changes(seq_t, label_t, target_label)
    
    # Denormalize
    suggested = suggested_t.cpu().numpy()[0] * std + mean
    
    return {
        "original": sequence[0],
        "suggested": suggested,
        "changes": suggested - sequence[0],
        "feature_cols": checkpoint["feature_cols"],
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()
    
    if args.suggest:
        print("Suggestion mode not fully implemented in CLI.")
        print("Use generate_advice_from_checkpoint() in Python instead.")
        return
    
    train(args)


if __name__ == "__main__":
    main()
