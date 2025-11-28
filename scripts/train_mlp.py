"""
================================================================================
TRAIN MLP CLASSIFIER
================================================================================
This script trains a Multi-Layer Perceptron (MLP) neural network to classify
burnout levels (Low/Medium/High) from tabular weekly features.

The MLP is a simple but effective baseline deep learning model:
- Input: Preprocessed weekly features (behavioral statistics + demographics)
- Architecture: 256 -> 128 -> 3 neurons with BatchNorm, ReLU, Dropout
- Output: 3-class probability distribution (softmax)

MLPs work well on tabular data when:
- Features are already engineered (aggregated, scaled)
- Dataset is medium-sized (thousands to millions of rows)
- Interpretability is less critical than accuracy

Usage:
    python scripts/train_mlp.py --epochs 80 --batch-size 256

Input:  data/processed/tabular_ml_ready.parquet (from preprocess.py)
Output: models/saved/mlp_classifier.pt

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
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
MODEL_DIR = Path("models/saved")

# Input and output paths
TABULAR_PATH = PROCESSED_DIR / "tabular_ml_ready.parquet"
SCALER_PATH = PROCESSED_DIR / "feature_scaler.joblib"
MODEL_OUT = MODEL_DIR / "mlp_classifier.pt"


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train MLP burnout classifier")
    parser.add_argument("--epochs", type=int, default=80, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size for SGD")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 regularization strength")
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the preprocessed tabular dataset.
    
    Returns:
        X: Feature matrix (N x D) as float32
        y: Target labels (N,) as int64
        feature_cols: List of feature column names
    """
    if not TABULAR_PATH.exists():
        raise FileNotFoundError("Run scripts/preprocess.py before training the MLP.")
    
    df = pd.read_parquet(TABULAR_PATH)
    
    # Separate features from targets
    feature_cols = [c for c in df.columns if c not in {"burnout_level", "burnout_score"}]
    X = df[feature_cols].values.astype(np.float32)
    y = df["burnout_level"].values.astype(np.int64)
    
    return X, y, feature_cols


def build_loaders(X: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[DataLoader, DataLoader, np.ndarray]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    We use a simple 80/20 split (not shuffled, so temporal order is preserved).
    For production, consider using proper cross-validation.
    
    Args:
        X: Feature matrix
        y: Target labels
        batch_size: Mini-batch size
        
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        y_val: Validation labels (for metric computation)
    """
    n = len(X)
    split = int(0.8 * n)  # 80% train, 20% validation
    
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Wrap in TensorDataset for PyTorch
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    # Create DataLoaders
    # shuffle=True for training (SGD benefits from random order)
    # shuffle=False for validation (consistent evaluation)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    
    return train_loader, val_loader, y_val


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_model(input_dim: int, num_classes: int) -> nn.Module:
    """
    Build the MLP architecture.
    
    Architecture:
        Input (D features)
            -> Linear(256) -> BatchNorm -> ReLU -> Dropout(0.3)
            -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.2)
            -> Linear(num_classes)
    
    Design choices:
    - BatchNorm: Stabilizes training, allows higher learning rates
    - ReLU: Simple, effective non-linearity
    - Dropout: Prevents overfitting by randomly dropping neurons
    - Decreasing width: Gradually compresses information toward output
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes (3 for burnout levels)
        
    Returns:
        PyTorch Sequential model
    """
    return nn.Sequential(
        # First hidden layer: expand to 256 neurons
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),      # Normalize activations
        nn.ReLU(),                 # Non-linearity
        nn.Dropout(0.3),           # 30% dropout for regularization
        
        # Second hidden layer: compress to 128 neurons
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),           # Less dropout in later layers
        
        # Output layer: one neuron per class
        nn.Linear(128, num_classes),
        # Note: No softmax here - CrossEntropyLoss expects raw logits
    )


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args: argparse.Namespace) -> None:
    """
    Main training function.
    
    Implements a standard PyTorch training loop:
    1. Forward pass: compute predictions
    2. Compute loss (CrossEntropy for classification)
    3. Backward pass: compute gradients
    4. Optimizer step: update weights
    5. Validate and save best model
    """
    # Select device (GPU if available, else CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Load data
    X, y, feature_cols = load_data()
    train_loader, val_loader, y_val = build_loaders(X, y, args.batch_size)
    num_classes = len(np.unique(y))
    
    # Build model
    model = build_model(X.shape[1], num_classes).to(DEVICE)
    
    # Loss function: CrossEntropy combines LogSoftmax + NLLLoss
    # Suitable for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Track best validation loss for early stopping / model selection
    best_val = float("inf")
    history = []

    # ========== TRAINING LOOP ==========
    for epoch in range(1, args.epochs + 1):
        
        # ----- Training phase -----
        model.train()  # Enable dropout and batch norm training mode
        train_losses = []
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()          # Clear previous gradients
            logits = model(xb)             # Forward pass
            loss = criterion(logits, yb)   # Compute loss
            loss.backward()                # Backward pass (compute gradients)
            optimizer.step()               # Update weights
            
            train_losses.append(loss.item())

        # ----- Validation phase -----
        model.eval()  # Disable dropout, use running stats for batch norm
        val_losses = []
        preds = []
        
        with torch.no_grad():  # No gradient computation needed
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        
        # Compute metrics
        val_loss = float(np.mean(val_losses))
        f1 = f1_score(y_val, preds, average="macro")
        
        # Log history
        history.append({
            "epoch": epoch, 
            "train_loss": float(np.mean(train_losses)), 
            "val_loss": val_loss, 
            "f1_macro": f1
        })
        
        # Print progress every 5 epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: val_loss={val_loss:.4f} f1={f1:.3f}")
        
        # Save best model (lowest validation loss)
        if val_loss < best_val:
            best_val = val_loss
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "scaler_path": str(SCALER_PATH),
            }, MODEL_OUT)

    # ========== TRAINING COMPLETE ==========
    print("Training complete. Model saved to", MODEL_OUT)
    print("Best epoch:", min(history, key=lambda x: x["val_loss"]))


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main() -> None:
    args = parse_args()
    train(args)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
