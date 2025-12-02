"""
================================================================================
TRAIN MLP CLASSIFIER WITH ENTITY EMBEDDINGS
================================================================================
This script trains a Multi-Layer Perceptron (MLP) neural network to classify
burnout levels (Low/Medium/High) from tabular weekly features.

ENHANCED with Entity Embeddings for categorical variables (job_type, work_arrangement,
chronotype). Entity embeddings learn dense vector representations of categorical
features, which can capture semantic relationships between categories.

Architecture Overview:
- Categorical inputs → Entity Embeddings (learnable dense vectors)
- Numeric inputs → Directly concatenated
- Combined → MLP layers → Output probabilities

Benefits of Entity Embeddings:
1. Learn meaningful representations (e.g., "knowledge_work" similar to "creative_work")
2. Handle many categories efficiently without one-hot explosion
3. Improve generalization through learned feature interactions
4. Allow job-specific pattern recognition (key for personalized advice)

Usage:
    python scripts/train_mlp.py --epochs 80 --batch-size 256

Input:  data/processed/tabular_ml_ready.parquet (from preprocess.py)
        data/processed/feature_columns.json (embedding metadata)
Output: models/saved/mlp_classifier.pt

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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
FEATURE_META_PATH = PROCESSED_DIR / "feature_columns.json"
MODEL_OUT = MODEL_DIR / "mlp_classifier.pt"

# Entity Embedding Configuration
# Rule of thumb: embedding_dim = min(50, (num_categories + 1) // 2)
# For small category counts (<10), use 4-8 dimensions
DEFAULT_EMBEDDING_DIM = 4

# Categorical columns that use entity embeddings
EMBEDDING_COLS = ["job_type_encoded", "work_arrangement_encoded", "chronotype_encoded"]


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
    parser.add_argument("--binary", action="store_true", default=True,
                        help="Use binary classification (Healthy vs At-Risk) - DEFAULT")
    parser.add_argument("--three-class", action="store_true",
                        help="Use 3-class classification (Low/Med/High) instead of binary")
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_embedding_metadata() -> Dict:
    """
    Load embedding vocabulary sizes from preprocessing metadata.
    
    Returns:
        Dictionary with embedding_vocab_sizes and embedding_cols
    """
    if not FEATURE_META_PATH.exists():
        return {"embedding_vocab_sizes": {}, "embedding_cols": []}
    
    with open(FEATURE_META_PATH, "r") as f:
        meta = json.load(f)
    return meta


def load_data(binary: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], Dict]:
    """
    Load the preprocessed tabular dataset with separate handling for embeddings.
    
    Args:
        binary: If True, use burnout_binary (2-class) instead of burnout_level (3-class)
    
    Returns:
        X_numeric: Numeric feature matrix (N x D_numeric) as float32
        X_cat: Categorical indices for embeddings (N x num_cat_cols) as int64
        y: Target labels (N,) as int64
        feature_cols: List of all feature column names
        embedding_info: Dict with vocab sizes and column mappings
    """
    if not TABULAR_PATH.exists():
        raise FileNotFoundError("Run scripts/preprocess.py before training the MLP.")
    
    df = pd.read_parquet(TABULAR_PATH)
    
    # Load embedding metadata
    meta = load_embedding_metadata()
    embedding_vocab_sizes = meta.get("embedding_vocab_sizes", {})
    embedding_cols = [c for c in EMBEDDING_COLS if c in df.columns]
    
    # Determine target column
    target_col = "burnout_binary" if binary else "burnout_level"
    target_cols_to_exclude = {"burnout_level", "burnout_score", "burnout_binary"}
    
    # If using binary but column doesn't exist, create it from burnout_level
    if binary and target_col not in df.columns:
        if "burnout_level" not in df.columns:
            raise ValueError("Neither burnout_binary nor burnout_level found in data")
        # Create binary: Low (0) = Healthy (0), Medium (1) + High (2) = At Risk (1)
        df["burnout_binary"] = (df["burnout_level"] >= 1).astype(int)
        print("Created burnout_binary from burnout_level (Low=Healthy, Med+High=At-Risk)")
    
    # Separate features from targets
    all_feature_cols = [c for c in df.columns if c not in target_cols_to_exclude]
    
    # Split into categorical (for embeddings) and numeric features
    cat_cols = [c for c in embedding_cols if c in df.columns]
    numeric_cols = [c for c in all_feature_cols if c not in cat_cols]
    
    # Extract categorical indices (convert to int for embedding lookup)
    if cat_cols:
        X_cat = df[cat_cols].values.astype(np.int64)
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.int64)
    
    # Extract numeric features
    X_numeric = df[numeric_cols].values.astype(np.float32)
    
    # Target labels
    y = df[target_col].values.astype(np.int64)
    
    # Build embedding info
    embedding_info = {
        "cat_cols": cat_cols,
        "numeric_cols": numeric_cols,
        "vocab_sizes": {
            col.replace("_encoded", ""): embedding_vocab_sizes.get(col.replace("_encoded", ""), 10)
            for col in cat_cols
        },
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
    }
    
    return X_numeric, X_cat, y, all_feature_cols, embedding_info


def build_loaders(
    X_numeric: np.ndarray, 
    X_cat: np.ndarray, 
    y: np.ndarray, 
    batch_size: int
) -> tuple[DataLoader, DataLoader, np.ndarray]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    We use a simple 80/20 split (not shuffled, so temporal order is preserved).
    For production, consider using proper cross-validation.
    
    Args:
        X_numeric: Numeric feature matrix
        X_cat: Categorical indices for embeddings
        y: Target labels
        batch_size: Mini-batch size
        
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        y_val: Validation labels (for metric computation)
    """
    n = len(X_numeric)
    split = int(0.8 * n)  # 80% train, 20% validation
    
    X_num_train, X_num_val = X_numeric[:split], X_numeric[split:]
    X_cat_train, X_cat_val = X_cat[:split], X_cat[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Wrap in TensorDataset for PyTorch
    train_ds = TensorDataset(
        torch.from_numpy(X_num_train), 
        torch.from_numpy(X_cat_train),
        torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_num_val), 
        torch.from_numpy(X_cat_val),
        torch.from_numpy(y_val)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    
    return train_loader, val_loader, y_val


# ============================================================================
# MODEL ARCHITECTURE WITH ENTITY EMBEDDINGS
# ============================================================================

class BurnoutMLPWithEmbeddings(nn.Module):
    """
    MLP classifier with entity embeddings for categorical features.
    
    Architecture:
        Categorical inputs → Embedding layers (one per categorical column)
        Embedded vectors + Numeric inputs → Concatenate
        Combined → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
                → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
                → Linear(num_classes)
    
    Entity embeddings learn dense vector representations for categories:
    - job_type: "knowledge_work" might learn to be close to "creative_work"
    - chronotype: "morning" vs "evening" patterns captured in embedding space
    
    This allows the model to:
    1. Learn semantic similarities between categories
    2. Share information across related categories
    3. Handle new/rare categories better through learned representations
    """
    
    def __init__(
        self, 
        numeric_dim: int, 
        embedding_specs: Dict[str, int],  # {col_name: vocab_size}
        embedding_dim: int = 4,
        num_classes: int = 3
    ):
        """
        Initialize the MLP with embeddings.
        
        Args:
            numeric_dim: Number of numeric input features
            embedding_specs: Dict mapping categorical column names to vocabulary sizes
            embedding_dim: Dimension for each embedding vector (default 4)
            num_classes: Number of output classes (3 for Low/Medium/High)
        """
        super().__init__()
        
        self.embedding_specs = embedding_specs
        self.embedding_dim = embedding_dim
        
        # Create embedding layers for each categorical column
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, embedding_dim)
            for col, vocab_size in embedding_specs.items()
        })
        
        # Total input dimension = numeric features + all embeddings concatenated
        total_embedding_dim = len(embedding_specs) * embedding_dim
        total_input_dim = numeric_dim + total_embedding_dim
        
        # MLP layers
        self.mlp = nn.Sequential(
            # First hidden layer: expand to 256 neurons
            nn.Linear(total_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second hidden layer: compress to 128 neurons
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(128, num_classes),
        )
        
        # Store column order for consistent embedding lookup
        self.cat_col_order = list(embedding_specs.keys())
    
    def forward(self, x_numeric: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x_numeric: Numeric features (batch_size x numeric_dim)
            x_cat: Categorical indices (batch_size x num_cat_cols)
            
        Returns:
            Logits for each class (batch_size x num_classes)
        """
        # Get embeddings for each categorical column
        embedded = []
        for i, col in enumerate(self.cat_col_order):
            cat_indices = x_cat[:, i].long()
            # Clamp indices to valid range (handle out-of-vocabulary)
            cat_indices = cat_indices.clamp(0, self.embeddings[col].num_embeddings - 1)
            embedded.append(self.embeddings[col](cat_indices))
        
        # Concatenate all embeddings with numeric features
        if embedded:
            embedded_concat = torch.cat(embedded, dim=1)
            x = torch.cat([x_numeric, embedded_concat], dim=1)
        else:
            x = x_numeric
        
        # Pass through MLP
        return self.mlp(x)
    
    def get_job_embedding(self, job_type_idx: int) -> np.ndarray:
        """
        Get the learned embedding vector for a specific job type.
        
        Useful for analysis: similar jobs should have similar embeddings.
        
        Args:
            job_type_idx: Integer index of the job type
            
        Returns:
            Embedding vector as numpy array
        """
        if "job_type" not in self.embeddings:
            return np.zeros(self.embedding_dim)
        
        with torch.no_grad():
            idx = torch.tensor([job_type_idx])
            return self.embeddings["job_type"](idx).numpy().flatten()


def build_model(
    numeric_dim: int, 
    embedding_info: Dict,
    num_classes: int
) -> nn.Module:
    """
    Build the MLP model with or without entity embeddings.
    
    This is a wrapper that handles both:
    - New V2 models with embeddings (when embedding_info has vocab_sizes)
    - Legacy models without embeddings (backward compatibility)
    
    Args:
        numeric_dim: Number of numeric input features
        embedding_info: Dict with vocab_sizes and embedding configuration
        num_classes: Number of output classes
        
    Returns:
        PyTorch model (either BurnoutMLPWithEmbeddings or Sequential)
    """
    vocab_sizes = embedding_info.get("vocab_sizes", {})
    embedding_dim = embedding_info.get("embedding_dim", DEFAULT_EMBEDDING_DIM)
    
    if vocab_sizes:
        # New model with entity embeddings
        print(f"Building model with entity embeddings: {vocab_sizes}")
        return BurnoutMLPWithEmbeddings(
            numeric_dim=numeric_dim,
            embedding_specs=vocab_sizes,
            embedding_dim=embedding_dim,
            num_classes=num_classes
        )
    else:
        # Fallback to simple MLP (backward compatibility)
        print("Building simple MLP without embeddings")
        return nn.Sequential(
            nn.Linear(numeric_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args: argparse.Namespace) -> None:
    """
    Main training function with entity embeddings support.
    
    Implements a standard PyTorch training loop:
    1. Forward pass: compute predictions (with embeddings + numeric)
    2. Compute loss (CrossEntropy for classification)
    3. Backward pass: compute gradients
    4. Optimizer step: update weights
    5. Validate and save best model
    """
    # Select device (GPU if available, else CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Determine classification mode (binary is default, --three-class overrides)
    is_binary = not getattr(args, 'three_class', False)
    mode_str = "BINARY (Healthy vs At-Risk)" if is_binary else "3-CLASS (Low/Med/High)"
    print(f"Classification mode: {mode_str}")
    
    # Load data with separate handling for embeddings
    X_numeric, X_cat, y, feature_cols, embedding_info = load_data(binary=is_binary)
    train_loader, val_loader, y_val = build_loaders(X_numeric, X_cat, y, args.batch_size)
    num_classes = len(np.unique(y))
    
    print(f"Numeric features: {X_numeric.shape[1]}")
    print(f"Categorical features for embedding: {X_cat.shape[1]}")
    print(f"Number of classes: {num_classes}")
    print(f"Embedding info: {embedding_info}")
    
    # Build model with embeddings
    model = build_model(
        numeric_dim=X_numeric.shape[1],
        embedding_info=embedding_info,
        num_classes=num_classes
    ).to(DEVICE)
    
    # Loss function: CrossEntropy combines LogSoftmax + NLLLoss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Track best validation loss for model selection
    best_val = float("inf")
    history = []
    
    # Model output path: add "_binary" suffix for binary models
    binary_suffix = "_binary" if is_binary else ""
    model_out_path = MODEL_DIR / f"mlp{binary_suffix}_classifier.pt"

    # ========== TRAINING LOOP ==========
    for epoch in range(1, args.epochs + 1):
        
        # ----- Training phase -----
        model.train()
        train_losses = []
        
        for x_num, x_cat, yb in train_loader:
            x_num = x_num.to(DEVICE)
            x_cat = x_cat.to(DEVICE)
            yb = yb.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass with both numeric and categorical inputs
            if isinstance(model, BurnoutMLPWithEmbeddings):
                logits = model(x_num, x_cat)
            else:
                # Fallback for simple MLP (no embeddings)
                logits = model(x_num)
            
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())

        # ----- Validation phase -----
        model.eval()
        val_losses = []
        preds = []
        
        with torch.no_grad():
            for x_num, x_cat, yb in val_loader:
                x_num = x_num.to(DEVICE)
                x_cat = x_cat.to(DEVICE)
                yb = yb.to(DEVICE)
                
                if isinstance(model, BurnoutMLPWithEmbeddings):
                    logits = model(x_num, x_cat)
                else:
                    logits = model(x_num)
                
                val_losses.append(criterion(logits, yb).item())
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        
        # Compute metrics
        val_loss = float(np.mean(val_losses))
        f1 = f1_score(y_val, preds, average="macro" if num_classes > 2 else "binary")
        
        # Log history
        history.append({
            "epoch": epoch, 
            "train_loss": float(np.mean(train_losses)), 
            "val_loss": val_loss, 
            "f1_macro": f1
        })
        
        # Print progress every 5 epochs
        if epoch % 5 == 0:
            acc = accuracy_score(y_val, preds)
            print(f"Epoch {epoch}: val_loss={val_loss:.4f} acc={acc:.3f} f1={f1:.3f}")
        
        # Save best model (lowest validation loss)
        if val_loss < best_val:
            best_val = val_loss
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "embedding_info": embedding_info,
                "scaler_path": str(SCALER_PATH),
                "model_type": "embedding" if isinstance(model, BurnoutMLPWithEmbeddings) else "simple",
                "is_binary": is_binary,
                "num_classes": num_classes,
            }, model_out_path)

    # ========== TRAINING COMPLETE ==========
    final_acc = accuracy_score(y_val, preds)
    final_f1 = f1_score(y_val, preds, average="macro" if num_classes > 2 else "binary")
    
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE - BURNOUT [{mode_str}]")
    print("=" * 60)
    print(f"Best Validation Loss: {best_val:.4f}")
    print(f"Final Accuracy: {final_acc*100:.2f}%")
    print(f"Final F1 ({'binary' if is_binary else 'macro'}): {final_f1*100:.2f}%")
    print(f"\nModel saved to: {model_out_path}")
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
