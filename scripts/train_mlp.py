"""Train a tabular MLP classifier on burnout_level.

Usage:
    python scripts/train_mlp.py --epochs 80 --batch-size 256
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

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models/saved")
TABULAR_PATH = PROCESSED_DIR / "tabular_ml_ready.parquet"
SCALER_PATH = PROCESSED_DIR / "feature_scaler.joblib"
MODEL_OUT = MODEL_DIR / "mlp_classifier.pt"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP burnout classifier")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()

def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not TABULAR_PATH.exists():
        raise FileNotFoundError("Run scripts/preprocess.py before training the MLP.")
    df = pd.read_parquet(TABULAR_PATH)
    feature_cols = [c for c in df.columns if c not in {"burnout_level", "burnout_score"}]
    X = df[feature_cols].values.astype(np.float32)
    y = df["burnout_level"].values.astype(np.int64)
    return X, y, feature_cols

def build_loaders(X: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[DataLoader, DataLoader]:
    n = len(X)
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader, y_val

def build_model(input_dim: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )

def train(args: argparse.Namespace) -> None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, feature_cols = load_data()
    train_loader, val_loader, y_val = build_loaders(X, y, args.batch_size)
    num_classes = len(np.unique(y))
    model = build_model(X.shape[1], num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses = []
            preds = []
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        val_loss = float(np.mean(val_losses))
        f1 = f1_score(y_val, preds, average="macro")
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss, "f1_macro": f1})
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: val_loss={val_loss:.4f} f1={f1:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "scaler_path": str(SCALER_PATH),
            }, MODEL_OUT)

    print("Training complete. Model saved to", MODEL_OUT)
    print("Best epoch:", min(history, key=lambda x: x["val_loss"]))

def main() -> None:
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
