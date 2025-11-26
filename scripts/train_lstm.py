"""Train sequence models (LSTM/GRU/CNN) on 7-day behavioral windows.

Usage:
    python scripts/train_lstm.py --model lstm --window 7 --epochs 40
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

PROCESSED_DIR = Path("data/processed")
DAILY_PATH = PROCESSED_DIR / "daily_with_burnout.parquet"
MODEL_DIR = Path("models/saved")

FEATURE_COLS = [
    "sleep_hours",
    "sleep_quality",
    "work_hours",
    "meetings_count",
    "tasks_completed",
    "exercise_minutes",
    "steps_count",
    "caffeine_mg",
    "alcohol_units",
    "screen_time_hours",
    "stress_level",
    "mood_score",
    "energy_level",
    "focus_score",
    "work_pressure",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train burnout sequence models")
    parser.add_argument("--model", choices=["lstm", "gru", "cnn"], default="lstm")
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()

def load_daily(window: int) -> Tuple[np.ndarray, np.ndarray]:
    if not DAILY_PATH.exists():
        raise FileNotFoundError("Run scripts/create_burnout_labels.py first to generate daily_with_burnout.parquet")
    daily = pd.read_parquet(DAILY_PATH)
    daily = daily.sort_values(["user_id", "date"]).reset_index(drop=True)

    sequences, labels = [], []
    for uid, group in daily.groupby("user_id"):
        feats = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        labs = group["burnout_level"].to_numpy(dtype=np.int64)
        if len(group) < window:
            continue
        for idx in range(window, len(group) + 1):
            seq = feats[idx - window: idx]
            label = labs[idx - 1]
            if np.isnan(seq).any():
                continue
            sequences.append(seq)
            labels.append(label)
    return np.stack(sequences), np.array(labels)

def build_loaders(X: np.ndarray, y: np.ndarray, batch_size: int) -> tuple[DataLoader, DataLoader, np.ndarray]:
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    return train_loader, val_loader, y_val

def build_model(model_type: str, input_dim: int, seq_len: int, num_classes: int) -> nn.Module:
    if model_type in {"lstm", "gru"}:
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.GRU
        class SequenceNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = rnn_cls(input_dim, 128, num_layers=2, batch_first=True, dropout=0.2)
                self.head = nn.Sequential(
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :])
        return SequenceNet()
    if model_type == "cnn":
        class CNN1D(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.transpose(1, 2)
                feats = self.conv(x)
                return self.head(feats)
        return CNN1D()
    raise ValueError("Unknown model type")

def train(args: argparse.Namespace) -> None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_X, seq_y = load_daily(args.window)
    train_loader, val_loader, y_val = build_loaders(seq_X, seq_y, args.batch_size)
    num_classes = len(np.unique(seq_y))
    model = build_model(args.model, seq_X.shape[2], seq_X.shape[1], num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    history = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{args.model}_sequence.pt"

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
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "model_type": args.model,
                "feature_cols": FEATURE_COLS,
                "window": args.window,
            }, model_path)

    accuracy = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    print("Finished training.")
    print({"val_loss": best_val, "accuracy": accuracy, "f1_macro": f1})
    print("Saved best model to", model_path)


def main() -> None:
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
