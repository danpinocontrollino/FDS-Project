"""
================================================================================
MODEL ARCHITECTURE DEFINITIONS
================================================================================
Shared model definitions for mental health prediction system.

Contains:
  - PositionalEncoding: Transformer positional embeddings
  - PredictionHead: Single-target prediction head
  - MentalHealthPredictor: Multi-task LSTM/GRU/Transformer model

Usage:
    from model_definitions import MentalHealthPredictor
    
    model = MentalHealthPredictor(
        input_dim=17,
        hidden_dim=128,
        num_layers=2,
        encoder_type="lstm",
        targets=["stress_level", "mood_score", ...]
    )

Author: FDS Project Team
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class PredictionHead(nn.Module):
    """Single prediction head for one target (regression only)."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.regression = nn.Linear(hidden_dim, 1)
        self.classification = nn.Linear(hidden_dim, 1)  # Single output for regression
    
    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        reg = self.regression(h).squeeze(-1)
        cls = self.classification(h).squeeze(-1)  # Same as regression for "both" task
        return reg, cls


class MentalHealthPredictor(nn.Module):
    """Multi-target mental health prediction model.
    
    Architecture:
      1. Encoder (LSTM/GRU/Transformer) processes 7-day sequences
      2. Shared representation layer
      3. Multiple prediction heads (one per target)
    
    Args:
        input_dim: Number of input features per timestep (default: 17)
        hidden_dim: Hidden layer dimension (default: 128)
        num_layers: Number of encoder layers (default: 2)
        encoder_type: Type of encoder ("lstm", "gru", or "transformer")
        targets: List of target variable names (e.g., ["stress_level", "mood_score"])
    
    Example:
        >>> model = MentalHealthPredictor(
        ...     input_dim=17,
        ...     hidden_dim=128,
        ...     num_layers=2,
        ...     encoder_type="lstm",
        ...     targets=["stress_level", "mood_score", "anxiety_level"]
        ... )
        >>> x = torch.randn(32, 7, 17)  # batch=32, seq_len=7, features=17
        >>> outputs = model(x)
        >>> print(outputs["stress_level"][0].shape)  # torch.Size([32])
    """
    
    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = "lstm",
        targets = None,
    ):
        super().__init__()
        
        self.targets = targets or []
        self.encoder_type = encoder_type
        
        # Build encoder
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
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Shared representation layer
        self.shared_repr = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Prediction heads (one per target)
        self.heads = nn.ModuleDict({
            target: PredictionHead(hidden_dim, hidden_dim // 2)
            for target in self.targets
        })
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to fixed representation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Encoded representation of shape (batch, hidden_dim)
        """
        if self.encoder_type in ["lstm", "gru"]:
            out, _ = self.encoder(x)
            return out[:, -1, :]  # Last timestep
        else:  # transformer
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.encoder(x)
            return x.mean(dim=1)  # Mean pooling
    
    def forward(self, x: torch.Tensor):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Dictionary mapping target names to (regression, classification) tuples.
            For regression-only tasks, both outputs are identical.
        """
        h = self.encode(x)
        h = self.shared_repr(h)
        
        outputs = {}
        for target in self.targets:
            reg, cls = self.heads[target](h)
            outputs[target] = (reg, cls)
        
        return outputs