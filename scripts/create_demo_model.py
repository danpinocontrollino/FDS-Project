"""
Create a demo LSTM model for Streamlit app.
This creates a minimal working model with correct architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

# ============================================================================
# MODEL ARCHITECTURE (same as demo_app.py)
# ============================================================================

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
        self.classification = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        reg = self.regression(h).squeeze(-1)
        cls = self.classification(h).squeeze(-1)
        return reg, cls

class MentalHealthPredictor(nn.Module):
    """Multi-target mental health prediction model."""
    
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
        
        self.shared_repr = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.heads = nn.ModuleDict({
            target: PredictionHead(hidden_dim, hidden_dim // 2)
            for target in self.targets
        })
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to fixed representation."""
        if self.encoder_type in ["lstm", "gru"]:
            out, _ = self.encoder(x)
            return out[:, -1, :]
        else:  # transformer
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.encoder(x)
            return x.mean(dim=1)
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        h = self.encode(x)
        h = self.shared_repr(h)
        
        outputs = {}
        for target in self.targets:
            reg, cls = self.heads[target](h)
            outputs[target] = (reg, cls)
        
        return outputs

# ============================================================================
# CREATE DEMO MODEL
# ============================================================================

def create_demo_model():
    """Create and save a demo LSTM model."""
    
    # Feature list
    feature_cols = [
        'sleep_hours', 'sleep_quality', 'work_hours', 'meetings_count',
        'tasks_completed', 'emails_received', 'commute_minutes',
        'exercise_minutes', 'steps_count', 'caffeine_mg', 'alcohol_units',
        'screen_time_hours', 'social_interactions', 'outdoor_time_minutes',
        'diet_quality', 'work_pressure', 'weather_mood_impact'
    ]
    
    # Target list (8 targets as per demo_app)
    targets = [
        'stress_level', 'mood_score', 'energy_level', 'focus_score',
        'perceived_stress_scale', 'anxiety_score', 'depression_score', 'job_satisfaction'
    ]
    
    # Create model
    model = MentalHealthPredictor(
        input_dim=len(feature_cols),
        hidden_dim=128,
        num_layers=2,
        encoder_type="lstm",
        targets=targets
    )
    
    # Create checkpoint with metadata
    checkpoint = {
        'model_state': model.state_dict(),
        'feature_cols': feature_cols,
        'targets': targets,
        'hidden_dim': 128,
        'num_layers': 2,
        'model_type': 'lstm',
        'input_dim': len(feature_cols),
        # Dummy normalization parameters (centered at 5, scale 1)
        'scaler_mean': np.array([7.0, 7.0, 8.0, 3.0, 6.0, 50.0, 20.0, 30.0, 5000.0, 200.0, 0.5, 6.0, 5.0, 30.0, 7.0, 5.0, 0.0]),
        'scaler_std': np.array([2.0, 2.0, 2.0, 2.0, 3.0, 20.0, 15.0, 20.0, 2000.0, 100.0, 1.0, 3.0, 3.0, 20.0, 2.0, 2.0, 1.0]),
    }
    
    # Save model
    model_dir = Path("models/saved")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "mental_health_lstm.pt"
    torch.save(checkpoint, model_path)
    
    print(f"✓ Demo model created: {model_path}")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Targets: {len(targets)}")
    print(f"  - Architecture: LSTM (128 hidden, 2 layers)")
    
    # Test loading
    print("\nTesting model load...")
    try:
        loaded = torch.load(model_path, weights_only=False)
        print(f"✓ Model loads successfully")
        print(f"  - Features: {len(loaded['feature_cols'])}")
        print(f"  - Targets: {loaded['targets']}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")

if __name__ == "__main__":
    create_demo_model()
