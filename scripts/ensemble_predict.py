"""
================================================================================
ENSEMBLE BURNOUT PREDICTOR
================================================================================
Combines predictions from multiple trained models for more robust predictions.

ENSEMBLE METHODS:
----------------
1. Voting: Each model votes for a class, majority wins
2. Averaging: Average the probability outputs from all models
3. Weighted: Weight models by their validation accuracy

WHY ENSEMBLE?
-------------
- Reduces variance: Different models make different errors
- More robust: Less sensitive to individual model quirks
- Often improves accuracy by 1-3% over best single model

MODELS COMBINED:
---------------
- LSTM: Sequential pattern learning
- Transformer: Attention-based global patterns
- MLP: Tabular feature interactions

Usage:
    from scripts.ensemble_predict import EnsemblePredictor
    ensemble = EnsemblePredictor()
    probs = ensemble.predict(features_dict)  # Returns class probabilities

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import torch
from torch import nn
import joblib


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = Path("models/saved")
PROCESSED_DIR = Path("data/processed")


# ============================================================================
# MODEL LOADERS
# ============================================================================

class ModelLoader:
    """Utility class to load different model types."""
    
    @staticmethod
    def load_lstm(path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
        """Load LSTM/GRU/CNN sequence model."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        feature_cols = checkpoint['feature_cols']
        model_type = checkpoint.get('model_type', 'lstm')
        window = checkpoint.get('window', 7)
        
        # Recreate model architecture
        input_dim = len(feature_cols)
        
        if model_type in ('lstm', 'gru'):
            rnn_cls = nn.LSTM if model_type == 'lstm' else nn.GRU
            
            class SequenceNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.rnn = rnn_cls(input_dim, 128, 2, batch_first=True, dropout=0.2)
                    self.head = nn.Sequential(
                        nn.LayerNorm(128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 3),
                    )
                def forward(self, x):
                    out, _ = self.rnn(x)
                    return self.head(out[:, -1, :])
            
            model = SequenceNet()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(checkpoint['model_state'])
        model.to(device).eval()
        
        return model, {
            'feature_cols': feature_cols,
            'window': window,
            'model_type': model_type,
            'metrics': checkpoint.get('metrics', {}),
        }
    
    @staticmethod
    def load_transformer(path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
        """Load Transformer sequence model."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        feature_cols = checkpoint['feature_cols']
        d_model = checkpoint.get('d_model', 128)
        nhead = checkpoint.get('nhead', 4)
        num_layers = checkpoint.get('num_layers', 2)
        window = checkpoint.get('window', 7)
        
        input_dim = len(feature_cols)
        
        # Positional encoding
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=512, dropout=0.1):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)
            
            def forward(self, x):
                x = x + self.pe[:, :x.size(1), :]
                return self.dropout(x)
        
        class TransformerClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.d_model = d_model
                self.input_projection = nn.Linear(input_dim, d_model)
                self.pos_encoder = PositionalEncoding(d_model, max_len=window)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=256,
                    dropout=0.1, activation='gelu', batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Dropout(0.1),
                    nn.Linear(d_model, 3),
                )
            
            def forward(self, x):
                x = self.input_projection(x)
                x = self.pos_encoder(x)
                x = self.transformer_encoder(x)
                x = x.mean(dim=1)
                return self.head(x)
        
        model = TransformerClassifier()
        model.load_state_dict(checkpoint['model_state'])
        model.to(device).eval()
        
        return model, {
            'feature_cols': feature_cols,
            'window': window,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'metrics': checkpoint.get('metrics', {}),
        }
    
    @staticmethod
    def load_mlp(path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
        """Load MLP tabular classifier."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        feature_cols = checkpoint['feature_cols']
        
        input_dim = len(feature_cols)
        
        # MLP is saved as a Sequential directly (not wrapped in a class)
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )
        
        model.load_state_dict(checkpoint['model_state'])
        model.to(device).eval()
        
        return model, {
            'feature_cols': feature_cols,
            'scaler_path': checkpoint.get('scaler_path'),
            'metrics': checkpoint.get('metrics', {}),
        }


# ============================================================================
# ENSEMBLE PREDICTOR
# ============================================================================

class EnsemblePredictor:
    """
    Ensemble predictor combining multiple burnout models.
    
    Supports:
    - LSTM sequence model
    - Transformer sequence model  
    - MLP tabular model
    
    Methods:
    - Soft voting (average probabilities)
    - Weighted voting (by validation accuracy)
    """
    
    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        device: Optional[torch.device] = None,
        method: str = "weighted",  # "soft" or "weighted"
    ):
        """
        Initialize ensemble by loading available models.
        
        Args:
            model_dir: Directory containing saved model checkpoints
            device: PyTorch device (auto-detected if None)
            method: Ensemble method ("soft" for equal weights, "weighted" for accuracy-based)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.method = method
        self.models = {}
        self.configs = {}
        self.weights = {}
        
        # Try loading each model type
        self._load_models(model_dir)
        
        # Compute weights based on validation accuracy
        self._compute_weights()
        
        print(f"Ensemble initialized with {len(self.models)} models:")
        for name, weight in self.weights.items():
            acc = self.configs[name].get('metrics', {}).get('val_acc', 'N/A')
            acc_str = f"{acc*100:.1f}%" if isinstance(acc, float) else acc
            print(f"  - {name}: weight={weight:.3f}, val_acc={acc_str}")
    
    def _load_models(self, model_dir: Path):
        """Load all available models from directory."""
        model_files = {
            'lstm': model_dir / 'lstm_sequence.pt',
            'transformer': model_dir / 'transformer_sequence.pt',
            'mlp': model_dir / 'mlp_classifier.pt',
        }
        
        for name, path in model_files.items():
            if not path.exists():
                warnings.warn(f"Model not found: {path}")
                continue
            
            try:
                if name == 'lstm':
                    model, config = ModelLoader.load_lstm(path, self.device)
                elif name == 'transformer':
                    model, config = ModelLoader.load_transformer(path, self.device)
                elif name == 'mlp':
                    model, config = ModelLoader.load_mlp(path, self.device)
                else:
                    continue
                
                self.models[name] = model
                self.configs[name] = config
            except Exception as e:
                warnings.warn(f"Failed to load {name}: {e}")
    
    def _compute_weights(self):
        """Compute model weights based on validation accuracy."""
        if self.method == "soft":
            # Equal weights
            for name in self.models:
                self.weights[name] = 1.0 / len(self.models)
        elif self.method == "weighted":
            # Weight by validation accuracy
            accuracies = {}
            for name in self.models:
                acc = self.configs[name].get('metrics', {}).get('val_acc', 0.5)
                accuracies[name] = acc if isinstance(acc, float) else 0.5
            
            # Normalize weights
            total = sum(accuracies.values())
            for name, acc in accuracies.items():
                self.weights[name] = acc / total if total > 0 else 1.0 / len(self.models)
    
    def predict_proba(
        self,
        sequence_data: Optional[np.ndarray] = None,
        tabular_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict class probabilities using ensemble.
        
        Args:
            sequence_data: (N, window, features) array for LSTM/Transformer
            tabular_data: (N, features) array for MLP
            
        Returns:
            Probability array of shape (N, 3) for LOW/MEDIUM/HIGH
        """
        all_probs = []
        all_weights = []
        
        # Sequence models (LSTM, Transformer)
        if sequence_data is not None:
            seq_tensor = torch.from_numpy(sequence_data).float().to(self.device)
            
            for name in ['lstm', 'transformer']:
                if name not in self.models:
                    continue
                
                with torch.no_grad():
                    logits = self.models[name](seq_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    all_probs.append(probs)
                    all_weights.append(self.weights[name])
        
        # Tabular model (MLP)
        if tabular_data is not None and 'mlp' in self.models:
            tab_tensor = torch.from_numpy(tabular_data).float().to(self.device)
            
            with torch.no_grad():
                logits = self.models['mlp'](tab_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_weights.append(self.weights['mlp'])
        
        if not all_probs:
            raise ValueError("No predictions made - check input data and loaded models")
        
        # Weighted average of probabilities
        all_weights = np.array(all_weights)
        all_weights /= all_weights.sum()  # Normalize
        
        ensemble_probs = sum(w * p for w, p in zip(all_weights, all_probs))
        return ensemble_probs
    
    def predict(
        self,
        sequence_data: Optional[np.ndarray] = None,
        tabular_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict burnout class labels using ensemble.
        
        Returns:
            Array of class labels (0=LOW, 1=MEDIUM, 2=HIGH)
        """
        probs = self.predict_proba(sequence_data, tabular_data)
        return np.argmax(probs, axis=1)
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        info = {}
        for name, config in self.configs.items():
            info[name] = {
                'weight': self.weights[name],
                'features': len(config.get('feature_cols', [])),
                'metrics': config.get('metrics', {}),
            }
            if 'window' in config:
                info[name]['window'] = config['window']
        return info


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Test ensemble predictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ensemble predictor")
    parser.add_argument("--method", choices=["soft", "weighted"], default="weighted",
                        help="Ensemble method (default: weighted)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENSEMBLE BURNOUT PREDICTOR TEST")
    print("=" * 60)
    
    # Initialize ensemble
    ensemble = EnsemblePredictor(method=args.method)
    
    # Print model info
    print("\nModel Information:")
    print("-" * 40)
    info = ensemble.get_model_info()
    for name, details in info.items():
        print(f"\n{name.upper()}:")
        for key, value in details.items():
            if key == 'metrics':
                for mk, mv in value.items():
                    if mk != 'confusion_matrix':
                        print(f"  {mk}: {mv:.4f}" if isinstance(mv, float) else f"  {mk}: {mv}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Ensemble ready for predictions!")
    print("=" * 60)


if __name__ == "__main__":
    main()
