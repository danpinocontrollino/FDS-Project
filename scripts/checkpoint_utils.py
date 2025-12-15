"""
================================================================================
ROBUST CHECKPOINT LOADING UTILITIES
================================================================================
Provides unified, error-tolerant checkpoint loading for PyTorch models.
Handles CUDA→CPU mapping, state_dict normalization, and schema validation.

Usage:
    from scripts.checkpoint_utils import robust_load_checkpoint, load_gru_model, load_lstm_model
    
    # Load any checkpoint with normalized keys
    ckpt = robust_load_checkpoint('models/saved/mental_health_lstm.pt')
    
    # Load ready-to-use models
    gru_model, gru_meta = load_gru_model('models/saved/best_behavioral_model.pt')
    lstm_model, lstm_meta = load_lstm_model('models/saved/mental_health_lstm.pt')

Author: FDS Project Team
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class CheckpointLoadError(Exception):
    """Raised when a checkpoint cannot be loaded or validated."""
    pass


def robust_load_checkpoint(
    path: str,
    map_location: str = 'cpu',
    required_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint with robust error handling.
    
    Features:
    - Always loads on CPU by default (avoids CUDA deserialization errors)
    - Normalizes state_dict key names (strips 'module.' prefix from DDP)
    - Validates required keys are present
    - Returns normalized dict with consistent key names
    
    Args:
        path: Path to checkpoint file (.pt or .pth)
        map_location: Device to load weights to (default: 'cpu')
        required_keys: List of keys that must be present in checkpoint
        
    Returns:
        Normalized checkpoint dictionary
        
    Raises:
        CheckpointLoadError: If loading fails or required keys missing
    """
    path = Path(path)
    
    if not path.exists():
        raise CheckpointLoadError(f"Checkpoint file not found: {path}")
    
    try:
        checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)
    except Exception as e:
        raise CheckpointLoadError(f"Failed to load checkpoint {path}: {e}")
    
    # Handle case where checkpoint is just a state_dict
    if not isinstance(checkpoint, dict):
        raise CheckpointLoadError(f"Checkpoint is not a dict: {type(checkpoint)}")
    
    # Normalize state_dict keys (strip 'module.' prefix from DataParallel/DDP)
    if 'model_state' in checkpoint:
        checkpoint['model_state'] = _normalize_state_dict(checkpoint['model_state'])
    if 'model_state_dict' in checkpoint:
        checkpoint['model_state'] = _normalize_state_dict(checkpoint['model_state_dict'])
        
    # Validate required keys
    if required_keys:
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise CheckpointLoadError(
                f"Checkpoint {path.name} missing required keys: {missing}. "
                f"Available keys: {list(checkpoint.keys())}"
            )
    
    logger.debug(f"Loaded checkpoint {path.name} with keys: {list(checkpoint.keys())}")
    return checkpoint


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip 'module.' prefix from state_dict keys (DDP artifact)."""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        normalized[new_key] = value
    return normalized


def get_checkpoint_info(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract standardized metadata from a checkpoint.
    
    Returns dict with:
        - model_type: str (e.g., 'lstm', 'gru', 'transformer')
        - input_dim: int
        - hidden_dim: int
        - targets: list
        - feature_cols: list
        - has_scalers: bool
    """
    info = {
        'model_type': checkpoint.get('model_type') or checkpoint.get('model_name', 'unknown'),
        'input_dim': checkpoint.get('model_kwargs', {}).get('input_dim'),
        'hidden_dim': checkpoint.get('hidden_dim') or checkpoint.get('model_kwargs', {}).get('hidden_dim', 128),
        'targets': checkpoint.get('targets', []),
        'feature_cols': checkpoint.get('feature_cols', []),
        'has_scalers': 'scaler_mean' in checkpoint or 'scaler_mean_X' in checkpoint,
    }
    
    # Infer input_dim from feature_cols if not explicit
    if info['input_dim'] is None and info['feature_cols']:
        info['input_dim'] = len(info['feature_cols'])
        
    return info


# ============================================================================
# GRU MODEL LOADING (Stage 1 - Behavioral Forecasting)
# ============================================================================

class GRUModel(nn.Module):
    """GRU model for behavioral forecasting (Stage 1)."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.name = "GRU"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def load_gru_model(
    path: str,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a GRU behavioral model from checkpoint.
    
    Args:
        path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Tuple of (model, metadata_dict)
        metadata_dict contains: feature_cols, scaler_mean_X, scaler_std_X, etc.
    """
    checkpoint = robust_load_checkpoint(
        path, 
        map_location=device,
        required_keys=['model_state', 'model_kwargs']
    )
    
    # Build model
    kwargs = checkpoint['model_kwargs']
    model = GRUModel(**kwargs)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    # Extract metadata
    metadata = {
        'feature_cols': checkpoint.get('feature_cols', []),
        'scaler_mean_X': np.array(checkpoint.get('scaler_mean_X', np.zeros(kwargs['input_dim']))),
        'scaler_std_X': np.array(checkpoint.get('scaler_std_X', np.ones(kwargs['input_dim']))),
        'scaler_mean_y': np.array(checkpoint.get('scaler_mean_y', np.zeros(kwargs['input_dim']))),
        'scaler_std_y': np.array(checkpoint.get('scaler_std_y', np.ones(kwargs['input_dim']))),
        'cv_results': checkpoint.get('cv_results', {}),
    }
    
    logger.info(f"Loaded GRU model: input_dim={kwargs['input_dim']}, hidden_dim={kwargs.get('hidden_dim', 64)}")
    return model, metadata


# ============================================================================
# LSTM MODEL LOADING (Stage 2 - Mental Health Prediction)
# ============================================================================

class PredictionHead(nn.Module):
    """Single prediction head for one target."""
    
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
    """Multi-target mental health prediction model (Stage 2)."""
    
    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = "lstm",
        targets: Optional[List[str]] = None,
    ):
        super().__init__()
        self.targets = targets or []
        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
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
        """Encode sequence to fixed representation."""
        if self.encoder_type == "lstm":
            out, _ = self.encoder(x)
        else:
            out, _ = self.encoder(x)
        return out[:, -1, :]  # Last timestep
    
    def forward(self, x: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning dict of (regression, classification) tuples."""
        h = self.encode(x)
        h = self.shared_repr(h)
        
        outputs = {}
        for target in self.targets:
            reg, cls = self.heads[target](h)
            outputs[target] = (reg, cls)
        
        return outputs


def load_lstm_model(
    path: str,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load an LSTM mental health model from checkpoint.
    
    Args:
        path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Tuple of (model, metadata_dict)
        metadata_dict contains: feature_cols, targets, scaler_mean, scaler_std, etc.
    """
    checkpoint = robust_load_checkpoint(
        path,
        map_location=device,
        required_keys=['model_state']
    )
    
    # Extract model parameters
    feature_cols = checkpoint.get('feature_cols', [])
    targets = checkpoint.get('targets', [])
    hidden_dim = checkpoint.get('hidden_dim', 128)
    encoder_type = checkpoint.get('model_type', 'lstm')
    
    # Build model
    model = MentalHealthPredictor(
        input_dim=len(feature_cols) if feature_cols else 17,
        hidden_dim=hidden_dim,
        num_layers=2,
        encoder_type=encoder_type,
        targets=targets
    )
    
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    # Extract metadata  
    metadata = {
        'feature_cols': feature_cols,
        'targets': targets,
        'scaler_mean': np.array(checkpoint.get('scaler_mean', np.zeros(len(feature_cols)))),
        'scaler_std': np.array(checkpoint.get('scaler_std', np.ones(len(feature_cols)))),
        'risk_thresholds': checkpoint.get('risk_thresholds', {}),
        'hidden_dim': hidden_dim,
        'encoder_type': encoder_type,
    }
    
    logger.info(f"Loaded LSTM model: input_dim={len(feature_cols)}, targets={len(targets)}")
    return model, metadata


# ============================================================================
# TWO-STAGE PIPELINE
# ============================================================================

class TwoStagePipeline:
    """
    Unified two-stage prediction pipeline (GRU → LSTM).
    
    Stage 1: GRU forecasts next-day behavioral metrics
    Stage 2: LSTM predicts mental health from forecasted behaviors
    """
    
    def __init__(
        self,
        gru_path: str,
        lstm_path: str,
        device: str = 'cpu'
    ):
        """
        Initialize pipeline from checkpoint paths.
        
        Args:
            gru_path: Path to GRU (Stage 1) checkpoint
            lstm_path: Path to LSTM (Stage 2) checkpoint
            device: Device to run inference on
        """
        self.device = device
        
        # Load Stage 1 (GRU)
        self.gru_model, self.gru_meta = load_gru_model(gru_path, device)
        
        # Load Stage 2 (LSTM)
        self.lstm_model, self.lstm_meta = load_lstm_model(lstm_path, device)
        
        # Feature mappings
        self.behavioral_features = self.gru_meta['feature_cols']
        self.mental_features = self.lstm_meta['feature_cols']
        self.targets = self.lstm_meta['targets']
        
        logger.info(f"Two-stage pipeline ready: {len(self.behavioral_features)} → {len(self.targets)} targets")
    
    def predict(self, behavioral_history: np.ndarray) -> Dict[str, Any]:
        """
        Make two-stage prediction.
        
        Args:
            behavioral_history: np.array of shape (7, 6) - last 7 days of behaviors
            
        Returns:
            Dict with behavioral_forecast, mental_health predictions, and metadata
        """
        # Ensure correct shape
        history = np.array(behavioral_history)
        if history.ndim == 1:
            history = np.tile(history, (7, 1))
        if history.shape[0] < 7:
            # Pad with first row repeated
            padding = np.tile(history[0:1], (7 - history.shape[0], 1))
            history = np.vstack([padding, history])
        
        # Stage 1: GRU behavioral forecast
        behavioral_forecast = self._run_gru(history[-7:])
        
        # Stage 2: LSTM mental health prediction
        mental_input = self._align_features(behavioral_forecast)
        mental_health = self._run_lstm(mental_input)
        
        return {
            'behavioral_forecast': dict(zip(self.behavioral_features, behavioral_forecast)),
            'mental_health': mental_health,
            'stage1_confidence': self._compute_confidence(history),
            'pipeline_version': '2.1_robust'
        }
    
    def _run_gru(self, history: np.ndarray) -> np.ndarray:
        """Run Stage 1 GRU model."""
        # Normalize input
        mean_X = self.gru_meta['scaler_mean_X']
        std_X = self.gru_meta['scaler_std_X']
        
        # Handle 1D vs 2D scalers
        if mean_X.ndim == 1:
            X_norm = (history - mean_X) / (std_X + 1e-8)
        else:
            X_norm = (history - mean_X[:history.shape[0]]) / (std_X[:history.shape[0]] + 1e-8)
        
        # Run model
        X_tensor = torch.FloatTensor(X_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_norm = self.gru_model(X_tensor).cpu().numpy()[0]
        
        # Denormalize output
        mean_y = self.gru_meta['scaler_mean_y']
        std_y = self.gru_meta['scaler_std_y']
        return pred_norm * std_y + mean_y
    
    def _align_features(self, behavioral_forecast: np.ndarray) -> np.ndarray:
        """Align 6 behavioral features to LSTM input features."""
        mental_input = np.zeros(len(self.mental_features))
        
        for i, bf in enumerate(self.behavioral_features):
            for j, mf in enumerate(self.mental_features):
                # Fuzzy match: check if behavioral feature name is in mental feature name
                bf_clean = bf.replace('_', '').lower()
                mf_clean = mf.replace('_', '').lower()
                if bf_clean in mf_clean or mf_clean.startswith(bf_clean[:5]):
                    mental_input[j] = behavioral_forecast[i]
                    break
        
        return mental_input
    
    def _run_lstm(self, mental_input: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run Stage 2 LSTM model."""
        # Normalize input
        mean = self.lstm_meta['scaler_mean']
        std = self.lstm_meta['scaler_std']
        X_norm = (mental_input - mean) / (std + 1e-8)
        
        # Run model (add batch and sequence dimensions)
        X_tensor = torch.FloatTensor(X_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.lstm_model(X_tensor)
        
        # Extract predictions
        results = {}
        for target in self.targets:
            if target in outputs:
                output = outputs[target]
                if isinstance(output, tuple):
                    reg, cls = output
                    value = reg.item() if hasattr(reg, 'item') else float(reg)
                    confidence = torch.sigmoid(cls).item() if hasattr(cls, 'item') else 0.5
                else:
                    value = output.item() if hasattr(output, 'item') else float(output)
                    confidence = 0.5
                
                results[target] = {
                    'value': float(value),
                    'confidence': float(confidence),
                    'risk_level': self._assess_risk(target, value)
                }
        
        return results
    
    def _compute_confidence(self, history: np.ndarray) -> float:
        """Compute data quality-based confidence score."""
        completeness = 1.0 - (np.isnan(history).sum() / max(history.size, 1))
        variance = np.nanstd(history) / (np.nanmean(np.abs(history)) + 1e-8)
        return float(np.clip(completeness * (1 - min(variance, 0.5)), 0, 1))
    
    def _assess_risk(self, target: str, value: float) -> str:
        """Assess risk level based on clinical thresholds."""
        thresholds = {
            'perceived_stress_scale': {'high': 20, 'moderate': 14},
            'anxiety_score': {'high': 15, 'moderate': 10},
            'depression_score': {'high': 20, 'moderate': 15},
            'job_satisfaction': {'low': 5, 'moderate': 7},
            'stress_level': {'high': 7, 'moderate': 5},
            'mood_score': {'low': 4, 'moderate': 6},
            'energy_level': {'low': 4, 'moderate': 6},
            'focus_score': {'low': 4, 'moderate': 6}
        }
        
        if target not in thresholds:
            return 'normal'
        
        t = thresholds[target]
        
        # Inverted targets (higher is better)
        if target in ['job_satisfaction', 'mood_score', 'energy_level', 'focus_score']:
            if value < t.get('low', 4):
                return 'high_risk'
            elif value < t.get('moderate', 6):
                return 'moderate_risk'
            else:
                return 'low_risk'
        # Regular targets (lower is better)
        else:
            if value >= t.get('high', 20):
                return 'high_risk'
            elif value >= t.get('moderate', 14):
                return 'moderate_risk'
            else:
                return 'low_risk'


def load_pipeline(
    gru_path: str = 'models/saved/best_behavioral_model.pt',
    lstm_path: str = 'models/saved/mental_health_lstm.pt',
    device: str = 'cpu'
) -> TwoStagePipeline:
    """
    Convenience function to load the two-stage pipeline.
    
    Raises:
        CheckpointLoadError: If either checkpoint cannot be loaded
    """
    gru_path = Path(gru_path)
    lstm_path = Path(lstm_path)
    
    if not gru_path.exists():
        raise CheckpointLoadError(f"GRU model not found: {gru_path}")
    if not lstm_path.exists():
        raise CheckpointLoadError(f"LSTM model not found: {lstm_path}")
    
    return TwoStagePipeline(str(gru_path), str(lstm_path), device)
