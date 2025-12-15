"""
================================================================================
DEMO STATE MANAGEMENT
================================================================================
Centralizes all demo state and configuration in a single module.
Replaces fragile global variables with a proper state container.

Usage:
    from scripts.demo_state import DemoState
    
    # Initialize state
    state = DemoState.get_instance()
    
    # Access loaded components
    if state.is_pipeline_available:
        result = state.pipeline.predict(history)

Author: FDS Project Team
================================================================================
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class DemoState:
    """
    Singleton container for all demo application state.
    
    Centralizes:
    - Two-stage pipeline (GRU + LSTM)
    - Feature mappings
    - Configuration and thresholds
    - Scaler parameters
    """
    
    _instance: Optional['DemoState'] = None
    
    def __init__(self):
        # Pipeline components
        self.pipeline = None
        self.lstm_model = None
        
        # Feature mappings
        self.feature_index: Dict[str, int] = {}
        self.behavioral_features: List[str] = []
        self.mental_features: List[str] = []
        self.targets: List[str] = []
        
        # Scalers
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        
        # Configuration
        self.thresholds: Dict[str, Any] = {}
        self.job_categories: Dict[str, Any] = {}
        
        # State flags
        self._initialized = False
        self._pipeline_type: str = 'none'  # 'two_stage', 'single_model', 'none'
    
    @classmethod
    def get_instance(cls) -> 'DemoState':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing)."""
        cls._instance = None
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def is_pipeline_available(self) -> bool:
        """Check if two-stage pipeline is available."""
        return self.pipeline is not None and self._pipeline_type == 'two_stage'
    
    @property
    def is_model_available(self) -> bool:
        """Check if any model (pipeline or single) is available."""
        return self.pipeline is not None or self.lstm_model is not None
    
    @property
    def pipeline_type(self) -> str:
        """Get the type of pipeline loaded."""
        return self._pipeline_type
    
    def initialize(
        self,
        config_dir: Path,
        model_dir: Path,
        prefer_two_stage: bool = True
    ) -> bool:
        """
        Initialize the demo state with models and configuration.
        
        Args:
            config_dir: Path to config directory
            model_dir: Path to models/saved directory
            prefer_two_stage: Try to load two-stage pipeline first
            
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.debug("DemoState already initialized")
            return True
        
        # Load configuration
        self._load_config(config_dir)
        
        # Try to load models
        if prefer_two_stage:
            success = self._try_load_two_stage_pipeline(model_dir)
            if not success:
                success = self._try_load_single_model(model_dir)
        else:
            success = self._try_load_single_model(model_dir)
        
        self._initialized = success
        return success
    
    def _load_config(self, config_dir: Path):
        """Load configuration files."""
        try:
            thresholds_path = config_dir / "thresholds.json"
            if thresholds_path.exists():
                with open(thresholds_path, 'r') as f:
                    self.thresholds = json.load(f)
                logger.debug("Loaded thresholds.json")
        except Exception as e:
            logger.warning(f"Failed to load thresholds.json: {e}")
        
        try:
            jobs_path = config_dir / "job_categories.json"
            if jobs_path.exists():
                with open(jobs_path, 'r') as f:
                    self.job_categories = json.load(f)
                logger.debug("Loaded job_categories.json")
        except Exception as e:
            logger.warning(f"Failed to load job_categories.json: {e}")
    
    def _try_load_two_stage_pipeline(self, model_dir: Path) -> bool:
        """Try to load the two-stage GRU+LSTM pipeline."""
        gru_path = model_dir / "best_behavioral_model.pt"
        lstm_path = model_dir / "mental_health_lstm.pt"
        
        if not (gru_path.exists() and lstm_path.exists()):
            logger.info("Two-stage checkpoints not found, skipping pipeline load")
            return False
        
        try:
            from scripts.checkpoint_utils import load_pipeline
            
            self.pipeline = load_pipeline(str(gru_path), str(lstm_path))
            self._pipeline_type = 'two_stage'
            
            # Populate feature mappings
            self.behavioral_features = self.pipeline.behavioral_features
            self.mental_features = self.pipeline.mental_features
            self.targets = self.pipeline.targets
            self.feature_index = {
                name: idx for idx, name in enumerate(self.mental_features)
            }
            
            # Get scalers from pipeline
            self.scaler_mean = self.pipeline.lstm_meta['scaler_mean']
            self.scaler_std = self.pipeline.lstm_meta['scaler_std']
            
            # Also store LSTM model for single-model fallback
            self.lstm_model = self.pipeline.lstm_model
            
            logger.info(f"Loaded two-stage pipeline: {len(self.behavioral_features)} → {len(self.targets)} targets")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load two-stage pipeline: {e}")
            return False
    
    def _try_load_single_model(self, model_dir: Path) -> bool:
        """Try to load just the LSTM model."""
        lstm_path = model_dir / "mental_health_lstm.pt"
        
        if not lstm_path.exists():
            logger.warning(f"LSTM checkpoint not found: {lstm_path}")
            return False
        
        try:
            from scripts.checkpoint_utils import load_lstm_model
            
            self.lstm_model, metadata = load_lstm_model(str(lstm_path))
            self._pipeline_type = 'single_model'
            
            # Populate feature mappings
            self.mental_features = metadata['feature_cols']
            self.targets = metadata['targets']
            self.feature_index = {
                name: idx for idx, name in enumerate(self.mental_features)
            }
            
            # Get scalers
            self.scaler_mean = metadata['scaler_mean']
            self.scaler_std = metadata['scaler_std']
            
            logger.info(f"Loaded single LSTM model: {len(self.mental_features)} features → {len(self.targets)} targets")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load single LSTM model: {e}")
            return False
    
    def predict(self, behavioral_data: np.ndarray, apply_amplification: bool = True) -> Dict[str, Any]:
        """
        Make a prediction using the best available method.
        
        Args:
            behavioral_data: Input behavioral data
            apply_amplification: Whether to apply presentation amplification
            
        Returns:
            Dict mapping target names to prediction results
        """
        if self._pipeline_type == 'two_stage' and self.pipeline is not None:
            return self._predict_two_stage(behavioral_data)
        elif self.lstm_model is not None:
            return self._predict_single_model(behavioral_data, apply_amplification)
        else:
            raise RuntimeError("No model available for prediction")
    
    def _predict_two_stage(self, behavioral_data: np.ndarray) -> Dict[str, Any]:
        """Run two-stage pipeline prediction."""
        # Build 7x6 behavioral history
        history = self._build_behavioral_history(behavioral_data)
        
        # Run pipeline
        result = self.pipeline.predict(history)
        
        # Convert to standard output format
        predictions = {}
        for target, info in result.get('mental_health', {}).items():
            predictions[target] = {
                'value': info.get('value', 0.0),
                'at_risk_prob': info.get('confidence', 0.5),
                'risk_level': info.get('risk_level', 'normal')
            }
        
        return predictions
    
    def _predict_single_model(self, behavioral_data: np.ndarray, apply_amplification: bool) -> Dict[str, Any]:
        """Run single LSTM model prediction."""
        import torch
        
        # Normalize input
        data = np.array(behavioral_data)
        normalized = (data - self.scaler_mean) / (self.scaler_std + 1e-8)
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        # Convert to tensor
        X = torch.FloatTensor(normalized).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = self.lstm_model(X)
        
        # Extract predictions
        predictions = {}
        for target, output in outputs.items():
            if isinstance(output, tuple):
                reg, cls = output
                value = reg.item() if hasattr(reg, 'item') else float(reg)
                at_risk_prob = torch.sigmoid(cls).item() if hasattr(cls, 'item') else 0.5
            else:
                value = output.item() if hasattr(output, 'item') else float(output)
                at_risk_prob = 0.5
            
            predictions[target] = {
                'value': float(value),
                'at_risk_prob': float(at_risk_prob)
            }
        
        return predictions
    
    def _build_behavioral_history(self, behavioral_data: np.ndarray) -> np.ndarray:
        """Build 7x6 behavioral history for GRU from input data."""
        behavioral_names = [
            'sleep_hours', 'exercise_minutes', 'steps_count',
            'screen_time_hours', 'social_interactions', 'work_hours'
        ]
        
        data = np.array(behavioral_data)
        
        if data.ndim == 2 and data.shape[1] >= 1:
            history = []
            for name in behavioral_names:
                idx = self.feature_index.get(name)
                if idx is not None and data.shape[1] > idx:
                    col = data[-7:, idx] if data.shape[0] >= 7 else np.tile(data[-1, idx], 7)
                else:
                    col = np.zeros(7)
                if len(col) < 7:
                    col = np.pad(col, (7 - len(col), 0), mode='edge')
                history.append(col)
            return np.vstack(history).T
        else:
            # Fallback: tile last row
            last = data[-1] if data.ndim == 2 else data
            return np.tile(last[:6] if len(last) >= 6 else np.zeros(6), (7, 1))
    
    def get_feature_index(self, feature_name: str) -> Optional[int]:
        """Get the index of a feature by name."""
        return self.feature_index.get(feature_name)
    
    def get_threshold(self, key: str, default: Any = None) -> Any:
        """Get a threshold value from configuration."""
        return self.thresholds.get(key, default)
    
    def get_safety_thresholds(self) -> Dict[str, Any]:
        """Get safety threshold configuration."""
        return self.thresholds.get('safety_thresholds', {})
    
    def get_amplification_config(self) -> Dict[str, Any]:
        """Get amplification configuration."""
        return self.thresholds.get('amplification', {})
