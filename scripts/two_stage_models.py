"""
Two-Stage Mental Health Prediction Pipeline
===========================================
Stage 1: GRU model (real StudentLife data) → behavioral forecast
Stage 2: LSTM model (synthetic Kaggle data) → mental health predictions

Models:
- GRU: Best for real data (R²=0.483, MAE=162.67)
- LSTM: Best for synthetic data (R²=0.98, multi-task)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class GRUModel(nn.Module):
    """GRU model for behavioral forecasting (Stage 1)"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.name = "GRU"
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class PredictionHead(nn.Module):
    """Multi-task prediction head for mental health outcomes"""
    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_regression = nn.Linear(hidden_dim, 1)
        self.fc_classification = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        regression = self.fc_regression(x)
        classification = torch.sigmoid(self.fc_classification(x))
        return regression, classification


class MentalHealthPredictor(nn.Module):
    """LSTM model for mental health prediction (Stage 2)"""
    def __init__(self, input_dim, hidden_dim=128, encoder_type="lstm", targets=None):
        super().__init__()
        self.name = "MentalHealthLSTM"
        self.encoder_type = encoder_type
        self.targets = targets or []
        
        if encoder_type == "lstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                                  batch_first=True, dropout=0.3)
        elif encoder_type == "gru":
            self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=2,
                                 batch_first=True, dropout=0.3)
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")
        
        self.prediction_heads = nn.ModuleDict({
            target: PredictionHead(hidden_dim) for target in self.targets
        })
    
    def forward(self, x):
        if self.encoder_type == "lstm":
            _, (hidden, _) = self.encoder(x)
        else:
            _, hidden = self.encoder(x)
        
        hidden = hidden[-1]
        outputs = {}
        for target, head in self.prediction_heads.items():
            reg, cls = head(hidden)
            outputs[target] = {'regression': reg.squeeze(1), 'classification': cls.squeeze(1)}
        return outputs


# ============================================================================
# TWO-STAGE PIPELINE
# ============================================================================

class TwoStagePipeline:
    """
    Two-stage prediction pipeline:
    Stage 1: GRU forecasts next-day behaviors from real data
    Stage 2: LSTM predicts mental health from forecasted behaviors
    """
    
    def __init__(self, gru_checkpoint_path, lstm_checkpoint_path):
        """
        Args:
            gru_checkpoint_path: Path to best_behavioral_model.pt (GRU)
            lstm_checkpoint_path: Path to mental_health_lstm.pt
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load GRU model (Stage 1)
        print("Loading Stage 1: GRU behavioral model...")
        self.gru_checkpoint = torch.load(gru_checkpoint_path, weights_only=False)
        self.gru_model = self._build_gru_model()
        self.gru_model.eval()
        print(f"  ✓ GRU loaded: {self.gru_checkpoint.get('model_name', 'GRU')}")
        print(f"    R²={self.gru_checkpoint.get('cv_results', {}).get('r2_mean', 0):.4f}")
        
        # Load LSTM model (Stage 2)
        print("\nLoading Stage 2: LSTM mental health model...")
        self.lstm_checkpoint = torch.load(lstm_checkpoint_path, weights_only=False)
        self.lstm_model = self._build_lstm_model()
        self.lstm_model.eval()
        print(f"  ✓ LSTM loaded: 8 prediction heads")
        
        # Feature mappings
        self.behavioral_features = self.gru_checkpoint['feature_cols']  # 6 features
        self.mental_features = self.lstm_checkpoint['feature_cols']     # 17 features
        self.targets = self.lstm_checkpoint['targets']                  # 8 targets
        
        print(f"\n📊 Pipeline ready:")
        print(f"  Stage 1: {len(self.behavioral_features)} behavioral features")
        print(f"  Stage 2: {len(self.mental_features)} mental health features")
        print(f"  Output: {len(self.targets)} predictions")
    
    def _build_gru_model(self):
        """Build GRU from checkpoint"""
        kwargs = self.gru_checkpoint['model_kwargs']
        model = GRUModel(**kwargs)
        model.load_state_dict(self.gru_checkpoint['model_state'])
        model.to(self.device)
        return model
    
    def _build_lstm_model(self):
        """Build LSTM from checkpoint"""
        model = MentalHealthPredictor(
            input_dim=len(self.mental_features),
            hidden_dim=self.lstm_checkpoint.get('hidden_dim', 128),
            encoder_type=self.lstm_checkpoint.get('encoder_type', 'lstm'),
            targets=self.targets
        )
        model.load_state_dict(self.lstm_checkpoint['model_state'])
        model.to(self.device)
        return model
    
    def _align_features(self, behavioral_pred):
        """
        Align 6 behavioral features to 17 mental health features
        
        Behavioral (6): sleep_hours, exercise_minutes, steps_count,
                       screen_time_hours, social_interactions, work_hours
        
        Mental (17): 7 days × (sleep, work, exercise, caffeine, screen_time, etc.)
        """
        # Map behavioral to mental health feature indices
        # This is a simplified mapping - adjust based on actual feature order
        mental_input = np.zeros(len(self.mental_features))
        
        # Find matching features
        for i, bf in enumerate(self.behavioral_features):
            # Map to most recent day in 7-day window (e.g., sleep_hours → sleep_hours_sunday)
            for j, mf in enumerate(self.mental_features):
                if bf.replace('_', '') in mf.replace('_', '').lower():
                    mental_input[j] = behavioral_pred[i]
                    break
        
        return mental_input
    
    def predict(self, behavioral_history):
        """
        Make two-stage prediction
        
        Args:
            behavioral_history: np.array of shape (7, 6) - last 7 days of behaviors
        
        Returns:
            dict with:
                - behavioral_forecast: Predicted next-day behaviors (6 features)
                - mental_health: Mental health predictions (8 targets)
                - confidence: Prediction confidence scores
        """
        # Stage 1: Forecast next day behaviors
        X_gru = self._normalize_gru_input(behavioral_history)
        X_tensor = torch.FloatTensor(X_gru).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_norm = self.gru_model(X_tensor).cpu().numpy()[0]
        
        behavioral_forecast = self._denormalize_gru_output(pred_norm)
        
        # Stage 2: Predict mental health
        mental_input = self._align_features(behavioral_forecast)
        X_lstm = self._normalize_lstm_input(mental_input)
        X_lstm_tensor = torch.FloatTensor(X_lstm).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.lstm_model(X_lstm_tensor)
        
        # Denormalize mental health predictions
        mental_health = {}
        for target in self.targets:
            reg_norm = predictions[target]['regression'].cpu().item()
            confidence = predictions[target]['classification'].cpu().item()
            
            value = self._denormalize_lstm_output(reg_norm, target)
            mental_health[target] = {
                'value': value,
                'confidence': confidence,
                'risk_level': self._assess_risk(target, value)
            }
        
        return {
            'behavioral_forecast': dict(zip(self.behavioral_features, behavioral_forecast)),
            'mental_health': mental_health,
            'stage1_confidence': self._compute_gru_confidence(behavioral_history),
            'pipeline_version': '2.0_GRU+LSTM'
        }
    
    def _normalize_gru_input(self, X):
        """Normalize behavioral history"""
        mean = np.array(self.gru_checkpoint['scaler_mean_X'])
        std = np.array(self.gru_checkpoint['scaler_std_X'])
        return (X - mean) / std
    
    def _denormalize_gru_output(self, y_norm):
        """Denormalize behavioral forecast"""
        mean = np.array(self.gru_checkpoint['scaler_mean_y'])
        std = np.array(self.gru_checkpoint['scaler_std_y'])
        return y_norm * std + mean
    
    def _normalize_lstm_input(self, X):
        """Normalize mental health input"""
        mean = np.array(self.lstm_checkpoint['scaler_mean'])
        std = np.array(self.lstm_checkpoint['scaler_std'])
        return (X - mean) / std
    
    def _denormalize_lstm_output(self, y_norm, target):
        """Denormalize mental health prediction"""
        idx = self.targets.index(target)
        mean = self.lstm_checkpoint['scaler_mean'][idx]
        std = self.lstm_checkpoint['scaler_std'][idx]
        return y_norm * std + mean
    
    def _compute_gru_confidence(self, history):
        """Compute confidence based on data quality"""
        completeness = 1.0 - (np.isnan(history).sum() / history.size)
        variance = np.nanstd(history) / (np.nanmean(history) + 1e-8)
        return float(np.clip(completeness * (1 - min(variance, 0.5)), 0, 1))
    
    def _assess_risk(self, target, value):
        """Assess risk level based on clinical thresholds"""
        thresholds = {
            'pss_score': {'high': 20, 'moderate': 14},
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
            if value < t['low']: return 'high_risk'
            elif value < t['moderate']: return 'moderate_risk'
            else: return 'low_risk'
        # Regular targets (lower is better)
        else:
            if value >= t['high']: return 'high_risk'
            elif value >= t['moderate']: return 'moderate_risk'
            else: return 'low_risk'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_pipeline(gru_path='models/saved/best_behavioral_model.pt',
                 lstm_path='models/saved/mental_health_lstm.pt'):
    """
    Convenience function to load the two-stage pipeline
    
    Args:
        gru_path: Path to GRU checkpoint
        lstm_path: Path to LSTM checkpoint
    
    Returns:
        TwoStagePipeline instance
    """
    if not os.path.exists(gru_path):
        raise FileNotFoundError(f"GRU model not found: {gru_path}")
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"LSTM model not found: {lstm_path}")
    
    return TwoStagePipeline(gru_path, lstm_path)


def create_sample_history():
    """Create sample 7-day behavioral history for testing"""
    return np.array([
        [7.0, 30.0, 5000.0, 6.0, 5.0, 5.0],  # Day 1
        [6.5, 45.0, 7000.0, 7.0, 8.0, 6.0],  # Day 2
        [7.5, 60.0, 8000.0, 5.0, 6.0, 5.5],  # Day 3
        [6.0, 20.0, 4000.0, 8.0, 3.0, 7.0],  # Day 4
        [8.0, 90.0, 10000.0, 4.0, 10.0, 4.0], # Day 5
        [7.0, 30.0, 6000.0, 6.0, 7.0, 5.0],  # Day 6
        [6.5, 40.0, 5500.0, 7.0, 5.0, 6.0],  # Day 7
    ])


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TWO-STAGE PIPELINE TEST")
    print("="*70)
    
    # Load pipeline
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nMake sure you have:")
        print("  1. models/saved/best_behavioral_model.pt (from Kaggle experiment)")
        print("  2. models/saved/mental_health_lstm.pt (synthetic model)")
        exit(1)
    
    # Create sample data
    history = create_sample_history()
    print("\n📊 Input: 7 days of behavioral data")
    print(f"   Shape: {history.shape}")
    
    # Make prediction
    print("\n🔮 Running two-stage prediction...")
    result = pipeline.predict(history)
    
    # Display results
    print("\n" + "="*70)
    print("STAGE 1: BEHAVIORAL FORECAST (GRU)")
    print("="*70)
    for feature, value in result['behavioral_forecast'].items():
        print(f"  {feature}: {value:.2f}")
    print(f"\n  Confidence: {result['stage1_confidence']:.1%}")
    
    print("\n" + "="*70)
    print("STAGE 2: MENTAL HEALTH PREDICTIONS (LSTM)")
    print("="*70)
    for target, pred in result['mental_health'].items():
        risk = pred['risk_level'].replace('_', ' ').upper()
        print(f"  {target}: {pred['value']:.2f} ({pred['confidence']:.1%} confidence) - {risk}")
    
    print("\n✅ Two-stage pipeline test complete!")
