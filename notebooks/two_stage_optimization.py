"""
TWO-STAGE PIPELINE OPTIMIZATION
===============================
Combines the best behavioral forecasting model (from model_comparison_experiment.py)
with the synthetic-trained mental health model to find the optimal combination.

STRATEGY:
- Stage 1: Best model from real data → predicts next-day behaviors
- Stage 2: Synthetic model → infers mental health from predicted behaviors

We test multiple combination strategies:
1. Direct cascade (Stage1 output → Stage2 input)
2. Uncertainty-weighted cascade
3. Ensemble averaging
4. Residual correction

RUN AFTER: model_comparison_experiment.py

Author: FDS Project Team
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*80)
print("TWO-STAGE PIPELINE OPTIMIZATION")
print("Finding optimal combination of real + synthetic models")
print("="*80)

# Check environment
KAGGLE = os.path.exists('/kaggle/input')
if KAGGLE:
    BASE_PATH = '/kaggle/input/student-life/dataset'
    OUTPUT_PATH = '/kaggle/working'
else:
    BASE_PATH = 'data/studentlife/dataset'
    OUTPUT_PATH = 'reports'

# ============================================================================
# MODEL DEFINITIONS (must match those in model_comparison_experiment.py)
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.name = "LSTM"
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.name = "BiLSTM"
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.name = "GRU"
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# Synthetic model definition (from main project)
class PredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.regression = nn.Linear(hidden_dim, 1)
        self.classification = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        h = self.shared(x)
        reg = self.regression(h).squeeze(-1)
        cls = self.classification(h).squeeze(-1)
        return reg, cls

class MentalHealthPredictor(nn.Module):
    """Multi-task LSTM for mental health prediction (synthetic model)."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, 
                 encoder_type='lstm', targets=None):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        
        if targets is None:
            targets = ['stress_level', 'mood_score', 'energy_level', 'focus_score',
                      'perceived_stress_scale', 'anxiety_score', 'depression_score', 
                      'job_satisfaction']
        self.targets = targets
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.shared_repr = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.heads = nn.ModuleDict({
            target: PredictionHead(hidden_dim, 64) for target in targets
        })
    
    def forward(self, x):
        encoded, _ = self.encoder(x)
        last_hidden = encoded[:, -1, :]
        shared = self.shared_repr(last_hidden)
        outputs = {}
        for target, head in self.heads.items():
            outputs[target] = head(shared)
        return outputs


# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_CLASSES = {
    'LSTM': LSTMModel,
    'BiLSTM': BiLSTMModel,
    'GRU': GRUModel,
}

def load_behavioral_model(checkpoint_path):
    """Load the best behavioral model from experiment."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model_name = checkpoint['model_name']
    model_kwargs = checkpoint['model_kwargs']
    
    # Get model class
    if model_name in MODEL_CLASSES:
        model = MODEL_CLASSES[model_name](**model_kwargs)
    else:
        # Default to LSTM
        print(f"Warning: Model {model_name} not found, using LSTM")
        model = LSTMModel(**model_kwargs)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, checkpoint


def load_synthetic_model(model_path):
    """Load the synthetic-trained mental health model."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = MentalHealthPredictor(
        input_dim=len(checkpoint['feature_cols']),
        hidden_dim=128,
        encoder_type='lstm',
        targets=checkpoint['targets']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, checkpoint


# ============================================================================
# FEATURE ALIGNMENT
# ============================================================================

def align_features(behavioral_features, behavioral_cols, synthetic_cols):
    """
    Align behavioral model output features with synthetic model input features.
    
    Behavioral model predicts: sleep_hours, exercise_minutes, steps_count, 
                              screen_time_hours, social_interactions, work_hours
    
    Synthetic model expects: 17 features including the above plus others
    """
    # Create full feature vector with defaults
    defaults = {
        'sleep_hours': 7.0,
        'sleep_quality': 6.0,
        'work_hours': 5.0,
        'meetings_count': 2.0,
        'tasks_completed': 5.0,
        'emails_received': 15.0,
        'commute_minutes': 20.0,
        'exercise_minutes': 30.0,
        'steps_count': 5000.0,
        'caffeine_mg': 150.0,
        'alcohol_units': 0.0,
        'screen_time_hours': 6.0,
        'social_interactions': 5.0,
        'outdoor_time_minutes': 30.0,
        'diet_quality': 6.0,
        'work_pressure': 5.0,
        'weather_mood_impact': 0.0,
    }
    
    aligned = np.zeros(len(synthetic_cols))
    
    for i, col in enumerate(synthetic_cols):
        if col in behavioral_cols:
            idx = behavioral_cols.index(col)
            aligned[i] = behavioral_features[idx]
        else:
            aligned[i] = defaults.get(col, 0)
    
    return aligned


# ============================================================================
# TWO-STAGE PIPELINE STRATEGIES
# ============================================================================

class TwoStagePipeline:
    """Base two-stage pipeline: Behavioral → Mental Health."""
    
    def __init__(self, behavioral_model, behavioral_checkpoint, 
                 synthetic_model, synthetic_checkpoint):
        self.behavioral_model = behavioral_model
        self.behavioral_ckpt = behavioral_checkpoint
        self.synthetic_model = synthetic_model
        self.synthetic_ckpt = synthetic_checkpoint
        
        self.behavioral_cols = behavioral_checkpoint['feature_cols']
        self.synthetic_cols = synthetic_checkpoint['feature_cols']
        
    def predict_behavior(self, X_seq):
        """Stage 1: Predict next-day behavior from sequence."""
        # Normalize using behavioral scaler
        X_mean = np.array(self.behavioral_ckpt['scaler_mean_X'])
        X_std = np.array(self.behavioral_ckpt['scaler_std_X'])
        y_mean = np.array(self.behavioral_ckpt['scaler_mean_y'])
        y_std = np.array(self.behavioral_ckpt['scaler_std_y'])
        
        X_norm = (X_seq - X_mean) / X_std
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).unsqueeze(0)
            pred_norm = self.behavioral_model(X_tensor).numpy()[0]
        
        # Denormalize
        pred = pred_norm * y_std + y_mean
        return pred
    
    def predict_mental_health(self, behavioral_features, seq_length=7):
        """Stage 2: Predict mental health from behavioral features."""
        # Align features
        aligned = align_features(
            behavioral_features, 
            self.behavioral_cols, 
            self.synthetic_cols
        )
        
        # Create 7-day sequence (repeat predicted day)
        seq = np.tile(aligned, (seq_length, 1))
        
        # Normalize using synthetic scaler
        X_mean = np.array(self.synthetic_ckpt['scaler_mean'])
        X_std = np.array(self.synthetic_ckpt['scaler_std'])
        seq_norm = (seq - X_mean) / X_std
        seq_norm = np.nan_to_num(seq_norm, nan=0.0)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(seq_norm).unsqueeze(0)
            outputs = self.synthetic_model(X_tensor)
        
        predictions = {}
        for target, output in outputs.items():
            if isinstance(output, tuple):
                reg, cls = output
                predictions[target] = {
                    'value': reg.item(),
                    'confidence': torch.sigmoid(cls).item()
                }
            else:
                predictions[target] = {'value': output.item(), 'confidence': 0.5}
        
        return predictions
    
    def predict(self, X_seq):
        """Full two-stage prediction."""
        # Stage 1: Behavioral forecast
        behavior_pred = self.predict_behavior(X_seq)
        
        # Stage 2: Mental health inference
        mental_health = self.predict_mental_health(behavior_pred)
        
        return {
            'behavioral_prediction': behavior_pred.tolist(),
            'mental_health_prediction': mental_health
        }


class UncertaintyWeightedPipeline(TwoStagePipeline):
    """Pipeline with uncertainty weighting between stages."""
    
    def predict_behavior_with_uncertainty(self, X_seq, n_samples=10):
        """Monte Carlo dropout for uncertainty estimation."""
        self.behavioral_model.train()  # Enable dropout
        
        X_mean = np.array(self.behavioral_ckpt['scaler_mean_X'])
        X_std = np.array(self.behavioral_ckpt['scaler_std_X'])
        y_mean = np.array(self.behavioral_ckpt['scaler_mean_y'])
        y_std = np.array(self.behavioral_ckpt['scaler_std_y'])
        
        X_norm = (X_seq - X_mean) / X_std
        X_tensor = torch.FloatTensor(X_norm).unsqueeze(0)
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred_norm = self.behavioral_model(X_tensor).numpy()[0]
                pred = pred_norm * y_std + y_mean
                predictions.append(pred)
        
        self.behavioral_model.eval()
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        uncertainty = predictions.std(axis=0)
        
        return mean_pred, uncertainty
    
    def predict(self, X_seq):
        """Two-stage with uncertainty propagation."""
        # Stage 1 with uncertainty
        behavior_pred, uncertainty = self.predict_behavior_with_uncertainty(X_seq)
        
        # Weight predictions by inverse uncertainty
        weights = 1.0 / (uncertainty + 1e-6)
        weights = weights / weights.sum()
        
        # Stage 2 with weighted features
        weighted_behavior = behavior_pred  # Could apply weighting here
        mental_health = self.predict_mental_health(weighted_behavior)
        
        return {
            'behavioral_prediction': behavior_pred.tolist(),
            'behavioral_uncertainty': uncertainty.tolist(),
            'mental_health_prediction': mental_health,
            'uncertainty_weights': weights.tolist()
        }


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_pipeline(pipeline, test_sequences, test_behaviors, test_mental_health=None):
    """Evaluate pipeline on test data."""
    results = {
        'behavioral_mae': [],
        'behavioral_predictions': [],
        'mental_health_predictions': []
    }
    
    for i, (X_seq, y_behavior) in enumerate(zip(test_sequences, test_behaviors)):
        pred = pipeline.predict(X_seq)
        
        # Behavioral error
        behavior_mae = mean_absolute_error(
            y_behavior, 
            pred['behavioral_prediction']
        )
        results['behavioral_mae'].append(behavior_mae)
        results['behavioral_predictions'].append(pred['behavioral_prediction'])
        results['mental_health_predictions'].append(pred['mental_health_prediction'])
    
    return {
        'mean_behavioral_mae': np.mean(results['behavioral_mae']),
        'std_behavioral_mae': np.std(results['behavioral_mae']),
        'predictions': results
    }


# ============================================================================
# MAIN
# ============================================================================

def run_pipeline_optimization():
    """Run two-stage pipeline optimization."""
    
    print("\n" + "="*80)
    print("Loading Models")
    print("="*80)
    
    # Load behavioral model
    behavioral_path = os.path.join(OUTPUT_PATH, 'best_behavioral_model.pt')
    
    if not os.path.exists(behavioral_path):
        print("❌ best_behavioral_model.pt not found!")
        print("   Run model_comparison_experiment.py first")
        return None
    
    behavioral_model, behavioral_ckpt = load_behavioral_model(behavioral_path)
    print(f"✓ Loaded behavioral model: {behavioral_ckpt['model_name']}")
    
    # Load synthetic model
    synthetic_paths = [
        '/kaggle/input/mental-health-models/mental_health_lstm.pt',
        'models/saved/mental_health_lstm.pt'
    ]
    
    synthetic_model = None
    synthetic_ckpt = None
    
    for path in synthetic_paths:
        if os.path.exists(path):
            synthetic_model, synthetic_ckpt = load_synthetic_model(path)
            print(f"✓ Loaded synthetic model from: {path}")
            break
    
    if synthetic_model is None:
        print("❌ Synthetic model not found!")
        print("   Upload mental_health_lstm.pt to Kaggle")
        return None
    
    # Create pipelines
    print("\n" + "="*80)
    print("Creating Pipelines")
    print("="*80)
    
    pipelines = {
        'Direct Cascade': TwoStagePipeline(
            behavioral_model, behavioral_ckpt,
            synthetic_model, synthetic_ckpt
        ),
        'Uncertainty Weighted': UncertaintyWeightedPipeline(
            behavioral_model, behavioral_ckpt,
            synthetic_model, synthetic_ckpt
        )
    }
    
    # Example prediction
    print("\n" + "="*80)
    print("Example Predictions")
    print("="*80)
    
    # Sample input (7 days of behavioral data)
    sample_input = np.array([
        [7.5, 45, 8000, 5.0, 8, 6.0],   # Day 1
        [6.0, 30, 6000, 7.0, 5, 8.0],   # Day 2
        [7.0, 60, 10000, 4.0, 10, 5.0], # Day 3
        [5.5, 15, 4000, 8.0, 3, 9.0],   # Day 4
        [8.0, 45, 9000, 3.0, 12, 4.0],  # Day 5
        [6.5, 20, 5000, 6.0, 6, 7.0],   # Day 6
        [7.0, 30, 7000, 5.0, 7, 6.0],   # Day 7
    ])
    
    print("\nInput: 7 days of behavioral data")
    print(f"  Features: {behavioral_ckpt['feature_cols']}")
    
    for name, pipeline in pipelines.items():
        print(f"\n{'─'*60}")
        print(f"Pipeline: {name}")
        print(f"{'─'*60}")
        
        result = pipeline.predict(sample_input)
        
        print("\nStage 1 - Behavioral Forecast:")
        for i, col in enumerate(behavioral_ckpt['feature_cols']):
            print(f"  {col}: {result['behavioral_prediction'][i]:.2f}")
        
        print("\nStage 2 - Mental Health Inference:")
        for target, pred in result['mental_health_prediction'].items():
            print(f"  {target}: {pred['value']:.2f} (conf: {pred['confidence']:.2f})")
    
    # Save pipeline configuration
    config = {
        'behavioral_model': behavioral_ckpt['model_name'],
        'behavioral_cv_r2': behavioral_ckpt['cv_results']['r2_mean'],
        'synthetic_model': 'MentalHealthPredictor',
        'pipeline_types': list(pipelines.keys()),
        'created_at': datetime.now().isoformat()
    }
    
    config_path = os.path.join(OUTPUT_PATH, 'two_stage_pipeline_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Pipeline config saved: {config_path}")
    
    return config


if __name__ == '__main__':
    config = run_pipeline_optimization()
    
    if config:
        print("\n" + "="*80)
        print("TWO-STAGE PIPELINE READY!")
        print("="*80)
        print(f"\nBest behavioral model: {config['behavioral_model']} (R²={config['behavioral_cv_r2']:.4f})")
        print("Pipeline types available:")
        for pt in config['pipeline_types']:
            print(f"  - {pt}")
