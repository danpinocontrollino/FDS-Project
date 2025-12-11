import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is on sys.path so demo_app can be imported when this
# test is run from the scripts/ directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from demo_app import predict_mental_health

class MockModel:
    """Simple mock model that returns fixed outputs for each target."""
    def __call__(self, X):
        # Return a dict with each target mapped to (regression, classification)
        outputs = {}
        # regression raw values chosen to trigger energy high
        raw_vals = {
            'stress_level': 3.0,
            'mood_score': 8.5,
            'energy_level': 9.0,
            'focus_score': 7.5,
            'perceived_stress_scale': 12.0,
            'anxiety_score': 4.0,
            'depression_score': 6.0,
            'job_satisfaction': 8.0,
        }
        for k, v in raw_vals.items():
            reg = torch.tensor(v, dtype=torch.float32)
            # set a low logit so sigmoid -> near 0 (not at-risk)
            cls = torch.tensor(-10.0, dtype=torch.float32)
            outputs[k] = (reg, cls)
        return outputs


def run_test():
    # Create behavioral data: 7 days x 17 features
    seq_len = 7
    n_features = 17
    behavioral = np.zeros((seq_len, n_features), dtype=float)
    # Fill with reasonable defaults
    behavioral[:] = 5.0
    # Set last-day exercise_minutes (index 7) low to trigger safety
    behavioral[-1, 7] = 5.0
    # Set caffeine moderately high
    behavioral[-1, 9] = 450.0

    model = MockModel()
    preds = predict_mental_health(model, behavioral, scaler_mean=np.zeros(n_features), scaler_scale=np.ones(n_features), apply_amplification=True)

    print("--- Predictions (post-safety-layer) ---")
    for k, v in preds.items():
        print(f"{k}: value={v['value']}, at_risk_prob={v['at_risk_prob']}, safety_override={v.get('safety_override', False)}")

if __name__ == '__main__':
    run_test()
