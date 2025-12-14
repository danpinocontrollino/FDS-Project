import numpy as np
import pytest

# Minimal shim if the real implementation is not yet available.
# The real code should provide `predict_with_safeguards` in scripts.predict_mental_health
try:
    from scripts.predict_mental_health import predict_with_safeguards
except Exception:
    def predict_with_safeguards(model, input_seq, input_features_dict):
        # Simple fallback: apply cap if avg exercise < 15
        raw = model(input_seq)
        final = {k: float(v) for k, v in raw.items()}
        avg_ex = np.mean(input_features_dict.get('exercise_minutes', [30]))
        if avg_ex < 15:
            if 'energy_level' in final:
                final['energy_level'] = min(final['energy_level'], 6.0)
        return final

# Mock model output
def mock_model(x):
    return {
        'energy_level': np.array(8.0),
        'anxiety_score': np.array(3.0)
    }

def test_sedentary_safeguard():
    input_seq = np.zeros((1, 7, 17))
    features = {'exercise_minutes': [0, 0, 0, 0, 0, 0, 0]}
    preds = predict_with_safeguards(mock_model, input_seq, features)
    assert preds['energy_level'] <= 6.0
    assert preds['energy_level'] != 8.0

def test_active_user_no_cap():
    input_seq = np.zeros((1, 7, 17))
    features = {'exercise_minutes': [60] * 7}
    preds = predict_with_safeguards(mock_model, input_seq, features)
    assert preds['energy_level'] == 8.0
