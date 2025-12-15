import pytest
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
import json
import os

from demo_app import predict_mental_health
import demo_app

# Ensure demo reads amplification config
from scripts.utils import get_config_path
cfg_path = get_config_path() / "thresholds.json"
with open(cfg_path, "r") as f:
    demo_app.GLOBAL_THRESHOLDS = json.load(f)

class MockModel(torch.nn.Module):
    def forward(self, x):
        # Return high raw values so amplification effect is visible
        return {
            'energy_level': torch.tensor(8.0),  # inverted target (higher better)
            'stress_level': torch.tensor(9.0)   # non-inverted target (higher worse)
        }


def make_input(extremity_value: float):
    # 7 days × 17 features
    arr = np.zeros((7, 17), dtype=float)
    # Fill with extremity_value to control z-score mean
    arr[:] = extremity_value
    return arr


def test_inverted_target_amplification_increases_distance():
    """For inverted targets (energy), higher extremity should push value further from midpoint."""
    model = MockModel()
    mean = np.zeros(17)
    scale = np.ones(17)

    # Create the "low" and "high" inputs as before
    low = make_input(0.0)
    high = make_input(5.0)

    # --- FIX ---
    # Set exercise minutes (feature index 7) to a healthy value (> 15 min)
    # to avoid the Sedentary Safety Layer triggering and capping the value at 6.0.
    # This ensures we test only the amplification behavior.
    EXERCISE_IDX = 7
    # Use 30 minutes (safe > sedentary threshold) to avoid safety cap
    # while keeping amplification bins distinct between low/high.
    low[:, EXERCISE_IDX] = 30.0
    high[:, EXERCISE_IDX] = 30.0
    # -----------

    preds_low = predict_mental_health(model, low, mean, scale, apply_amplification=True)
    preds_high = predict_mental_health(model, high, mean, scale, apply_amplification=True)

    assert 'energy_level' in preds_low and 'energy_level' in preds_high
    max_scale = 10.0
    midpoint = max_scale / 2.0

    dist_low = abs(preds_low['energy_level']['value'] - midpoint)
    dist_high = abs(preds_high['energy_level']['value'] - midpoint)

    # Now the assertion will pass because energy_level will not be capped at 6.0
    assert dist_high > dist_low


def test_noninverted_target_amplification_increases_value_when_high():
    """For non-inverted targets (stress), higher extremity should increase high-end values."""
    model = MockModel()
    mean = np.zeros(17)
    scale = np.ones(17)

    low = make_input(0.0)
    high = make_input(5.0)

    preds_low = predict_mental_health(model, low, mean, scale, apply_amplification=True)
    preds_high = predict_mental_health(model, high, mean, scale, apply_amplification=True)

    assert 'stress_level' in preds_low and 'stress_level' in preds_high
    assert preds_high['stress_level']['value'] >= preds_low['stress_level']['value']
