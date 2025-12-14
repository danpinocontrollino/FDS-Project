import pytest
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from demo_app import predict_mental_health
from scripts.utils import get_config_path
import json

# Ensure thresholds are available for safety layer
cfg_path = get_config_path() / "thresholds.json"
with open(cfg_path, "r") as f:
    import demo_app
    demo_app.GLOBAL_THRESHOLDS = json.load(f)

# Mock Model
class MockModel(torch.nn.Module):
    def forward(self, x):
        # Returns high energy (9.0) regardless of input
        return {
            'energy_level': torch.tensor(9.0),
            'stress_level': torch.tensor(2.0)
        }


def test_sedentary_safety_cap():
    """Ensure energy is capped for sedentary inputs."""
    model = MockModel()
    
    # Create input with 0 exercise (Index 7)
    # 7 days, 17 features: zeros with exercise column 0
    input_data = np.zeros((7, 17)) 
    input_data[:, 7] = 0  # 0 minutes exercise
    
    # Mock scaler
    mean = np.zeros(17)
    scale = np.ones(17)
    
    # Run prediction (amplification disabled to test unconditional safety)
    preds = predict_mental_health(model, input_data, mean, scale, apply_amplification=False)
    
    # ASSERTION: Model predicted 9.0, but safety layer should cap at 6.0
    assert preds is not None
    assert 'energy_level' in preds
    assert preds['energy_level']['value'] <= 6.0
    assert preds['energy_level']['value'] != 9.0
