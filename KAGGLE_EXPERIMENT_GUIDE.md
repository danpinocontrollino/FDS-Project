# 🧪 Kaggle Model Comparison Experiment Guide

## Quick Start

### 1. Upload to Kaggle

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. Click **File → Upload Notebook**
4. Upload `notebooks/model_comparison_kaggle.ipynb`

### 2. Add Dataset

1. Click **"+ Add Data"** on the right sidebar
2. Search for **"StudentLife"** or add directly:
   - Dataset: `dartmouth-cs/student-life`
3. Verify path shows: `/kaggle/input/student-life/dataset`

### 3. Enable GPU (Optional)

1. Click **Settings** (gear icon) in right sidebar
2. Set **Accelerator** → **GPU T4 x2** (free tier)
3. This speeds up training ~10x

### 4. Run Experiment

1. Click **"Run All"** or **Ctrl+Shift+Enter**
2. Experiment takes ~15-30 minutes
3. Watch the leaderboard in Cell 7!

---

## Models Tested

| Model | Description |
|-------|-------------|
| **LSTM** | Standard recurrent baseline |
| **BiLSTM** | Bidirectional for richer context |
| **GRU** | Simpler gating, faster training |
| **Transformer** | Attention-based sequence modeling |
| **CNN-LSTM** | Local patterns + temporal |
| **MLP** | Simple baseline |

## Output Files

After running, download from `/kaggle/working`:

```
📁 Kaggle Output
├── best_behavioral_model.pt    ← Best model checkpoint
└── model_comparison_results.json ← All metrics
```

---

## Download Results

1. In Kaggle, go to **Output** tab
2. Click **Download All** or individual files
3. Place in `models/saved/` directory locally

---

## Expected Results

Based on StudentLife data characteristics:

| Model | Expected R² | Notes |
|-------|-------------|-------|
| CNN-LSTM | 0.70-0.85 | Often best for temporal patterns |
| BiLSTM | 0.65-0.80 | Strong bidirectional context |
| Transformer | 0.60-0.75 | Good with attention |
| GRU | 0.55-0.70 | Fast, competitive |
| LSTM | 0.50-0.65 | Solid baseline |
| MLP | 0.40-0.55 | No temporal info |

---

## After Kaggle

Once you have `best_behavioral_model.pt`, the two-stage pipeline automatically uses it:

```python
# In scripts/generate_profile.py
from two_stage_models import TwoStagePipeline

# Stage 1: Your best model (behavioral forecast)
# Stage 2: Existing synthetic model (mental health)
pipeline = TwoStagePipeline(
    behavioral_model_path="models/saved/best_behavioral_model.pt",
    mental_health_model_path="models/saved/mental_health_lstm.pt"
)
```

---

## Troubleshooting

### "Dataset not found"
- Verify path: `/kaggle/input/student-life/dataset`
- Re-add dataset from sidebar

### "CUDA out of memory"
- Reduce batch_size from 32 to 16
- Or disable GPU in settings

### Training too slow
- Enable GPU accelerator
- Reduce epochs from 50 to 30

---

## Code Location

- Notebook: [notebooks/model_comparison_kaggle.ipynb](notebooks/model_comparison_kaggle.ipynb)
- Python version: [notebooks/model_comparison_experiment.py](notebooks/model_comparison_experiment.py)
- Two-stage optimizer: [notebooks/two_stage_optimization.py](notebooks/two_stage_optimization.py)
