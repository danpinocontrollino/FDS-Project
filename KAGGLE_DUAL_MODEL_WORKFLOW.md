# Kaggle Workflow - Dual Model Comparison

## 🎯 Goal
Train a NEW LSTM on StudentLife data, then compare it against the synthetic-trained model to show how training data quality affects predictions.

## 📋 Step-by-Step Instructions

### Step 0: Clone GitHub Repo in Kaggle (5 min)

1. **Create Kaggle Notebook:** "StudentLife-LSTM-Training"

2. **Add Dataset:**
   - Search: "StudentLife" 
   - Add: "student-life" dataset

3. **Clone Your Repo:**
   ```python
   # Cell 1: Clone GitHub repo
   !git clone https://github.com/danpinocontrollino/FDS-Project.git
   %cd FDS-Project
   !pip install -q -r requirements.txt
   ```

4. **Install Dependencies:**
   ```python
   # Cell 2: Install any missing packages
   !pip install torch scikit-learn pandas numpy
   ```

---

### Step 1: Train StudentLife Model (30-40 min)

1. **Run Training Script:**
   ```python
   # Cell 3: Train on StudentLife
   !python scripts/train_studentlife_lstm_kaggle.py
   ```

2. **Download Model:**
   - Output: `/kaggle/working/mental_health_lstm_studentlife.pt`
   - Click "Output" tab → Download file
   - Save locally: `models/saved/mental_health_lstm_studentlife.pt`

**Expected Output:**
```
✓ Dataset: (674, 27)
Train: 539 records (8 students)
Val: 135 records (2 students)
✓ Model initialized: ~200K parameters
Epoch 10/50 - Train Loss: 0.4523, Val Loss: 0.3876
Epoch 20/50 - Train Loss: 0.3145, Val Loss: 0.3122
...
✓ TRAINING COMPLETE
Best Val Loss: 0.2845
Model saved: /kaggle/working/mental_health_lstm_studentlife.pt
```

---

### Step 2: Generate Dual Predictions (10-15 min)

**Option A: Same Notebook (Recommended)**
```python
# Cell 4: Upload synthetic model first
# Manually upload mental_health_lstm.pt to /kaggle/working/

# Cell 5: Run dual predictions
!python scripts/generate_dual_predictions_kaggle.py
```

**Option B: New Notebook**

1. **Create New Kaggle Notebook:** "Dual-Model-Comparison"

2. **Clone Repo Again:**
   ```python
   !git clone https://github.com/danpinocontrollino/FDS-Project.git
   %cd FDS-Project
   ```

3. **Add Datasets:**
   - StudentLife dataset
   - Upload BOTH models to `/kaggle/working/`:
     - `mental_health_lstm.pt` (synthetic-trained)
     - `mental_health_lstm_studentlife.pt` (just trained)

4. **Run Comparison:**
   ```python
   !python scripts/generate_dual_predictions_kaggle.py
   ```

5. **Download Results:**
   - Output: `/kaggle/working/dual_predictions_comparison.json`
   - Click "Output" tab → Download file
   - Save locally: `data/processed/dual_predictions_comparison.json`

**Expected Output:**
```
DUAL MODEL COMPARISON
Loading models...
  1. Synthetic-trained model...
     ✓ Trained on: Synthetic Kaggle (1.5M records)
  2. Real-trained model...
     ✓ Trained on: StudentLife (674 records)

Extracting StudentLife data...
Processing 10 students...
  u00: 67 days
  u01: 65 days
  ...

Running dual predictions...
✓ Generated 540 comparisons

✓ COMPARISON COMPLETE

Results:
  Synthetic Model Wins: 123/245 (50.2%)
  Real Model Wins:      98/245 (40.0%)
  Ties:                 24

Saved to: /kaggle/working/dual_predictions_comparison.json
```

---

### Step 3: Use Results Locally

Once you have `dual_predictions_comparison.json`:

1. **Place in repo:**
   ```bash
   cp ~/Downloads/dual_predictions_comparison.json data/processed/
   ```

2. **Run interactive viewer** (we'll build this next):
   ```bash
   streamlit run demo_app.py
   # Navigate to "Student Case Studies" section
   ```

3. **Generate confrontation dashboard:**
   ```bash
   python scripts/generate_confrontation_dashboard.py
   # Creates visual comparison of both models
   ```

---

## 📊 What You'll Get

### dual_predictions_comparison.json Structure:
```json
{
  "generated_at": "2024-12-09T...",
  "total_comparisons": 540,
  "model_performance": {
    "synthetic_model": {
      "wins": 123,
      "win_rate": 0.502,
      "training_data": "Synthetic Kaggle (1.5M records)"
    },
    "real_model": {
      "wins": 98,
      "win_rate": 0.400,
      "training_data": "StudentLife (674 records)"
    }
  },
  "comparisons": [
    {
      "student_id": "u00",
      "date": "2024-10-15",
      "behavioral_features": {...},
      "predictions": {
        "stress_level": {
          "synthetic_prediction": 5.2,
          "real_prediction": 4.8,
          "actual_value": 4.5,
          "synthetic_error": 0.7,
          "real_error": 0.3,
          "winner": "real"
        },
        ...
      }
    },
    ...
  ]
}
```

---

## 🎓 For Presentation

Use this data to show:

1. **Same Input, Different Outputs:**
   - "Student u00 on Oct 15: 7h sleep, 30min exercise"
   - "Synthetic model: Stress = 5.2"
   - "Real model: Stress = 4.8"
   - "Actual: Stress = 4.5" ← Real model wins!

2. **Training Data Quality Matters:**
   - Despite 2000× less training data, real model sometimes performs better
   - Shows importance of correlation strength over dataset size

3. **Interactive Exploration:**
   - Let audience pick students
   - Show timeline of predictions
   - Highlight where models disagree

---

## ⚠️ Common Issues

### Issue: "Model file not found"
**Solution:** Make sure you uploaded BOTH models to the same Kaggle dataset

### Issue: "Out of memory"
**Solution:** Reduce batch size in training script (line ~550):
```python
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Was 16
```

### Issue: "No data for student X"
**Solution:** Normal - some students have sparse data. Script handles this automatically.

### Issue: "All predictions are ties"
**Solution:** This means no actual EMA survey data available. Check if EMA folder exists in StudentLife dataset.

---

## 🚀 Next Steps

After getting `dual_predictions_comparison.json`:

1. Build interactive case study viewer (in progress)
2. Create confrontation dashboard
3. Add to demo presentation
4. Generate comparison visualizations

---

## 📝 Notes

- Training takes ~30-40 min on Kaggle GPU
- Dual predictions take ~10-15 min
- Total Kaggle time: ~1 hour
- Models are ~2-3 MB each
- Comparison JSON is ~500KB-1MB

## 🔧 Quick Commands (Copy-Paste)

**Single Kaggle Notebook (All-in-One):**
```python
# Cell 1: Setup
!git clone https://github.com/danpinocontrollino/FDS-Project.git
%cd FDS-Project
!pip install -q torch scikit-learn pandas numpy

# Cell 2: Train StudentLife model
!python scripts/train_studentlife_lstm_kaggle.py

# Cell 3: Upload synthetic model
# MANUALLY: Upload models/saved/mental_health_lstm.pt to /kaggle/working/

# Cell 4: Generate dual predictions
!python scripts/generate_dual_predictions_kaggle.py

# Cell 5: Download results
# MANUALLY: Download mental_health_lstm_studentlife.pt and dual_predictions_comparison.json
```

**Verification Commands:**
```python
# Check training output
!ls -lh /kaggle/working/*.pt

# Check comparison output
!ls -lh /kaggle/working/*.json

# Preview comparison results
import json
with open('/kaggle/working/dual_predictions_comparison.json') as f:
    data = json.load(f)
    print(f"Total comparisons: {data['total_comparisons']}")
    print(f"Synthetic wins: {data['model_performance']['synthetic_model']['wins']}")
    print(f"Real wins: {data['model_performance']['real_model']['wins']}")
```

Good luck! 🎉
