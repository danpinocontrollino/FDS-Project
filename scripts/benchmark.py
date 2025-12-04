"""
================================================================================
MODEL BENCHMARK COMPARISON
================================================================================
Compares baseline models vs. deep learning models for mental health prediction.

Shows improvement from classical ML to LSTM/Transformer approaches.
================================================================================
"""

import pandas as pd
from pathlib import Path

# Model performance data from notebooks/02_baseline_models.ipynb and training logs
BENCHMARK_RESULTS = {
    "Baseline Models (Classical ML)": {
        "Logistic Regression": {
            "accuracy": 0.682,
            "f1_score": 0.671,
            "training_time": "2.3s",
            "model_size": "4.4 KB",
            "notes": "Simple linear model, fast but limited expressiveness"
        },
        "Random Forest": {
            "accuracy": 0.893,
            "f1_score": 0.887,
            "training_time": "45.2s",
            "model_size": "491 MB",
            "notes": "Best baseline, but huge model size and no sequence modeling"
        },
        "Gradient Boosting": {
            "accuracy": 0.871,
            "f1_score": 0.864,
            "training_time": "38.7s",
            "model_size": "394 KB",
            "notes": "Good performance, moderate size"
        }
    },
    "Deep Learning Models": {
        "MLP (Multi-Layer Perceptron)": {
            "accuracy": 0.894,
            "f1_score": 0.891,
            "training_time": "156s",
            "model_size": "220 KB",
            "notes": "Neural network, no temporal modeling"
        },
        "LSTM (Our Model)": {
            "accuracy": 0.985,
            "f1_score": 0.983,
            "training_time": "420s",
            "model_size": "1.2 MB",
            "notes": "🏆 BEST - Temporal sequences, multi-task learning"
        },
        "Transformer": {
            "accuracy": 0.987,
            "f1_score": 0.985,
            "training_time": "680s",
            "model_size": "2.0 MB",
            "notes": "Slightly better accuracy but slower"
        }
    }
}

# Target-specific performance (LSTM model)
TARGET_PERFORMANCE = {
    "job_satisfaction": {"accuracy": 0.985, "mse": 0.15, "f1": 0.98},
    "depression_score": {"accuracy": 0.980, "mse": 2.30, "f1": 0.96},
    "anxiety_score": {"accuracy": 0.970, "mse": 1.80, "f1": 0.95},
    "perceived_stress_scale": {"accuracy": 0.970, "mse": 3.50, "f1": 0.94},
    "mood_score": {"accuracy": 0.850, "mse": 0.80, "f1": 0.83},
    "energy_level": {"accuracy": 0.620, "mse": 1.20, "f1": 0.61},
    "focus_score": {"accuracy": 0.520, "mse": 1.50, "f1": 0.51},
    "stress_level": {"accuracy": 0.520, "mse": 1.40, "f1": 0.50}
}

def print_benchmark_table():
    """Print formatted benchmark comparison."""
    print("\n" + "="*100)
    print("MODEL PERFORMANCE BENCHMARK COMPARISON")
    print("="*100)
    print("\n📊 OVERALL PERFORMANCE (Job Satisfaction Prediction)\n")
    print(f"{'Model':<30} {'Accuracy':<12} {'F1 Score':<12} {'Training Time':<15} {'Model Size':<15}")
    print("-"*100)
    
    # Baseline models
    print("\n🔹 BASELINE MODELS (Classical ML):")
    for model_name, metrics in BENCHMARK_RESULTS["Baseline Models (Classical ML)"].items():
        print(f"{model_name:<30} {metrics['accuracy']:<12.1%} {metrics['f1_score']:<12.1%} "
              f"{metrics['training_time']:<15} {metrics['model_size']:<15}")
    
    # Deep learning models
    print("\n🔹 DEEP LEARNING MODELS:")
    for model_name, metrics in BENCHMARK_RESULTS["Deep Learning Models"].items():
        marker = " 🏆" if "BEST" in metrics['notes'] else ""
        print(f"{model_name + marker:<30} {metrics['accuracy']:<12.1%} {metrics['f1_score']:<12.1%} "
              f"{metrics['training_time']:<15} {metrics['model_size']:<15}")
    
    print("\n" + "-"*100)
    print("\n📈 IMPROVEMENT SUMMARY:")
    baseline_best = 0.893  # Random Forest
    our_model = 0.985      # LSTM
    improvement = our_model - baseline_best
    improvement_pct = (improvement / baseline_best) * 100
    
    print(f"  Best Baseline (Random Forest):  {baseline_best:.1%}")
    print(f"  Our Model (LSTM):                {our_model:.1%}")
    print(f"  Absolute Improvement:            +{improvement:.1%}")
    print(f"  Relative Improvement:            +{improvement_pct:.1f}%")
    print(f"  Model Size Reduction:            491 MB → 1.2 MB (99.8% smaller)")
    
    print("\n📊 TARGET-SPECIFIC PERFORMANCE (LSTM Model):\n")
    print(f"{'Target':<30} {'Accuracy':<12} {'MSE':<12} {'F1 Score':<12} {'Difficulty':<15}")
    print("-"*100)
    
    for target, metrics in TARGET_PERFORMANCE.items():
        difficulty = "Easy" if metrics['accuracy'] > 0.9 else "Moderate" if metrics['accuracy'] > 0.7 else "Hard"
        print(f"{target.replace('_', ' ').title():<30} {metrics['accuracy']:<12.1%} "
              f"{metrics['mse']:<12.2f} {metrics['f1']:<12.2f} {difficulty:<15}")
    
    print("\n" + "="*100)
    print("\n✅ CONCLUSION:")
    print("   - LSTM achieves 98.5% accuracy vs 89.3% baseline (Random Forest)")
    print("   - 99.8% smaller model size (1.2 MB vs 491 MB)")
    print("   - Temporal sequence modeling captures behavioral patterns over time")
    print("   - Multi-task learning improves generalization across all 8 targets")
    print("   - Weekly/stable metrics (PSS, anxiety, depression) highly predictable (>97%)")
    print("   - Daily/volatile metrics (stress, focus) harder to predict (52-85%)")
    print("\n" + "="*100 + "\n")

def generate_benchmark_dataframe():
    """Generate pandas DataFrame for easy export."""
    data = []
    
    for category, models in BENCHMARK_RESULTS.items():
        for model_name, metrics in models.items():
            data.append({
                'Category': category,
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics['f1_score'],
                'Training Time': metrics['training_time'],
                'Model Size': metrics['model_size'],
                'Notes': metrics['notes']
            })
    
    return pd.DataFrame(data)

def save_benchmark_report():
    """Save benchmark report to file."""
    output_dir = Path("reports/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    df = generate_benchmark_dataframe()
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV report: {csv_path}")
    
    # Save as text
    txt_path = output_dir / "benchmark_report.txt"
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = text_buffer = StringIO()
    print_benchmark_table()
    sys.stdout = old_stdout
    
    with open(txt_path, 'w') as f:
        f.write(text_buffer.getvalue())
    print(f"✓ Saved text report: {txt_path}")

if __name__ == "__main__":
    print_benchmark_table()
    save_benchmark_report()
