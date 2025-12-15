#!/usr/bin/env python3
"""
Test the GRU + LSTM two-stage pipeline with sample data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

import numpy as np
from two_stage_models import load_pipeline, create_sample_history

def main():
    print("="*70)
    print("TWO-STAGE PIPELINE TEST: GRU + LSTM")
    print("="*70)
    
    # Check if models exist
    gru_path = Path("models/saved/best_behavioral_model.pt")
    lstm_path = Path("models/saved/mental_health_lstm.pt")
    
    print("\n📦 Checking model files...")
    print(f"  GRU model: {gru_path.exists()} - {gru_path}")
    print(f"  LSTM model: {lstm_path.exists()} - {lstm_path}")
    
    if not gru_path.exists():
        print("\n❌ GRU model not found!")
        print("   Download from Kaggle: best_behavioral_model.pt")
        print("   Place in: models/saved/")
        return
    
    if not lstm_path.exists():
        print("\n❌ LSTM model not found!")
        print("   Train with: python scripts/train_mental_health.py")
        return
    
    # Load pipeline
    print("\n🔄 Loading two-stage pipeline...")
    try:
        pipeline = load_pipeline(str(gru_path), str(lstm_path))
    except Exception as e:
        print(f"\n❌ Error loading pipeline: {e}")
        return
    
    # Create test data
    print("\n📊 Creating sample 7-day history...")
    history = create_sample_history()
    print(f"   Shape: {history.shape}")
    print(f"   Features: {pipeline.behavioral_features}")
    print("\n   Data preview:")
    for i, row in enumerate(history, 1):
        print(f"   Day {i}: Sleep={row[0]:.1f}h, Exercise={row[1]:.0f}min, "
              f"Steps={row[2]:.0f}, Screen={row[3]:.1f}h, Social={row[4]:.0f}, Work={row[5]:.1f}h")
    
    # Make prediction
    print("\n🔮 Running two-stage prediction...")
    print("   Stage 1: GRU forecasts next-day behaviors...")
    print("   Stage 2: LSTM predicts mental health...")
    
    try:
        result = pipeline.predict(history)
    except Exception as e:
        print(f"\n❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results
    print("\n" + "="*70)
    print("STAGE 1 OUTPUT: BEHAVIORAL FORECAST (GRU)")
    print("="*70)
    print(f"\n  Model: GRU (R²=0.48, MAE=162.67)")
    print(f"  Confidence: {result['stage1_confidence']:.1%}\n")
    
    for feature, value in result['behavioral_forecast'].items():
        print(f"  • {feature.replace('_', ' ').title()}: {value:.2f}")
    
    print("\n" + "="*70)
    print("STAGE 2 OUTPUT: MENTAL HEALTH PREDICTIONS (LSTM)")
    print("="*70)
    print(f"\n  Model: LSTM (R²=0.98, 8 prediction heads)")
    print(f"  Pipeline Version: {result['pipeline_version']}\n")
    
    # Group by category
    daily_metrics = ['stress_level', 'mood_score', 'energy_level', 'focus_score']
    weekly_metrics = ['pss_score', 'anxiety_score', 'depression_score', 'job_satisfaction']
    
    # Check what's actually in the results
    available_metrics = list(result['mental_health'].keys())
    
    print("  📅 Daily Metrics:")
    for target in daily_metrics:
        # Try both with and without prefix
        actual_target = None
        if target in available_metrics:
            actual_target = target
        elif target.replace('_score', '') in available_metrics:
            actual_target = target.replace('_score', '')
        elif f"perceived_{target}" in available_metrics:
            actual_target = f"perceived_{target}"
        
        if actual_target:
            pred = result['mental_health'][actual_target]
            risk_emoji = "🔴" if pred['risk_level'] == 'high_risk' else \
                        "🟡" if pred['risk_level'] == 'moderate_risk' else "🟢"
            print(f"     {risk_emoji} {actual_target.replace('_', ' ').title()}: "
                  f"{pred['value']:.2f} ({pred['confidence']:.0%} conf) - {pred['risk_level']}")
        else:
            print(f"     ⚠️  {target} not found in results")
    
    print("\n  📊 Weekly Metrics:")
    for target in weekly_metrics:
        # Try different name variations
        actual_target = None
        if target in available_metrics:
            actual_target = target
        elif target.replace('_score', '') in available_metrics:
            actual_target = target.replace('_score', '')
        elif f"perceived_{target}" in available_metrics:
            actual_target = f"perceived_{target}"
        
        if actual_target:
            pred = result['mental_health'][actual_target]
            risk_emoji = "🔴" if pred['risk_level'] == 'high_risk' else \
                        "🟡" if pred['risk_level'] == 'moderate_risk' else "🟢"
            print(f"     {risk_emoji} {actual_target.replace('_', ' ').title()}: "
                  f"{pred['value']:.2f} ({pred['confidence']:.0%} conf) - {pred['risk_level']}")
        else:
            print(f"     ⚠️  {target} not found in results")
    
    # Show what's actually available
    if len(available_metrics) > 0:
        print(f"\n  ℹ️  Available predictions: {', '.join(available_metrics)}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    high_risk = sum(1 for p in result['mental_health'].values() if p['risk_level'] == 'high_risk')
    moderate_risk = sum(1 for p in result['mental_health'].values() if p['risk_level'] == 'moderate_risk')
    low_risk = sum(1 for p in result['mental_health'].values() if p['risk_level'] == 'low_risk')
    
    print(f"\n  Risk Distribution:")
    print(f"    🔴 High Risk: {high_risk} metrics")
    print(f"    🟡 Moderate Risk: {moderate_risk} metrics")
    print(f"    🟢 Low Risk: {low_risk} metrics")
    
    print(f"\n  Overall Confidence: {result['stage1_confidence']:.1%}")
    print(f"\n✅ Two-stage pipeline test complete!")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
