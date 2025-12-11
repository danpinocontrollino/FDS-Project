"""
FIX MODEL COMPARISON DATA
=========================
This script fixes the dual_predictions_comparison.json file by:
1. Clipping all predictions to valid ranges
2. Adding metadata about the corrections
3. Recalculating error metrics and winners

The issue: The original script didn't apply output scaling/clipping,
so synthetic model predictions like 33.3 for a 0-10 scale are artifacts
of the raw neural network output without denormalization.

Run: python scripts/fix_comparison_data.py
"""

import json
from pathlib import Path
from datetime import datetime
import copy

# Valid ranges for each mental health target
TARGET_RANGES = {
    'stress_level': (0, 10),
    'mood_score': (0, 10),
    'energy_level': (0, 10),
    'focus_score': (0, 10),
    'perceived_stress_scale': (0, 40),  # PSS: 0-40 scale
    'anxiety_score': (0, 21),            # GAD-7 or similar
    'depression_score': (0, 27),         # PHQ-9 or similar  
    'job_satisfaction': (0, 10)
}

def clip_to_range(value, target):
    """Clip a value to the valid range for a target."""
    if value is None:
        return None
    min_val, max_val = TARGET_RANGES.get(target, (0, 10))
    return max(min_val, min(max_val, value))


def fix_comparison_data():
    """Fix the comparison JSON with proper value clipping."""
    
    input_path = Path("reports/dual_comparison/dual_predictions_comparison.json")
    output_path = input_path  # Overwrite
    backup_path = Path("reports/dual_comparison/dual_predictions_comparison_BACKUP.json")
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        print("   This file is generated on Kaggle. See KAGGLE_DUAL_MODEL_WORKFLOW.md")
        return False
    
    print("="*70)
    print("FIXING MODEL COMPARISON DATA")
    print("="*70)
    
    # Load original data
    with open(input_path) as f:
        data = json.load(f)
    
    # Create backup
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Backup created: {backup_path}")
    
    # Track corrections
    corrections = {
        'total_predictions_fixed': 0,
        'out_of_range_synthetic': 0,
        'out_of_range_real': 0,
        'targets_affected': {}
    }
    
    # Process each comparison
    fixed_data = copy.deepcopy(data)
    
    for comp in fixed_data['comparisons']:
        for target, pred_data in comp['predictions'].items():
            min_val, max_val = TARGET_RANGES.get(target, (0, 10))
            
            # Check and fix synthetic prediction
            if pred_data['synthetic_prediction'] is not None:
                original = pred_data['synthetic_prediction']
                clipped = clip_to_range(original, target)
                if abs(original - clipped) > 0.01:  # Significant change
                    corrections['out_of_range_synthetic'] += 1
                    corrections['total_predictions_fixed'] += 1
                    if target not in corrections['targets_affected']:
                        corrections['targets_affected'][target] = {'synthetic': 0, 'real': 0}
                    corrections['targets_affected'][target]['synthetic'] += 1
                pred_data['synthetic_prediction'] = clipped
                pred_data['synthetic_raw'] = original  # Keep original for reference
            
            # Check and fix real prediction
            if pred_data['real_prediction'] is not None:
                original = pred_data['real_prediction']
                clipped = clip_to_range(original, target)
                if abs(original - clipped) > 0.01:
                    corrections['out_of_range_real'] += 1
                    corrections['total_predictions_fixed'] += 1
                    if target not in corrections['targets_affected']:
                        corrections['targets_affected'][target] = {'synthetic': 0, 'real': 0}
                    corrections['targets_affected'][target]['real'] += 1
                pred_data['real_prediction'] = clipped
                pred_data['real_raw'] = original
            
            # Recalculate errors with clipped values
            if pred_data['actual_value'] is not None:
                actual = pred_data['actual_value']
                if pred_data['synthetic_prediction'] is not None:
                    pred_data['synthetic_error'] = abs(pred_data['synthetic_prediction'] - actual)
                if pred_data['real_prediction'] is not None:
                    pred_data['real_error'] = abs(pred_data['real_prediction'] - actual)
                
                # Recalculate winner
                if pred_data['synthetic_error'] is not None and pred_data['real_error'] is not None:
                    if pred_data['real_error'] < pred_data['synthetic_error']:
                        pred_data['winner'] = 'real'
                    elif pred_data['synthetic_error'] < pred_data['real_error']:
                        pred_data['winner'] = 'synthetic'
                    else:
                        pred_data['winner'] = 'tie'
    
    # Recalculate summary statistics
    synthetic_wins = 0
    real_wins = 0
    ties = 0
    
    for comp in fixed_data['comparisons']:
        for target, pred_data in comp['predictions'].items():
            if pred_data.get('winner') == 'synthetic':
                synthetic_wins += 1
            elif pred_data.get('winner') == 'real':
                real_wins += 1
            elif pred_data.get('winner') == 'tie':
                ties += 1
    
    total_with_gt = synthetic_wins + real_wins + ties
    
    fixed_data['total_predictions_with_ground_truth'] = total_with_gt
    fixed_data['model_performance']['synthetic_model']['wins'] = synthetic_wins
    fixed_data['model_performance']['synthetic_model']['win_rate'] = synthetic_wins / total_with_gt if total_with_gt > 0 else 0
    fixed_data['model_performance']['real_model']['wins'] = real_wins
    fixed_data['model_performance']['real_model']['win_rate'] = real_wins / total_with_gt if total_with_gt > 0 else 0
    fixed_data['model_performance']['ties'] = ties
    
    # Add correction metadata
    fixed_data['_corrections_applied'] = {
        'fixed_at': datetime.now().isoformat(),
        'reason': 'Raw neural network outputs were not clipped to valid target ranges',
        'total_predictions_fixed': corrections['total_predictions_fixed'],
        'out_of_range_synthetic': corrections['out_of_range_synthetic'],
        'out_of_range_real': corrections['out_of_range_real'],
        'targets_affected': corrections['targets_affected'],
        'target_ranges': TARGET_RANGES
    }
    
    # Save fixed data
    with open(output_path, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    print(f"\n📊 Corrections Applied:")
    print(f"   Total predictions fixed: {corrections['total_predictions_fixed']}")
    print(f"   Out-of-range synthetic:  {corrections['out_of_range_synthetic']}")
    print(f"   Out-of-range real:       {corrections['out_of_range_real']}")
    
    print(f"\n📊 Targets affected:")
    for target, counts in corrections['targets_affected'].items():
        range_str = f"[{TARGET_RANGES[target][0]}-{TARGET_RANGES[target][1]}]"
        print(f"   {target} {range_str}: {counts['synthetic']} synthetic, {counts['real']} real")
    
    print(f"\n📊 Updated Win Rates:")
    print(f"   Synthetic: {synthetic_wins}/{total_with_gt} ({synthetic_wins/total_with_gt*100:.1f}%)")
    print(f"   Real:      {real_wins}/{total_with_gt} ({real_wins/total_with_gt*100:.1f}%)")
    print(f"   Ties:      {ties}")
    
    print(f"\n✅ Fixed data saved to: {output_path}")
    return True


if __name__ == "__main__":
    fix_comparison_data()
