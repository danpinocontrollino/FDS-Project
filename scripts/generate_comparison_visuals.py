"""
Data Quality Comparison Visualization
======================================
Compare synthetic Kaggle data vs real StudentLife data

Generates:
1. Data Availability Chart (sensor days vs survey days)
2. Feature Variance Comparison (behavioral vs mental health)
3. Correlation Comparison (expected vs synthetic vs StudentLife)
4. Summary statistics table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# LOAD DATA
# ============================================================================

def load_synthetic_data():
    """Load synthetic Kaggle training data"""
    try:
        df = pd.read_parquet('data/processed/daily_with_burnout.parquet')
        print(f"✓ Loaded synthetic data: {df.shape}")
        return df
    except Exception as e:
        print(f"✗ Error loading synthetic data: {e}")
        return None


def load_studentlife_data():
    """Load StudentLife processed data (if exists locally)"""
    try:
        # This would be the output from kaggle_studentlife_extraction.py
        # For now, we'll use the statistics we know from the Kaggle run
        return None  # Will use hardcoded stats below
    except Exception as e:
        return None


# ============================================================================
# STUDENTLIFE STATISTICS (from Kaggle execution)
# ============================================================================

STUDENTLIFE_STATS = {
    'total_records': 674,
    'students': 10,
    'feature_variance': {
        'sleep_hours': {'variance': 1.97, 'unique': 406},
        'exercise_minutes': {'variance': 198.11, 'unique': 431},
        'screen_time_hours': {'variance': 23.16, 'unique': 417},
        'social_interactions': {'variance': 602.42, 'unique': 179},
        'work_hours': {'variance': 0.59, 'unique': 389},
        'stress_level': {'variance': 0.00, 'unique': 1},
        'mood_score': {'variance': 0.12, 'unique': 8},
    },
    'data_availability': {
        'u00': {'sleep': 40, 'exercise': 66, 'screen': 61, 'social': 85, 'work': 84, 'stress': 47, 'mood': 9},
        'u01': {'sleep': 41, 'exercise': 62, 'screen': 57, 'social': 65, 'work': 65, 'stress': 27, 'mood': 2},
        'u02': {'sleep': 40, 'exercise': 60, 'screen': 57, 'social': 66, 'work': 66, 'stress': 39, 'mood': 5},
        'u03': {'sleep': 43, 'exercise': 58, 'screen': 54, 'social': 71, 'work': 71, 'stress': 24, 'mood': 1},
        'u04': {'sleep': 41, 'exercise': 55, 'screen': 51, 'social': 64, 'work': 64, 'stress': 15, 'mood': 3},
        'u05': {'sleep': 38, 'exercise': 56, 'screen': 52, 'social': 60, 'work': 60, 'stress': 11, 'mood': 1},
        'u07': {'sleep': 41, 'exercise': 66, 'screen': 62, 'social': 74, 'work': 74, 'stress': 9, 'mood': 0},
        'u08': {'sleep': 39, 'exercise': 61, 'screen': 57, 'social': 66, 'work': 66, 'stress': 4, 'mood': 1},
        'u09': {'sleep': 42, 'exercise': 61, 'screen': 57, 'social': 77, 'work': 77, 'stress': 18, 'mood': 2},
        'u10': {'sleep': 36, 'exercise': 58, 'screen': 54, 'social': 70, 'work': 70, 'stress': 9, 'mood': 0},
    },
    'correlations': {
        'sleep_vs_stress': np.nan,  # No variance in stress
        'exercise_vs_mood': -0.0350,
        'work_vs_stress': np.nan,  # No variance in stress
        'social_vs_mood': -0.0698,
    }
}


# ============================================================================
# VISUALIZATION 1: DATA AVAILABILITY COMPARISON
# ============================================================================

def plot_data_availability():
    """Compare data availability: Synthetic (perfect) vs StudentLife (sparse)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Synthetic data: Perfect daily alignment
    categories = ['Behavioral\nFeatures', 'Mental Health\nTargets']
    synthetic_availability = [100, 100]  # Perfect daily data
    
    ax1.bar(categories, synthetic_availability, color=['#2ecc71', '#3498db'], alpha=0.7, edgecolor='black')
    ax1.set_ylim(0, 105)
    ax1.set_ylabel('Data Availability (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Synthetic Kaggle Dataset\n(1.479M records, 500K users)', 
                  fontsize=13, fontweight='bold')
    ax1.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='100% Coverage')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (cat, val) in enumerate(zip(categories, synthetic_availability)):
        ax1.text(i, val + 2, f'{val}%', ha='center', fontsize=12, fontweight='bold')
    
    # StudentLife data: Sparse mental health surveys
    behavioral_avg = np.mean([
        np.mean([s['sleep'], s['exercise'], s['screen'], s['social'], s['work']]) 
        for s in STUDENTLIFE_STATS['data_availability'].values()
    ])
    mental_health_avg = np.mean([
        np.mean([s['stress'], s['mood']]) 
        for s in STUDENTLIFE_STATS['data_availability'].values()
    ])
    
    # Calculate as percentage of 60-85 day study period (use 70 days average)
    total_days_avg = 70
    behavioral_pct = (behavioral_avg / total_days_avg) * 100
    mental_health_pct = (mental_health_avg / total_days_avg) * 100
    
    studentlife_availability = [behavioral_pct, mental_health_pct]
    colors = ['#2ecc71', '#e74c3c']  # Green for good, red for sparse
    
    bars = ax2.bar(categories, studentlife_availability, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Data Availability (%)', fontsize=12, fontweight='bold')
    ax2.set_title('StudentLife Dataset\n(674 records, 49 students, 10 weeks)', 
                  fontsize=13, fontweight='bold')
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='100% Coverage')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (cat, val) in enumerate(zip(categories, studentlife_availability)):
        ax2.text(i, val + 2, f'{val:.0f}%', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = 'reports/comparison_data_availability.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# VISUALIZATION 2: FEATURE VARIANCE COMPARISON
# ============================================================================

def plot_feature_variance(df_synthetic):
    """Compare feature variance between synthetic and StudentLife data"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    features = ['sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                'social_interactions', 'work_hours', 'stress_level', 'mood_score']
    
    synthetic_variances = []
    studentlife_variances = []
    
    for feat in features:
        # Synthetic variance
        if df_synthetic is not None and feat in df_synthetic.columns:
            synthetic_var = df_synthetic[feat].var()
            synthetic_variances.append(synthetic_var)
        else:
            synthetic_variances.append(0)
        
        # StudentLife variance
        studentlife_var = STUDENTLIFE_STATS['feature_variance'].get(feat, {}).get('variance', 0)
        studentlife_variances.append(studentlife_var)
    
    # Create grouped bar chart
    x = np.arange(len(features))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, synthetic_variances, width, label='Synthetic (Kaggle)', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, studentlife_variances, width, label='Real (StudentLife)', 
                   color='#2ecc71', alpha=0.7, edgecolor='black')
    
    # Highlight mental health features (low variance)
    for i, feat in enumerate(features):
        if feat in ['stress_level', 'mood_score']:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='red', zorder=0)
    
    ax.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Variance: Synthetic vs Real-World Data\n(Red highlight = Mental Health Targets)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax.text(0.02, 0.98, 
            'Real data: Excellent behavioral variance\nSparse mental health self-reports',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = 'reports/comparison_feature_variance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# VISUALIZATION 3: CORRELATION COMPARISON
# ============================================================================

def plot_correlation_comparison(df_synthetic):
    """Compare correlations: Expected vs Synthetic vs StudentLife"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define expected clinical correlations
    correlations = [
        {
            'name': 'Sleep → Stress\n(negative)',
            'expected': -0.40,
            'synthetic': -0.0706 if df_synthetic is not None else -0.07,
            'studentlife': np.nan,  # No variance
            'y_pos': 4
        },
        {
            'name': 'Exercise → Mood\n(positive)',
            'expected': 0.35,
            'synthetic': 0.0887 if df_synthetic is not None else 0.09,
            'studentlife': -0.0350,
            'y_pos': 3
        },
        {
            'name': 'Work → Stress\n(positive)',
            'expected': 0.35,
            'synthetic': 0.2035 if df_synthetic is not None else 0.20,
            'studentlife': np.nan,
            'y_pos': 2
        },
        {
            'name': 'Social → Mood\n(positive)',
            'expected': 0.30,
            'synthetic': 0.15 if df_synthetic is not None else 0.15,
            'studentlife': -0.0698,
            'y_pos': 1
        },
    ]
    
    # Calculate actual correlations from synthetic if available
    if df_synthetic is not None:
        if 'sleep_hours' in df_synthetic.columns and 'stress_level' in df_synthetic.columns:
            correlations[0]['synthetic'] = df_synthetic[['sleep_hours', 'stress_level']].corr().iloc[0, 1]
        if 'exercise_minutes' in df_synthetic.columns and 'mood_score' in df_synthetic.columns:
            correlations[1]['synthetic'] = df_synthetic[['exercise_minutes', 'mood_score']].corr().iloc[0, 1]
        if 'work_hours' in df_synthetic.columns and 'stress_level' in df_synthetic.columns:
            correlations[2]['synthetic'] = df_synthetic[['work_hours', 'stress_level']].corr().iloc[0, 1]
        if 'social_interactions' in df_synthetic.columns and 'mood_score' in df_synthetic.columns:
            correlations[3]['synthetic'] = df_synthetic[['social_interactions', 'mood_score']].corr().iloc[0, 1]
    
    # Plot horizontal bars
    bar_height = 0.2
    colors = {'expected': '#2ecc71', 'synthetic': '#3498db', 'studentlife': '#e74c3c'}
    
    for corr in correlations:
        y = corr['y_pos']
        
        # Expected correlation
        ax.barh(y + bar_height, corr['expected'], bar_height, 
                label='Expected (Clinical)' if y == 4 else '', 
                color=colors['expected'], alpha=0.8, edgecolor='black')
        ax.text(corr['expected'] + 0.02, y + bar_height, f"{corr['expected']:.2f}", 
                va='center', fontweight='bold', fontsize=9)
        
        # Synthetic correlation
        ax.barh(y, corr['synthetic'], bar_height, 
                label='Synthetic (Kaggle)' if y == 4 else '', 
                color=colors['synthetic'], alpha=0.8, edgecolor='black')
        ax.text(corr['synthetic'] + 0.02, y, f"{corr['synthetic']:.3f}", 
                va='center', fontweight='bold', fontsize=9)
        
        # StudentLife correlation (may be nan)
        if not np.isnan(corr['studentlife']):
            ax.barh(y - bar_height, corr['studentlife'], bar_height, 
                    label='Real (StudentLife)' if y == 4 else '', 
                    color=colors['studentlife'], alpha=0.8, edgecolor='black')
            ax.text(corr['studentlife'] + 0.02, y - bar_height, f"{corr['studentlife']:.3f}", 
                    va='center', fontweight='bold', fontsize=9)
        else:
            # Show "No Data" for nan
            ax.text(-0.5, y - bar_height, 'No Data (constant)', 
                    va='center', fontsize=8, style='italic', color='gray')
    
    ax.set_yticks([c['y_pos'] for c in correlations])
    ax.set_yticklabels([c['name'] for c in correlations], fontsize=11)
    ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Correlation Strength: Clinical Literature vs Training Data vs Validation\n' +
                 'Synthetic data shows 3-7× weaker correlations than expected',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(-0.6, 0.5)
    
    plt.tight_layout()
    output_path = 'reports/comparison_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# VISUALIZATION 4: DATA QUALITY SUMMARY TABLE
# ============================================================================

def generate_summary_table():
    """Generate comparison summary table as image"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Synthetic (Kaggle)', 'Real (StudentLife)', 'Implication'],
        ['Total Records', '1,479,000', '674', 'Synthetic: Large scale\nReal: Limited sample'],
        ['Users/Students', '500,000', '49', 'Synthetic: Population-level\nReal: Single cohort'],
        ['Study Duration', 'Simulated (daily)', '10 weeks (real)', 'Synthetic: Perfect coverage\nReal: Realistic span'],
        ['Behavioral Data', '100% coverage', '85-90% coverage', 'Both excellent'],
        ['Mental Health Data', '100% coverage', '10-20% coverage', 'Real: Sparse self-reports'],
        ['Sleep-Stress Corr', 'r = -0.07 (weak)', 'r = N/A (no variance)', 'Synthetic: Too weak\nReal: No signal'],
        ['Exercise-Mood Corr', 'r = +0.09 (weak)', 'r = -0.04 (wrong sign)', 'Both problematic'],
        ['Data Quality', 'Perfect alignment\nWeak relationships', 'Strong sensors\nSparse surveys', 'Trade-off between\nscale and realism'],
    ]
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.2, 0.25, 0.25, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=10)
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('#ffffff')
            cell.set_edgecolor('#95a5a6')
    
    plt.title('Data Quality Comparison: Synthetic vs Real-World Mental Health Data',
              fontsize=14, fontweight='bold', pad=20)
    
    output_path = 'reports/comparison_summary_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("GENERATING DATA QUALITY COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    # Load synthetic data
    df_synthetic = load_synthetic_data()
    
    # Create output directory
    Path('reports').mkdir(exist_ok=True)
    
    # Generate all visualizations
    print("\n1. Data Availability Comparison...")
    plot_data_availability()
    
    print("\n2. Feature Variance Comparison...")
    plot_feature_variance(df_synthetic)
    
    print("\n3. Correlation Comparison...")
    plot_correlation_comparison(df_synthetic)
    
    print("\n4. Summary Table...")
    generate_summary_table()
    
    print("\n" + "=" * 80)
    print("✓ ALL VISUALIZATIONS GENERATED")
    print("=" * 80)
    print("\nOutput files:")
    print("  - reports/comparison_data_availability.png")
    print("  - reports/comparison_feature_variance.png")
    print("  - reports/comparison_correlations.png")
    print("  - reports/comparison_summary_table.png")
    print("\nUse these in your presentation to show synthetic vs real-world trade-offs!")


if __name__ == '__main__':
    main()
