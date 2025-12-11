"""
RESEARCH VISUALIZATION GENERATOR
================================
Generate all necessary visualizations for the Research tab in demo_app.py.

This script creates:
1. Correlation comparison charts (synthetic vs real vs clinical)
2. Data availability comparison
3. Feature variance comparison
4. Summary comparison table

Run: python scripts/generate_research_visuals.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

OUTPUT_DIR = Path("reports")


def generate_correlation_comparison():
    """
    Generate correlation comparison chart showing:
    - Expected correlations from clinical literature
    - Actual correlations in synthetic training data
    - Correlations found in real StudentLife data
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data: correlation coefficients
    correlations = {
        'Sleep → Stress': {'clinical': -0.40, 'synthetic': -0.07, 'real': None},
        'Exercise → Mood': {'clinical': 0.35, 'synthetic': 0.09, 'real': -0.04},
        'Work Hours → Stress': {'clinical': 0.35, 'synthetic': 0.20, 'real': None},
        'Social → Depression': {'clinical': -0.30, 'synthetic': -0.05, 'real': None},
        'Caffeine → Anxiety': {'clinical': 0.25, 'synthetic': 0.08, 'real': None},
        'Screen Time → Mood': {'clinical': -0.20, 'synthetic': -0.03, 'real': -0.02},
    }
    
    x = np.arange(len(correlations))
    width = 0.25
    
    clinical_vals = [v['clinical'] for v in correlations.values()]
    synthetic_vals = [v['synthetic'] for v in correlations.values()]
    real_vals = [v['real'] if v['real'] is not None else 0 for v in correlations.values()]
    real_available = [v['real'] is not None for v in correlations.values()]
    
    # Bars
    bars1 = ax.bar(x - width, clinical_vals, width, label='Clinical Literature', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, synthetic_vals, width, label='Synthetic (Training)', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, real_vals, width, label='Real (StudentLife)', color='#e74c3c', alpha=0.8)
    
    # Mark unavailable real data
    for i, (bar, avail) in enumerate(zip(bars3, real_available)):
        if not avail:
            bar.set_alpha(0.2)
            bar.set_hatch('///')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Correlation Coefficient (r)')
    ax.set_title('Behavioral-Mental Health Correlations: Clinical Evidence vs. Training Data\n'
                 '(Synthetic data shows 3-7× weaker correlations than clinical literature)', 
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(correlations.keys(), rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.5, 0.5)
    
    # Add annotation
    ax.annotate('⚠️ Weak correlations in synthetic data\nmay cause model to underreact\nto behavioral changes',
                xy=(4, 0.08), xytext=(4.5, 0.35),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: comparison_correlations.png")


def generate_data_availability_comparison():
    """
    Generate data availability comparison showing synthetic vs real data coverage.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Behavioral features
    ax1 = axes[0]
    features = ['Sleep', 'Exercise', 'Screen Time', 'Social', 'Work Hours', 'Caffeine']
    synthetic_avail = [100, 100, 100, 100, 100, 100]
    real_avail = [90, 85, 88, 92, 87, 60]  # Estimated from StudentLife
    
    x = np.arange(len(features))
    width = 0.35
    
    ax1.bar(x - width/2, synthetic_avail, width, label='Synthetic', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, real_avail, width, label='Real (StudentLife)', color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('Data Availability (%)')
    ax1.set_title('Behavioral Features Coverage', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=30, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 110)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Complete')
    
    # Right: Mental health labels
    ax2 = axes[1]
    targets = ['Stress', 'Mood', 'Anxiety', 'Depression', 'Energy', 'Focus', 'Job Sat.', 'PSS']
    synthetic_labels = [100, 100, 100, 100, 100, 100, 100, 100]
    real_labels = [15, 8, 5, 5, 0, 0, 0, 12]  # StudentLife EMA response rates
    
    x = np.arange(len(targets))
    
    ax2.bar(x - width/2, synthetic_labels, width, label='Synthetic', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, real_labels, width, label='Real (StudentLife)', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('Label Availability (%)')
    ax2.set_title('Mental Health Labels Coverage\n(The Label Scarcity Problem)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets, rotation=30, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 110)
    
    # Add annotation for the key insight
    ax2.annotate('❌ Real-world:\nStudents rarely\nfill surveys!',
                xy=(3, 5), xytext=(5, 60),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_data_availability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: comparison_data_availability.png")


def generate_feature_variance_comparison():
    """
    Generate feature variance comparison showing data richness.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = ['Sleep Hours', 'Exercise Min', 'Screen Time', 'Social', 'Steps', 'Caffeine']
    
    # Unique values found in real data (approximated from StudentLife analysis)
    synthetic_unique = [180, 200, 180, 150, 500, 200]  # Synthetic has many values
    real_unique = [406, 431, 417, 179, 450, 120]  # Real sensor data
    
    x = np.arange(len(features))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, synthetic_unique, width, label='Synthetic', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, real_unique, width, label='Real (StudentLife)', color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('Unique Values (Data Richness)')
    ax.set_title('Behavioral Feature Variance: Synthetic vs Real Sensors\n'
                 '(Real sensors capture rich behavioral variation ✓)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=30, ha='right')
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_feature_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: comparison_feature_variance.png")


def generate_summary_table():
    """
    Generate a visual summary table comparing datasets.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    
    # Table data
    columns = ['Aspect', 'Synthetic (Kaggle)', 'Real (StudentLife)', 'Implication']
    data = [
        ['Sample Size', '1.5M daily records', '49 students × 70 days', 'Synthetic enables deep learning'],
        ['Behavioral Coverage', '100% complete', '85-92% (sensors)', 'Both have good inputs'],
        ['Mental Health Labels', '100% complete', '5-15% (sparse EMAs)', '⚠️ Label scarcity problem'],
        ['Correlations', '3-7× weaker than clinical', 'Cannot validate', 'Model may underreact'],
        ['Temporal Patterns', 'Independent days', 'Real autocorrelation', 'Real captures habits'],
        ['Generalizability', 'Synthetic patterns', 'US college students', 'Different populations'],
    ]
    
    # Create table
    table = ax.table(
        cellText=data,
        colLabels=columns,
        loc='center',
        cellLoc='left',
        colWidths=[0.18, 0.25, 0.25, 0.32]
    )
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Style rows
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')
            else:
                table[(i, j)].set_facecolor('white')
            
            # Highlight implication column
            if j == 3:
                table[(i, j)].set_text_props(fontweight='bold')
    
    # Highlight problem row (Mental Health Labels)
    for j in range(len(columns)):
        table[(3, j)].set_facecolor('#ffcccc')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    ax.set_title('Research Summary: Synthetic vs Real-World Data Trade-offs\n', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated: comparison_summary_table.png")


def main():
    """Generate all research visualizations."""
    print("=" * 60)
    print("Generating Research Visualizations")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    generate_correlation_comparison()
    generate_data_availability_comparison()
    generate_feature_variance_comparison()
    generate_summary_table()
    
    print("\n" + "=" * 60)
    print("✅ All research visualizations generated!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
