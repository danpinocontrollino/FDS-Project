"""
================================================================================
FEATURE IMPORTANCE ANALYSIS - KAGGLE NOTEBOOK
================================================================================
Run this in Kaggle with the mental health dataset to analyze which behavioral
features are most predictive of mental health outcomes.

SETUP IN KAGGLE:
1. Add your dataset as input (e.g., /kaggle/input/mental-health-daily-logs)
2. Clone the repo and run preprocessing:
   !git clone https://github.com/danpinocontrollino/FDS-Project.git
   %cd FDS-Project
   !python scripts/create_burnout_labels.py --input-dir /kaggle/input/your-dataset/
   !python scripts/preprocess.py
3. Run this notebook (uses data/processed/daily_with_burnout.parquet)

Copy this entire file into a Kaggle notebook cell.
================================================================================
"""

# ============================================================================
# SETUP & IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature columns (17 behavioral features)
FEATURE_COLS = [
    "sleep_hours",
    "sleep_quality", 
    "work_hours",
    "meetings_count",
    "tasks_completed",
    "emails_received",
    "commute_minutes",
    "exercise_minutes",
    "steps_count",
    "caffeine_mg",
    "alcohol_units",
    "screen_time_hours",
    "social_interactions",
    "outdoor_time_minutes",
    "diet_quality",
    "work_pressure",
    "weather_mood_impact",
]

# Target columns - adjust based on your dataset
TARGET_COLS = [
    "stress_level",
    "mood_score",
    "energy_level",
    "focus_score",
    # Add these if available in your dataset:
    # "perceived_stress_scale",
    # "anxiety_score",
    # "depression_score",
    # "job_satisfaction",
]

FEATURE_NAMES = {
    "sleep_hours": "Sleep Hours",
    "sleep_quality": "Sleep Quality",
    "work_hours": "Work Hours",
    "meetings_count": "Meetings",
    "tasks_completed": "Tasks Completed",
    "emails_received": "Emails",
    "commute_minutes": "Commute Time",
    "exercise_minutes": "Exercise",
    "steps_count": "Steps",
    "caffeine_mg": "Caffeine",
    "alcohol_units": "Alcohol",
    "screen_time_hours": "Screen Time",
    "social_interactions": "Social Interactions",
    "outdoor_time_minutes": "Outdoor Time",
    "diet_quality": "Diet Quality",
    "work_pressure": "Work Pressure",
    "weather_mood_impact": "Weather Impact",
}

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("LOADING DATA")
print("="*80)

# Load preprocessed data
# After running create_burnout_labels.py, load the processed file:
df = pd.read_parquet('data/processed/daily_with_burnout.parquet')

print(f"✓ Loaded {len(df):,} samples")
print(f"✓ Columns: {df.columns.tolist()[:10]}...")

# Sample if too large (for faster analysis)
SAMPLE_SIZE = 50000
if len(df) > SAMPLE_SIZE:
    print(f"📉 Sampling {SAMPLE_SIZE:,} random samples...")
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"✓ Using {len(df):,} samples")

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df, target):
    """Prepare features and target for analysis."""
    print(f"\n{'='*80}")
    print(f"PREPARING DATA FOR: {target.upper()}")
    print(f"{'='*80}")
    
    # Check available features
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    print(f"✓ Found {len(available_features)}/{len(FEATURE_COLS)} features")
    
    if target not in df.columns:
        print(f"❌ Target '{target}' not found in dataset")
        return None, None, None
    
    # Select data
    df_clean = df[available_features + [target]].copy()
    
    # Drop missing targets
    df_clean = df_clean.dropna(subset=[target])
    
    # Handle categorical features
    if 'work_pressure' in df_clean.columns:
        pressure_map = {'low': 0, 'medium': 1, 'high': 2}
        df_clean['work_pressure'] = df_clean['work_pressure'].map(pressure_map)
    
    # Fill missing features with mean
    for col in available_features:
        if df_clean[col].isna().any():
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    X = df_clean[available_features].values
    y = df_clean[target].values
    
    print(f"✓ Final dataset: {len(X):,} samples × {len(available_features)} features")
    print(f"✓ Target stats: mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, available_features

# ============================================================================
# METHOD 1: RANDOM FOREST IMPORTANCE
# ============================================================================

def random_forest_importance(X, y, feature_names):
    """Calculate feature importance using Random Forest."""
    print(f"\n{'='*80}")
    print("METHOD 1: RANDOM FOREST FEATURE IMPORTANCE")
    print(f"{'='*80}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    
    # Evaluate
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"✓ R² Score - Train: {train_score:.3f}, Test: {test_score:.3f}")
    
    # Get importances
    importances = rf.feature_importances_
    
    # Create DataFrame
    df_importance = pd.DataFrame({
        'Feature': [FEATURE_NAMES.get(f, f) for f in feature_names],
        'Importance': importances,
        'Importance_pct': importances * 100
    }).sort_values('Importance', ascending=False)
    
    print("\n🏆 Top 10 Most Important Features:")
    for idx, row in df_importance.head(10).iterrows():
        print(f"  {idx+1}. {row['Feature']}: {row['Importance_pct']:.2f}%")
    
    return df_importance

# ============================================================================
# METHOD 2: PERMUTATION IMPORTANCE
# ============================================================================

def permutation_importance_analysis(X, y, feature_names):
    """Calculate permutation importance."""
    print(f"\n{'='*80}")
    print("METHOD 2: PERMUTATION IMPORTANCE")
    print(f"{'='*80}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Calculate permutation importance
    print("Calculating permutation importance...")
    perm_imp = permutation_importance(
        rf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Create DataFrame
    df_importance = pd.DataFrame({
        'Feature': [FEATURE_NAMES.get(f, f) for f in feature_names],
        'Importance': perm_imp.importances_mean,
        'Std': perm_imp.importances_std
    }).sort_values('Importance', ascending=False)
    
    print("\n🏆 Top 10 Most Important Features:")
    for idx, row in df_importance.head(10).iterrows():
        print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")
    
    return df_importance

# ============================================================================
# METHOD 3: CORRELATION ANALYSIS
# ============================================================================

def correlation_analysis(df, target, feature_cols):
    """Calculate correlations with target."""
    print(f"\n{'='*80}")
    print("METHOD 3: CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    correlations = []
    for feature in feature_cols:
        if feature in df.columns:
            corr = df[feature].corr(df[target])
            correlations.append({
                'Feature': FEATURE_NAMES.get(feature, feature),
                'Correlation': corr,
                'Abs_Correlation': abs(corr),
                'Direction': '↑ Positive' if corr > 0 else '↓ Negative'
            })
    
    df_corr = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
    
    print("\n🏆 Top 10 Strongest Correlations:")
    for idx, row in df_corr.head(10).iterrows():
        print(f"  {idx+1}. {row['Feature']} {row['Direction']}: r = {row['Correlation']:.3f}")
    
    return df_corr

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_feature_importance(df_importance, target, method):
    """Plot feature importance."""
    plt.figure(figsize=(10, 8))
    
    # Top 15 features
    df_plot = df_importance.head(15).copy()
    
    # Sort for horizontal bar
    if 'Importance' in df_plot.columns:
        df_plot = df_plot.sort_values('Importance')
        y_col = 'Importance'
    else:
        df_plot = df_plot.sort_values('Abs_Correlation')
        y_col = 'Abs_Correlation'
    
    # Create plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_plot)))
    plt.barh(df_plot['Feature'], df_plot[y_col], color=colors)
    
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Feature Importance: {target.replace("_", " ").title()}\n({method})',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, target, feature_cols):
    """Plot correlation heatmap."""
    # Get available features
    available = [f for f in feature_cols if f in df.columns]
    
    # Calculate correlations
    corr_data = df[available + [target]].corr()[[target]].drop(target)
    corr_data.index = [FEATURE_NAMES.get(f, f) for f in corr_data.index]
    
    plt.figure(figsize=(8, 10))
    sns.heatmap(
        corr_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Correlation'},
        linewidths=0.5
    )
    plt.title(f'Feature Correlations: {target.replace("_", " ").title()}',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def plot_comparison(results, target):
    """Compare methods side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (method, df_result) in enumerate(results.items()):
        ax = axes[idx]
        
        # Get top 10
        df_plot = df_result.head(10).copy()
        
        # Determine column
        if 'Importance' in df_plot.columns:
            y_col = 'Importance'
        else:
            y_col = 'Abs_Correlation'
        
        df_plot = df_plot.sort_values(y_col)
        
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(df_plot)))
        ax.barh(df_plot['Feature'], df_plot[y_col], color=colors)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(method, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    fig.suptitle(f'Method Comparison: {target.replace("_", " ").title()}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_target(df, target):
    """Run complete analysis for one target."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {target.replace('_', ' ').upper()}")
    print(f"{'='*80}")
    
    # Prepare data
    X, y, feature_names = prepare_data(df, target)
    
    if X is None:
        return None
    
    # Run all methods
    results = {}
    
    # Method 1: Random Forest
    df_rf = random_forest_importance(X, y, feature_names)
    results['Random Forest'] = df_rf
    plot_feature_importance(df_rf, target, 'Random Forest')
    
    # Method 2: Permutation
    df_perm = permutation_importance_analysis(X, y, feature_names)
    results['Permutation'] = df_perm
    plot_feature_importance(df_perm, target, 'Permutation Importance')
    
    # Method 3: Correlation
    df_clean = df[[f for f in FEATURE_COLS if f in df.columns] + [target]].dropna(subset=[target])
    if 'work_pressure' in df_clean.columns:
        df_clean['work_pressure'] = df_clean['work_pressure'].map({'low': 0, 'medium': 1, 'high': 2})
    df_corr = correlation_analysis(df_clean, target, feature_names)
    results['Correlation'] = df_corr
    plot_correlation_heatmap(df_clean, target, feature_names)
    
    # Comparison
    plot_comparison(results, target)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {target.replace('_', ' ').upper()}")
    print(f"{'='*80}")
    
    # Consensus top features
    all_features = set()
    for method_df in results.values():
        all_features.update(method_df['Feature'].tolist())
    
    consensus = []
    for feature in all_features:
        ranks = []
        for method_df in results.values():
            if feature in method_df['Feature'].values:
                rank = method_df[method_df['Feature'] == feature].index[0] + 1
            else:
                rank = 999
            ranks.append(rank)
        
        avg_rank = np.mean(ranks)
        consensus.append({'Feature': feature, 'Avg_Rank': avg_rank})
    
    df_consensus = pd.DataFrame(consensus).sort_values('Avg_Rank')
    
    print("\n🏆 CONSENSUS TOP 10 FEATURES (across all methods):")
    for idx, row in df_consensus.head(10).iterrows():
        print(f"  {idx+1}. {row['Feature']} (Avg Rank: {row['Avg_Rank']:.1f})")
    
    return results

# ============================================================================
# RUN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Analyze each target
for target in TARGET_COLS:
    if target in df.columns:
        results = analyze_target(df, target)
    else:
        print(f"\n⚠️  Skipping {target} - not found in dataset")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
