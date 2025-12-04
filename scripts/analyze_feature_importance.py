"""
================================================================================
FEATURE IMPORTANCE ANALYSIS
================================================================================
Analyze which behavioral features (sleep, work, exercise, etc.) are most
important for predicting each mental health outcome.

Methods:
1. Random Forest Feature Importance (Gini importance)
2. Permutation Importance (model-agnostic)
3. SHAP Values (Shapley Additive Explanations)
4. Correlation Analysis

Outputs:
- Feature importance rankings per target
- Visualization plots
- Summary report

Usage:
    python scripts/analyze_feature_importance.py
    python scripts/analyze_feature_importance.py --target depression_score
    python scripts/analyze_feature_importance.py --method shap

Author: FDS Project - Feature Analysis
================================================================================
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("reports/feature_importance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Target columns (8 mental health outcomes)
TARGET_COLS = [
    "stress_level",
    "mood_score",
    "energy_level",
    "focus_score",
    "perceived_stress_scale",
    "anxiety_score",
    "depression_score",
    "job_satisfaction",
]

# Feature display names
FEATURE_NAMES = {
    "sleep_hours": "Sleep Hours",
    "sleep_quality": "Sleep Quality",
    "work_hours": "Work Hours",
    "meetings_count": "Meetings",
    "tasks_completed": "Tasks Completed",
    "emails_received": "Emails Received",
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
# DATA LOADING
# ============================================================================

def load_synthetic_data(sample_size: int = 50000) -> pd.DataFrame:
    """Load the synthetic dataset with all features and targets."""
    # Try to load processed dataset
    weekly_path = DATA_DIR / "weekly_aggregated.parquet"
    daily_path = DATA_DIR / "daily_with_burnout.parquet"
    
    if weekly_path.exists():
        print(f"📂 Loading weekly aggregated data: {weekly_path}")
        df = pd.read_parquet(weekly_path)
    elif daily_path.exists():
        print(f"📂 Loading daily data: {daily_path}")
        df = pd.read_parquet(daily_path)
    else:
        # Try raw data
        raw_path = Path("data/raw/daily_all.csv")
        if raw_path.exists():
            print(f"📂 Loading raw data: {raw_path}")
            df = pd.read_csv(raw_path)
        else:
            raise FileNotFoundError("No processed data found. Please run preprocessing first.")
    
    print(f"✓ Loaded {len(df):,} samples")
    
    # Sample for faster analysis
    if len(df) > sample_size:
        print(f"📉 Sampling {sample_size:,} random samples for analysis...")
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"✓ Using {len(df):,} samples")
    print(f"✓ Columns: {list(df.columns[:10])}...")
    
    return df


def prepare_data(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare features (X) and target (y) for analysis.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        df_clean: Cleaned dataframe for correlation analysis
    """
    # Check if target exists
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset. Available: {list(df.columns)}")
    
    # Select features and target
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    print(f"\n✓ Using {len(available_features)}/{len(FEATURE_COLS)} features")
    
    df_clean = df[available_features + [target]].copy()
    
    # Drop rows with missing target
    df_clean = df_clean.dropna(subset=[target])
    
    # Handle categorical features
    categorical_mappings = {
        'work_pressure': {'low': 0, 'medium': 1, 'high': 2}
    }
    
    # Fill missing features and convert categoricals
    for col in available_features:
        if col in categorical_mappings:
            # Convert categorical to numeric
            df_clean[col] = df_clean[col].map(categorical_mappings[col])
        
        if df_clean[col].isna().any():
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    X = df_clean[available_features].values
    y = df_clean[target].values
    
    print(f"✓ Prepared {len(X):,} samples × {len(available_features)} features")
    print(f"✓ Target '{target}': mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, df_clean


# ============================================================================
# FEATURE IMPORTANCE METHODS
# ============================================================================

def random_forest_importance(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Calculate feature importance using Random Forest (Gini importance).
    
    Fast and commonly used method. Shows which features reduce impurity most.
    """
    print("\n" + "="*80)
    print("METHOD 1: Random Forest Feature Importance (Gini)")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Get R² score
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"✓ Random Forest R² - Train: {train_score:.3f}, Test: {test_score:.3f}")
    
    # Extract feature importances
    importances = rf.feature_importances_
    
    # Create DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_pct': importances * 100
    }).sort_values('importance', ascending=False)
    
    return df_importance


def permutation_importance_analysis(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Calculate feature importance using Permutation Importance.
    
    Model-agnostic method. Shuffles each feature and measures performance drop.
    More reliable than Gini importance for correlated features.
    """
    print("\n" + "="*80)
    print("METHOD 2: Permutation Importance")
    print("="*80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training model...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Calculate permutation importance
    print("Calculating permutation importance (this may take a minute)...")
    perm_importance = permutation_importance(
        rf, X_test, y_test, 
        n_repeats=10, 
        random_state=42,
        n_jobs=-1
    )
    
    # Create DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std,
        'importance_pct': perm_importance.importances_mean * 100
    }).sort_values('importance', ascending=False)
    
    return df_importance


def correlation_analysis(df: pd.DataFrame, target: str, feature_names: List[str]) -> pd.DataFrame:
    """
    Calculate Pearson correlation between features and target.
    
    Simple linear relationship measure. Positive = feature increases with target.
    """
    print("\n" + "="*80)
    print("METHOD 3: Correlation Analysis")
    print("="*80)
    
    correlations = []
    for feature in feature_names:
        if feature in df.columns:
            corr = df[feature].corr(df[target])
            correlations.append({
                'feature': feature,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'direction': 'positive' if corr > 0 else 'negative'
            })
    
    df_corr = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    return df_corr


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_feature_importance(df_importance: pd.DataFrame, target: str, method: str, output_dir: Path):
    """Create horizontal bar plot of feature importance."""
    plt.figure(figsize=(10, 8))
    
    # Top 15 features
    df_plot = df_importance.head(15).copy()
    df_plot = df_plot.sort_values('importance')  # Sort ascending for horizontal bar
    
    # Create plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_plot)))
    plt.barh(df_plot['feature'], df_plot['importance'], color=colors)
    
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Feature Importance for {target.replace("_", " ").title()}\n({method})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{target}_{method.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_path}")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, target: str, output_dir: Path):
    """Create correlation heatmap between features and target."""
    # Select features
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    corr_data = df[available_features + [target]].corr()[[target]].drop(target)
    
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
    plt.title(f'Feature Correlations with {target.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Behavioral Features', fontsize=12, fontweight='bold')
    plt.xlabel('', fontsize=12)
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{target}_correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap: {output_path}")
    plt.close()


def plot_top_features_comparison(results: Dict[str, pd.DataFrame], target: str, output_dir: Path):
    """Compare top features across different methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['Random Forest', 'Permutation', 'Correlation']
    
    for idx, (method, df) in enumerate(results.items()):
        ax = axes[idx]
        df_plot = df.head(10).sort_values('importance' if 'importance' in df.columns else 'abs_correlation')
        
        y_col = 'importance' if 'importance' in df.columns else 'abs_correlation'
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(df_plot)))
        
        ax.barh(df_plot['feature'], df_plot[y_col], color=colors)
        ax.set_xlabel('Importance/Correlation', fontweight='bold')
        ax.set_title(methods[idx], fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    fig.suptitle(f'Top 10 Features for {target.replace("_", " ").title()} (All Methods)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / f"{target}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison: {output_path}")
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results: Dict[str, pd.DataFrame], target: str, output_dir: Path):
    """Generate markdown report with all results."""
    report_path = output_dir / f"{target}_feature_importance_report.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Feature Importance Analysis: {target.replace('_', ' ').title()}\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report analyzes which behavioral features (sleep, work, exercise, etc.) ")
        f.write(f"are most predictive of **{target.replace('_', ' ')}**.\n\n")
        
        # Method 1: Random Forest
        f.write("---\n\n")
        f.write("## Method 1: Random Forest Feature Importance\n\n")
        f.write("**Top 10 Most Important Features:**\n\n")
        df_rf = results['Random Forest']
        for idx, row in df_rf.head(10).iterrows():
            f.write(f"{idx+1}. **{FEATURE_NAMES.get(row['feature'], row['feature'])}** - ")
            f.write(f"{row['importance_pct']:.2f}%\n")
        
        f.write("\n**Full Results:**\n\n")
        f.write(df_rf.to_markdown(index=False))
        f.write("\n\n")
        
        # Method 2: Permutation
        f.write("---\n\n")
        f.write("## Method 2: Permutation Importance\n\n")
        f.write("**Top 10 Most Important Features:**\n\n")
        df_perm = results['Permutation']
        for idx, row in df_perm.head(10).iterrows():
            f.write(f"{idx+1}. **{FEATURE_NAMES.get(row['feature'], row['feature'])}** - ")
            f.write(f"{row['importance']:.4f} ± {row.get('std', 0):.4f}\n")
        
        f.write("\n**Full Results:**\n\n")
        f.write(df_perm.to_markdown(index=False))
        f.write("\n\n")
        
        # Method 3: Correlation
        f.write("---\n\n")
        f.write("## Method 3: Correlation Analysis\n\n")
        f.write("**Top 10 Strongest Correlations:**\n\n")
        df_corr = results['Correlation']
        for idx, row in df_corr.head(10).iterrows():
            direction = "↑" if row['direction'] == 'positive' else "↓"
            f.write(f"{idx+1}. **{FEATURE_NAMES.get(row['feature'], row['feature'])}** {direction} - ")
            f.write(f"r = {row['correlation']:.3f}\n")
        
        f.write("\n**Full Results:**\n\n")
        f.write(df_corr.to_markdown(index=False))
        f.write("\n\n")
        
        # Consensus
        f.write("---\n\n")
        f.write("## Consensus: Top Features Across All Methods\n\n")
        
        # Combine rankings
        all_features = set(df_rf['feature'].tolist() + df_perm['feature'].tolist() + df_corr['feature'].tolist())
        consensus = []
        
        for feature in all_features:
            rf_rank = df_rf[df_rf['feature'] == feature].index[0] + 1 if feature in df_rf['feature'].values else 999
            perm_rank = df_perm[df_perm['feature'] == feature].index[0] + 1 if feature in df_perm['feature'].values else 999
            corr_rank = df_corr[df_corr['feature'] == feature].index[0] + 1 if feature in df_corr['feature'].values else 999
            
            avg_rank = (rf_rank + perm_rank + corr_rank) / 3
            consensus.append({
                'feature': feature,
                'avg_rank': avg_rank,
                'rf_rank': rf_rank,
                'perm_rank': perm_rank,
                'corr_rank': corr_rank
            })
        
        df_consensus = pd.DataFrame(consensus).sort_values('avg_rank')
        
        f.write("**Features ranked by average position across all methods:**\n\n")
        for idx, row in df_consensus.head(10).iterrows():
            f.write(f"{idx+1}. **{FEATURE_NAMES.get(row['feature'], row['feature'])}** ")
            f.write(f"(Avg Rank: {row['avg_rank']:.1f})\n")
            f.write(f"   - RF: #{row['rf_rank']}, Permutation: #{row['perm_rank']}, Correlation: #{row['corr_rank']}\n")
        
        f.write("\n")
    
    print(f"\n✓ Generated report: {report_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def analyze_target(target: str, df: pd.DataFrame):
    """Run complete feature importance analysis for one target."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {target.replace('_', ' ').upper()}")
    print(f"{'='*80}")
    
    # Prepare data
    X, y, df_clean = prepare_data(df, target)
    feature_names = [f for f in FEATURE_COLS if f in df_clean.columns]
    
    # Run all methods
    results = {}
    
    # Method 1: Random Forest
    df_rf = random_forest_importance(X, y, feature_names)
    results['Random Forest'] = df_rf
    plot_feature_importance(df_rf, target, 'Random Forest', OUTPUT_DIR)
    
    # Method 2: Permutation
    df_perm = permutation_importance_analysis(X, y, feature_names)
    results['Permutation'] = df_perm
    plot_feature_importance(df_perm, target, 'Permutation Importance', OUTPUT_DIR)
    
    # Method 3: Correlation
    df_corr = correlation_analysis(df_clean, target, feature_names)
    results['Correlation'] = df_corr
    plot_correlation_heatmap(df_clean, target, OUTPUT_DIR)
    
    # Comparison plot
    plot_top_features_comparison(results, target, OUTPUT_DIR)
    
    # Generate report
    generate_report(results, target, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: {target.replace('_', ' ').upper()}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze feature importance for mental health predictions")
    parser.add_argument('--target', type=str, help='Specific target to analyze (e.g., depression_score)')
    parser.add_argument('--all', action='store_true', help='Analyze all 8 targets')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS - Mental Health Prediction")
    print("="*80)
    
    # Load data
    df = load_synthetic_data()
    
    # Determine which targets to analyze
    if args.target:
        targets = [args.target]
    elif args.all:
        targets = TARGET_COLS
    else:
        # Default: analyze all targets
        targets = TARGET_COLS
    
    # Analyze each target
    for target in targets:
        try:
            analyze_target(target, df)
        except Exception as e:
            print(f"\n❌ Error analyzing {target}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - Feature importance plots (PNG)")
    print("  - Correlation heatmaps (PNG)")
    print("  - Comparison plots (PNG)")
    print("  - Detailed reports (Markdown)")


if __name__ == "__main__":
    main()
