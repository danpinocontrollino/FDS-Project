#!/bin/bash
# ==============================================================================
# FINAL REPOSITORY CLEANUP SCRIPT
# ==============================================================================
# Purpose: Remove experimental artifacts and legacy files before academic submission.
# This script preserves only production-ready code and research-relevant materials.
#
# Usage: chmod +x cleanup_final.sh && ./cleanup_final.sh
# To preview without deleting, run: DRY_RUN=1 ./cleanup_final.sh
# ==============================================================================

set -e  # Exit on any error

DRY_RUN=${DRY_RUN:-0}

function safe_rm() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY RUN] Would remove: $1"
    else
        if [ -e "$1" ]; then
            rm -rf "$1"
            echo "✓ Removed: $1"
        else
            echo "⊘ Not found (skipping): $1"
        fi
    fi
}

echo "=============================================="
echo "  Mental Health Profiling System - Cleanup   "
echo "=============================================="
echo ""

# ------------------------------------------------------------------------------
# PHASE 1: Remove archive directories
# ------------------------------------------------------------------------------
echo "Phase 1: Removing archive directories..."
safe_rm "archive/"
safe_rm "scripts/archive/"
safe_rm "models/archive/"
safe_rm "data/archive/"
safe_rm "notebooks/archive/"

# ------------------------------------------------------------------------------
# PHASE 2: Remove obsolete experimental scripts
# ------------------------------------------------------------------------------
echo ""
echo "Phase 2: Removing obsolete experimental scripts..."

# Training scripts that are no longer part of the core pipeline
safe_rm "scripts/train_cvae.py"
safe_rm "scripts/train_mae.py"
safe_rm "scripts/train_mlp.py"
safe_rm "scripts/train_transformer.py"
safe_rm "scripts/ensemble_predict.py"

# Backup/experimental scripts
safe_rm "scripts/generate_two_stage_html_backup.py"

# ------------------------------------------------------------------------------
# PHASE 3: Remove unused heavy model files
# ------------------------------------------------------------------------------
echo ""
echo "Phase 3: Removing unused model artifacts..."

# Legacy classifiers
safe_rm "models/saved/rf_classifier.pkl"
safe_rm "models/saved/gb_classifier.pkl"
safe_rm "models/saved/logistic_classifier.pkl"
safe_rm "models/saved/cvae_advisor.pt"

# Old sequence models (replaced by mental_health_lstm.pt)
safe_rm "models/saved/lstm_sequence.pt"
safe_rm "models/saved/gru_sequence.pt"
safe_rm "models/saved/transformer_sequence.pt"

# MAE experiments
safe_rm "models/saved/mae_encoder.pt"
safe_rm "models/saved/mae_full.pt"

# ------------------------------------------------------------------------------
# PHASE 4: Clean redundant reports (keep examples/ and two_stage_analysis/)
# ------------------------------------------------------------------------------
echo ""
echo "Phase 4: Cleaning redundant reports..."

# Remove top-level profile JSONs (keep organized subdirectories)
# Keep: reports/examples/, reports/two_stage_analysis/, reports/benchmarks/
for f in reports/profile_*.json; do
    safe_rm "$f"
done

for f in reports/report_*.html; do
    safe_rm "$f"
done

# Remove duplicate comparison images at reports root (keep in subdirs)
safe_rm "reports/comparison_correlations.png"
safe_rm "reports/comparison_data_availability.png"
safe_rm "reports/comparison_feature_variance.png"
safe_rm "reports/comparison_summary_table.png"

# Clean duplicate two_stage_predictions.json in models/saved
safe_rm "models/saved/two_stage_predictions (1).json"

# ------------------------------------------------------------------------------
# PHASE 5: Remove Python cache
# ------------------------------------------------------------------------------
echo ""
echo "Phase 5: Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "✓ Cleaned Python cache"

# ------------------------------------------------------------------------------
# PHASE 6: Remove empty directories
# ------------------------------------------------------------------------------
echo ""
echo "Phase 6: Removing empty directories..."
find . -type d -empty -delete 2>/dev/null || true
echo "✓ Cleaned empty directories"

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Cleanup Complete!                          "
echo "=============================================="
echo ""
echo "Preserved (Production System):"
echo "  ✓ demo_app.py"
echo "  ✓ scripts/generate_profile.py"
echo "  ✓ scripts/predict_mental_health.py"
echo "  ✓ scripts/model_definitions.py"
echo "  ✓ models/saved/mental_health_lstm.pt"
echo ""
echo "Preserved (Research Pipeline):"
echo "  ✓ scripts/train_stage1_behavioral_kaggle.py"
echo "  ✓ scripts/run_two_stage_pipeline_kaggle.py"
echo "  ✓ scripts/analyze_two_stage_pipeline.py"
echo "  ✓ reports/two_stage_analysis/"
echo ""
echo "Preserved (Configuration):"
echo "  ✓ config/thresholds.json"
echo "  ✓ config/job_categories.json"
echo ""
echo "Preserved (Examples):"
echo "  ✓ reports/examples/"
echo "  ✓ reports/benchmarks/"
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "NOTE: This was a dry run. No files were actually deleted."
    echo "Run without DRY_RUN=1 to execute cleanup."
fi
