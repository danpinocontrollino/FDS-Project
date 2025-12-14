#!/bin/bash
# =============================================================================
# Mental Health Profiling System - Repository Cleanup Script
# =============================================================================
# Purpose: Remove redundant documentation, duplicate models, temporary files,
#          and lecture materials that don't belong in a production repository.
#
# Author: FDS Project Team
# Date: December 2024
#
# Usage: bash cleanup.sh [--dry-run]
#        --dry-run: Show what would be deleted without actually deleting
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be deleted ==="
    echo ""
fi

delete_item() {
    local path="$1"
    local desc="$2"
    
    if [[ -e "$path" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "[DRY-RUN] Would delete: $path ($desc)"
        else
            rm -rf "$path"
            echo "[DELETED] $path ($desc)"
        fi
    fi
}

echo "=============================================="
echo "  Repository Cleanup - Mental Health System"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# SECTION 1: Redundant Documentation in Root
# Keep: README.md, requirements.txt, core guides
# Remove: Session summaries, implementation logs, duplicate quick-starts
# -----------------------------------------------------------------------------
echo "--- Removing Redundant Root Documentation ---"

# Implementation/session summaries (internal development notes)
delete_item "$PROJECT_ROOT/ARCHITECTURE_CLEANUP_SUMMARY.md" "Old cleanup session log"
delete_item "$PROJECT_ROOT/CLEANUP_COMPLETED.md" "Old cleanup completion log"
delete_item "$PROJECT_ROOT/CLEANUP_SUMMARY.md" "Old cleanup summary"
delete_item "$PROJECT_ROOT/DATA_QUALITY_INTEGRATION_SUMMARY.md" "Integration session log"
delete_item "$PROJECT_ROOT/GOOGLE_FORM_IMPLEMENTATION_SUMMARY.md" "Implementation log"
delete_item "$PROJECT_ROOT/HISTORY_IMPLEMENTATION_COMPLETE.md" "Implementation log"
delete_item "$PROJECT_ROOT/PRODUCTION_CLEANUP_SUMMARY.md" "Cleanup session log"
delete_item "$PROJECT_ROOT/STAGE1_DENORMALIZATION_FIX.md" "Bug fix log"
delete_item "$PROJECT_ROOT/TWO_STAGE_DEMO_COMPLETE.md" "Demo completion log"
delete_item "$PROJECT_ROOT/TWO_STAGE_DEMO_INTEGRATION.md" "Integration log"
delete_item "$PROJECT_ROOT/TWO_STAGE_HTML_GENERATION.md" "HTML generation log"
delete_item "$PROJECT_ROOT/VISUALIZATION_FIXES_COMPLETE.md" "Fix completion log"

# Redundant quick-starts (consolidate into one)
delete_item "$PROJECT_ROOT/GOOGLE_FORM_QUICK_START.md" "Duplicate quick-start"
delete_item "$PROJECT_ROOT/QUICK_START_PRESENTATION.md" "Duplicate quick-start"
delete_item "$PROJECT_ROOT/TWO_STAGE_QUICK_REF.md" "Duplicate quick-ref"
delete_item "$PROJECT_ROOT/QUICK_REFERENCE.md" "Duplicate quick-ref"

# Duplicate presentation guides
delete_item "$PROJECT_ROOT/PRESENTATION_CHECKLIST.md" "Presentation prep doc"
delete_item "$PROJECT_ROOT/PRESENTATION_STRATEGY.md" "Presentation prep doc"

# Duplicate pipeline summaries
delete_item "$PROJECT_ROOT/TWO_STAGE_PIPELINE_SUMMARY.md" "Duplicate - see docs/"
delete_item "$PROJECT_ROOT/TWO_STAGE_DEMO_SCRIPT.md" "Demo script notes"

# Duplicate Google Form docs (keep one comprehensive guide)
delete_item "$PROJECT_ROOT/GOOGLE_FORM_SETUP_GUIDE.md" "Duplicate - see INSTRUCTIONS"
delete_item "$PROJECT_ROOT/GOOGLE_FORM_TEMPLATE.md" "Template in docs/"

# Technical notes that belong in code comments
delete_item "$PROJECT_ROOT/KAGGLE_DUAL_MODEL_WORKFLOW.md" "Move to docstring"

echo ""

# -----------------------------------------------------------------------------
# SECTION 2: Lecture PDFs that shouldn't be in the repo
# These are course materials, not project deliverables
# -----------------------------------------------------------------------------
echo "--- Removing Lecture Materials from docs/ ---"

delete_item "$PROJECT_ROOT/docs/deep_neural networks.pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/shallow_neural_networks.pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/Trees_Ensembles.pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/Transformers 2025 (1).pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/VAE GAN Lecture 14 (1).pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/Grid Architecture 2025.pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/PCA.pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/exercise_convolution.pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/intro_generation.pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/representation.pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/vectors&projections.pdf" "Lecture PDF"
delete_item "$PROJECT_ROOT/docs/Untitled Page 7.pdf" "Unknown lecture PDF"

echo ""

# -----------------------------------------------------------------------------
# SECTION 3: Duplicate Model Files
# Keep: mental_health_lstm.pt (main), stage1_behavioral_forecasting.pt
# Remove: Duplicates with (1) suffix
# -----------------------------------------------------------------------------
echo "--- Removing Duplicate Model Files ---"

delete_item "$PROJECT_ROOT/models/saved/stage1_behavioral_forecasting (1).pt" "Duplicate model"
delete_item "$PROJECT_ROOT/models/saved/two_stage_predictions (1).json" "Duplicate predictions"

echo ""

# -----------------------------------------------------------------------------
# SECTION 4: Old/Test Report Files
# Keep: examples/, two_stage_analysis/ (for presentation)
# Remove: Timestamped test profiles that clutter the directory
# -----------------------------------------------------------------------------
echo "--- Removing Old Test Report Files ---"

# Remove timestamped JSON profiles (keep HTML reports)
find "$PROJECT_ROOT/reports" -maxdepth 1 -name "profile_*.json" -type f | while read -r file; do
    delete_item "$file" "Timestamped test profile"
done

# Remove comparison PNGs (generated during analysis, not for delivery)
delete_item "$PROJECT_ROOT/reports/comparison_correlations.png" "Analysis artifact"
delete_item "$PROJECT_ROOT/reports/comparison_data_availability.png" "Analysis artifact"
delete_item "$PROJECT_ROOT/reports/comparison_feature_variance.png" "Analysis artifact"
delete_item "$PROJECT_ROOT/reports/comparison_summary_table.png" "Analysis artifact"

echo ""

# -----------------------------------------------------------------------------
# SECTION 5: Python Cache and Temporary Files
# Note: Exclude .venv directory - those are installed package caches
# -----------------------------------------------------------------------------
echo "--- Removing Python Cache ---"

find "$PROJECT_ROOT" -type d -name "__pycache__" -not -path "*/.venv/*" | while read -r dir; do
    delete_item "$dir" "Python cache"
done

find "$PROJECT_ROOT" -type f -name "*.pyc" -not -path "*/.venv/*" | while read -r file; do
    delete_item "$file" "Compiled Python"
done

delete_item "$PROJECT_ROOT/download.txt" "Temporary download file"

echo ""

# -----------------------------------------------------------------------------
# SECTION 6: Empty Directories
# -----------------------------------------------------------------------------
echo "--- Checking for Empty Directories ---"

# data/raw is typically empty placeholder
if [[ -d "$PROJECT_ROOT/data/raw" ]] && [[ -z "$(ls -A "$PROJECT_ROOT/data/raw" 2>/dev/null)" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would keep empty dir: data/raw (placeholder)"
    else
        echo "[KEPT] data/raw (placeholder directory)"
    fi
fi

echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "=============================================="
echo "  Cleanup Complete!"
echo "=============================================="
echo ""
echo "Files KEPT (essential):"
echo "  - README.md (main project documentation)"
echo "  - requirements.txt (dependencies)"
echo "  - BEHAVIORAL_INTERVENTIONS.md (clinical documentation)"
echo "  - CLINICAL_VALIDATION_COMPLETE.md (validation methodology)"
echo "  - MODEL_VALIDATION_VS_CLINICAL_BENCHMARKS.md (benchmark analysis)"
echo "  - LONGITUDINAL_TRACKING.md (feature documentation)"
echo "  - GOOGLE_FORM_INSTRUCTIONS.md (user guide)"
echo "  - PRESENTATION_GUIDE.md (presentation notes)"
echo "  - PROJECT_EVALUATION_CHECKLIST.md (grading criteria)"
echo "  - CITATION.md (academic citation)"
echo "  - UNCERTAINTY_CALCULATION.md (methodology)"
echo ""
echo "Models KEPT:"
echo "  - mental_health_lstm.pt (production model)"
echo "  - mental_health_lstm_studentlife.pt (research variant)"
echo "  - mental_health_transformer.pt (comparison model)"
echo "  - stage1_behavioral_forecasting.pt (Stage 1 model)"
echo "  - two_stage_predictions.json (cached predictions)"
echo ""
echo "Directories KEPT:"
echo "  - docs/ (comprehensive documentation)"
echo "  - reports/two_stage_analysis/ (research visualizations)"
echo "  - reports/examples/ (demo outputs)"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo ">>> This was a DRY RUN. Run without --dry-run to actually delete files."
fi
