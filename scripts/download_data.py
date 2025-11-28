"""
================================================================================
DOWNLOAD DATA SCRIPT
================================================================================
This script downloads the "Work-Life Balance Synthetic Daily Wellness Dataset"
from Kaggle and saves it to data/raw/.

The dataset contains daily behavioral logs and weekly mental health summaries
for synthetic users, which we use to predict burnout risk.

Dataset source: https://www.kaggle.com/datasets/wafaaelhusseini/worklife-balance-synthetic-daily-wellness-dataset

Usage:
    1. Install kaggle: pip install kaggle
    2. Set your API credentials:
        export KAGGLE_USERNAME=your_username
        export KAGGLE_KEY=your_api_key
    3. Run: python scripts/download_data.py

Author: University Project - Burnout Prediction
================================================================================
"""

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# ============================================================================
# CONFIGURATION
# ============================================================================

# Kaggle dataset identifier (owner/dataset-name format)
DATASET_SLUG = "wafaaelhusseini/worklife-balance-synthetic-daily-wellness-dataset"

# Output directory for raw data files
OUT_DIR = Path("data/raw")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main() -> None:
    """
    Main entry point: authenticates with Kaggle API and downloads the dataset.
    
    The dataset will be automatically unzipped into OUT_DIR, creating files like:
    - daily_logs.csv: Daily behavioral data (sleep, exercise, work hours, etc.)
    - weekly_summaries.csv: Weekly mental health scores (stress, anxiety, etc.)
    - interventions.csv: Records of wellness interventions per user
    - daily_all.csv: Extended daily data with categorical attributes
    """
    
    # Create output directory if it doesn't exist
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize and authenticate with Kaggle API
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        # Authentication failed - user needs to set up credentials
        print("❌ Kaggle API authentication failed.")
        print("Make sure you have set KAGGLE_USERNAME and KAGGLE_KEY in your environment.")
        print("Error:", e)
        return

    # Download and unzip the dataset
    print(f"⬇️  Downloading dataset '{DATASET_SLUG}' to '{OUT_DIR}' ...")
    api.dataset_download_files(
        DATASET_SLUG,
        path=str(OUT_DIR),
        unzip=True,  # Automatically extract CSV files
    )
    
    print("✅ Download complete.")
    print("Files saved under:", OUT_DIR.resolve())


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
