"""
Download the Kaggle dataset:
    "Work-Life Balance Synthetic Daily Wellness Dataset"
into data/raw/.

Usage:
    1. Install kaggle:
        pip install kaggle

    2. Set your API token in the environment, e.g. in Git Bash:
        export KAGGLE_API_TOKEN=KGAT_...

       or in PowerShell:
        $env:KAGGLE_API_TOKEN = "KGAT_..."

    3. From the project root, run:
        python scripts/download_data.py
"""

from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_SLUG = "wafaaelhusseini/worklife-balance-synthetic-daily-wellness-dataset"
OUT_DIR = Path("data/raw")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print("❌ Kaggle API authentication failed.")
        print("Make sure you have set KAGGLE_API_TOKEN in your environment.")
        print("Error:", e)
        return

    print(f"⬇️  Downloading dataset '{DATASET_SLUG}' to '{OUT_DIR}' ...")
    api.dataset_download_files(
        DATASET_SLUG,
        path=str(OUT_DIR),
        unzip=True,
    )
    print("✅ Download complete.")
    print("Files saved under: ", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
