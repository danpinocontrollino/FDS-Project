"""
================================================================================
BURNOUT PREDICTION PIPELINE ORCHESTRATOR
================================================================================
Complete pipeline that coordinates:
1. Google Form CSV parsing
2. Burnout risk prediction
3. Mental health profiling
4. HTML report generation

USAGE:
    python scripts/main_pipeline.py --csv responses.csv --user john@example.com
    python scripts/main_pipeline.py --csv data.csv --user john@example.com --output-dir reports/

FEATURES:
    ✓ Single command execution
    ✓ Automatic error handling
    ✓ Clear progress logging
    ✓ Multiple output formats (JSON, HTML)
    ✓ Consistent data parsing across modules

Author: University Project - Burnout Prediction
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Import unified parser
from form_parser import GoogleFormParser

# Import modules (will be loaded with proper error handling)
try:
    import torch
except ImportError:
    print("⚠️  PyTorch not installed. Install with: pip install torch")
    sys.exit(1)


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Model paths
MODEL_DIR = PROJECT_ROOT / "models/saved"
BURNOUT_MODEL = MODEL_DIR / "lstm_sequence.pt"
MENTAL_HEALTH_MODEL = MODEL_DIR / "mental_health_lstm.pt"

# Default output directory
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"


# ============================================================================
# STEP 1: PARSE GOOGLE FORM CSV
# ============================================================================

def step_parse_form(csv_path: str | Path) -> Tuple[pd.DataFrame, bool]:
    """
    Step 1: Parse Google Form CSV using unified parser.
    
    Args:
        csv_path: Path to Google Form CSV export
        
    Returns:
        (parsed_dataframe, is_daily_format)
    """
    print("\n" + "=" * 70)
    print("📂 STEP 1: Parsing Google Form CSV")
    print("=" * 70)
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        parser = GoogleFormParser()
        df = parser.parse_google_form_csv(csv_path)
        is_daily = parser.detect_format(df)
        
        print(f"✓ Loaded: {csv_path.name}")
        print(f"✓ Rows: {len(df)}")
        print(f"✓ Columns: {len(df.columns)}")
        print(f"✓ Format: {'DAILY' if is_daily == 'daily' else 'WEEKLY'}")
        
        return df, (is_daily == "daily")
        
    except Exception as e:
        print(f"❌ Error parsing CSV: {str(e)}")
        raise


# ============================================================================
# STEP 2: BURNOUT RISK PREDICTION
# ============================================================================

def step_predict_burnout(df: pd.DataFrame, user_email: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Step 2: Predict burnout risk using the trained LSTM model.
    
    Args:
        df: Parsed dataframe from Step 1
        user_email: User email for identification
        model_path: Optional path to custom model
        
    Returns:
        Dictionary with burnout predictions and analysis
    """
    print("\n" + "=" * 70)
    print("🧠 STEP 2: Predicting Burnout Risk")
    print("=" * 70)
    
    try:
        # Import predict_burnout module
        import sys
        sys.path.insert(0, str(SCRIPT_DIR))
        from predict_burnout import (
            load_model, create_weekly_sequence, predict, 
            FEATURE_COLS, DEFAULTS, RISK_LEVELS, BINARY_RISK_LEVELS,
            get_user_context, get_specific_advice, detect_contradictions,
            get_cvae_advisor
        )
        
        # Determine model to use
        model_file = Path(model_path) if model_path else BURNOUT_MODEL
        
        if not model_file.exists():
            raise FileNotFoundError(f"Burnout model not found: {model_file}")
        
        # Load model
        print(f"📦 Loading model: {model_file.name}...")
        model, model_type, feature_cols, is_binary = load_model(model_file)
        print(f"✓ Model type: {model_type}")
        
        # Prepare data for prediction
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Use first row (or only row)
        row = df.iloc[0]
        
        # Extract features
        data = {}
        for feature in FEATURE_COLS:
            if feature in row.index and pd.notna(row[feature]):
                data[feature] = row[feature]
            else:
                data[feature] = DEFAULTS.get(feature, 5)
        
        # Add V2 features for context
        v2_features = ["social_quality", "social_interactions", "outdoor_time_minutes",
                      "diet_quality", "job_satisfaction", "recovery_ability", "job_type",
                      "job_requires_screen", "loneliness_level", "work_life_boundary",
                      "morning_mood", "productivity_today", "day_overall_rating"]
        
        for v2_feat in v2_features:
            if v2_feat in row.index and pd.notna(row[v2_feat]):
                data[v2_feat] = row[v2_feat]
        
        # Create sequence and predict
        sequence = create_weekly_sequence(data, feature_cols)
        pred_class, probs = predict(model, sequence, feature_cols)
        
        # Get risk level information
        risk_levels = BINARY_RISK_LEVELS if is_binary else RISK_LEVELS
        risk_info = risk_levels.get(int(pred_class), {})
        
        # Get context-aware advice
        context = get_user_context(data)
        job_type = data.get("job_type", "other")
        advice = get_specific_advice(data, job_type, context)
        
        # Detect contradictions
        contradictions = detect_contradictions(data)
        
        # Get CVAE advisor insights
        cvae_advice = get_cvae_advisor(data)
        
        print(f"✓ Prediction complete")
        print(f"✓ Risk level: {risk_info.get('name', 'UNKNOWN')} {risk_info.get('emoji', '')}")
        
        result = {
            "user_email": user_email,
            "timestamp": datetime.now().isoformat(),
            "burnout": {
                "prediction_class": int(pred_class),
                "risk_level": risk_info.get("name", "UNKNOWN"),
                "risk_emoji": risk_info.get("emoji", ""),
                "risk_description": risk_info.get("desc", ""),
                "probability": float(probs[int(pred_class)]) if isinstance(probs, (list, np.ndarray)) else probs,
                "all_probabilities": [float(p) for p in probs] if isinstance(probs, (list, np.ndarray)) else [float(probs)],
            },
            "data": data,
            "advice": advice,
            "contradictions": contradictions,
            "cvae_insights": cvae_advice,
            "context": context,
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error during burnout prediction: {str(e)}")
        traceback.print_exc()
        raise


# ============================================================================
# STEP 3: MENTAL HEALTH PROFILING
# ============================================================================

def step_generate_profile(df: pd.DataFrame, user_email: str) -> Dict[str, Any]:
    """
    Step 3: Generate mental health profile.
    
    Args:
        df: Parsed dataframe from Step 1
        user_email: User email
        
    Returns:
        Dictionary with mental health profile data
    """
    print("\n" + "=" * 70)
    print("❤️  STEP 3: Generating Mental Health Profile")
    print("=" * 70)
    
    try:
        import sys
        sys.path.insert(0, str(SCRIPT_DIR))
        from generate_profile import (
            FEATURE_COLS as PROFILE_FEATURE_COLS,
            ALL_TARGETS, INVERTED_TARGETS
        )
        
        # Extract features for profile
        profile_features = {}
        for feature in PROFILE_FEATURE_COLS:
            if feature in df.columns and pd.notna(df[feature].iloc[0]):
                profile_features[feature] = df[feature].iloc[0]
        
        print(f"✓ Extracted {len(profile_features)} features")
        print(f"✓ Tracking {len(ALL_TARGETS)} mental health metrics")
        
        # Create profile structure
        profile = {
            "user_email": user_email,
            "timestamp": datetime.now().isoformat(),
            "features": profile_features,
            "targets": ALL_TARGETS,
            "inverted_targets": list(INVERTED_TARGETS),
        }
        
        return profile
        
    except Exception as e:
        print(f"⚠️  Warning during profile generation: {str(e)}")
        # Return basic profile on error
        return {
            "user_email": user_email,
            "timestamp": datetime.now().isoformat(),
            "features": {},
            "targets": [],
            "error": str(e)
        }


# ============================================================================
# STEP 4: GENERATE REPORT
# ============================================================================

def step_generate_report(
    burnout_data: Dict[str, Any],
    profile_data: Dict[str, Any],
    user_email: str,
    output_dir: Path
) -> Path:
    """
    Step 4: Generate HTML report.
    
    Args:
        burnout_data: Output from Step 2
        profile_data: Output from Step 3
        user_email: User email
        output_dir: Directory to save report
        
    Returns:
        Path to generated HTML report
    """
    print("\n" + "=" * 70)
    print("📊 STEP 4: Generating HTML Report")
    print("=" * 70)
    
    try:
        import sys
        sys.path.insert(0, str(SCRIPT_DIR))
        from generate_report import generate_single_report, FEATURE_LABELS
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename
        safe_email = user_email.replace("@", "_at_").replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"report_{safe_email}_{timestamp}.html"
        report_path = output_dir / report_name
        
        # Prepare data for report
        data = burnout_data.get("data", {})
        pred_class = burnout_data.get("burnout", {}).get("prediction_class", 0)
        probs = burnout_data.get("burnout", {}).get("all_probabilities", [0.33, 0.33, 0.33])
        
        # Load model for report generation (needed for what-if analysis)
        model_path = Path(BURNOUT_MODEL)
        if not model_path.exists():
            print(f"⚠️  Model not found for report generation: {model_path}")
            model = None
        else:
            from predict_burnout import load_model
            model, _, _, _ = load_model(model_path)
        
        # Generate single report
        print(f"📝 Creating report...")
        generate_single_report(
            data=data,
            person_name=user_email,
            model=model,
            feature_cols=list(FEATURE_LABELS.keys()),
            output_path=str(report_path)
        )
        
        print(f"✓ Report saved: {report_path.name}")
        
        return report_path
        
    except Exception as e:
        print(f"⚠️  Warning during report generation: {str(e)}")
        # Create a minimal report anyway
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        safe_email = user_email.replace("@", "_at_").replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"report_{safe_email}_{timestamp}.html"
        report_path = output_dir / report_name
        
        minimal_html = f"""
        <html>
            <head>
                <title>Burnout Risk Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .warning {{ color: #d9534f; }}
                </style>
            </head>
            <body>
                <h1>Burnout Risk Report</h1>
                <p>User: {user_email}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <div class="warning">
                    <p>Report generation encountered an issue, but predictions were completed.</p>
                    <p>Error: {str(e)}</p>
                </div>
                <h2>Predictions</h2>
                <pre>{json.dumps(burnout_data, indent=2)}</pre>
            </body>
        </html>
        """
        
        report_path.write_text(minimal_html)
        return report_path


# ============================================================================
# STEP 5: SAVE OUTPUTS
# ============================================================================

def step_save_outputs(
    burnout_data: Dict[str, Any],
    profile_data: Dict[str, Any],
    output_dir: Path,
    user_email: str
) -> Dict[str, Path]:
    """
    Step 5: Save all outputs (JSON, metadata).
    
    Args:
        burnout_data: Burnout predictions
        profile_data: Mental health profile
        output_dir: Output directory
        user_email: User email
        
    Returns:
        Dictionary with paths to all saved files
    """
    print("\n" + "=" * 70)
    print("💾 STEP 5: Saving Outputs")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_email = user_email.replace("@", "_at_").replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    # Save burnout predictions
    burnout_file = output_dir / f"burnout_{safe_email}_{timestamp}.json"
    with open(burnout_file, "w") as f:
        json.dump(burnout_data, f, indent=2, default=str)
    print(f"✓ Burnout predictions: {burnout_file.name}")
    saved_files["burnout"] = burnout_file
    
    # Save profile data
    profile_file = output_dir / f"profile_{safe_email}_{timestamp}.json"
    with open(profile_file, "w") as f:
        json.dump(profile_data, f, indent=2, default=str)
    print(f"✓ Mental health profile: {profile_file.name}")
    saved_files["profile"] = profile_file
    
    return saved_files


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_pipeline(
    csv_path: str | Path,
    user_email: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete burnout prediction pipeline.
    
    Args:
        csv_path: Path to Google Form CSV
        user_email: User email for identification
        output_dir: Directory to save outputs
        model_path: Optional custom model path
        
    Returns:
        Dictionary with all pipeline results
    """
    
    print("\n" + "=" * 70)
    print("🚀 BURNOUT PREDICTION PIPELINE STARTED")
    print("=" * 70)
    print(f"User: {user_email}")
    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")
    
    try:
        # Step 1: Parse
        df, is_daily = step_parse_form(csv_path)
        
        # Step 2: Burnout prediction
        burnout_data = step_predict_burnout(df, user_email, model_path)
        
        # Step 3: Profile
        profile_data = step_generate_profile(df, user_email)
        
        # Step 4: Report
        report_path = step_generate_report(burnout_data, profile_data, user_email, output_dir)
        
        # Step 5: Save
        saved_files = step_save_outputs(burnout_data, profile_data, output_dir, user_email)
        saved_files["report"] = report_path
        
        # Final summary
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📁 Generated files:")
        for file_type, file_path in saved_files.items():
            print(f"   • {file_type.capitalize()}: {file_path.name}")
        
        result = {
            "success": True,
            "user_email": user_email,
            "timestamp": datetime.now().isoformat(),
            "burnout": burnout_data.get("burnout", {}),
            "profile": profile_data,
            "files": {k: str(v) for k, v in saved_files.items()},
        }
        
        return result
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ PIPELINE FAILED")
        print("=" * 70)
        print(f"Error: {str(e)}")
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete burnout prediction pipeline (form → predictions → report)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/main_pipeline.py --csv responses.csv --user sarah@example.com
  
  # Custom output directory
  python scripts/main_pipeline.py --csv data.csv --user john@example.com --output-dir ./my_reports
  
  # Using custom model
  python scripts/main_pipeline.py --csv data.csv --user test@test.com --model-path ./my_model.pt
        """
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to Google Form CSV export"
    )
    
    parser.add_argument(
        "--user",
        type=str,
        required=True,
        help="User email (for identification and file naming)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to custom model .pt file (default: models/saved/lstm_sequence.pt)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    result = run_pipeline(
        csv_path=args.csv,
        user_email=args.user,
        output_dir=args.output_dir,
        model_path=args.model_path,
    )
    
    # Exit with appropriate code
    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()
