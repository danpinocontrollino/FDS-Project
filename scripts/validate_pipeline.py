#!/usr/bin/env python
"""
Quick validation script for main_pipeline.py
Checks that all required modules and functions are present.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath.name}")
        return True
    else:
        print(f"✗ {description}: NOT FOUND - {filepath}")
        return False

def check_function_exists(module_path, function_name, description):
    """Check if a function exists in a module."""
    try:
        with open(module_path, 'r') as f:
            content = f.read()
            if f"def {function_name}(" in content:
                print(f"✓ {description}: {function_name}()")
                return True
            else:
                print(f"✗ {description}: Function {function_name}() NOT FOUND")
                return False
    except Exception as e:
        print(f"✗ {description}: Error reading file - {e}")
        return False

def validate_pipeline():
    """Validate that main_pipeline.py is properly structured."""
    
    print("\n" + "=" * 70)
    print("VALIDATING MAIN_PIPELINE.PY STRUCTURE")
    print("=" * 70 + "\n")
    
    all_good = True
    
    # Check required files exist
    print("📂 Checking required files...")
    required_files = [
        (SCRIPT_DIR / "form_parser.py", "Form Parser Module"),
        (SCRIPT_DIR / "predict_burnout.py", "Burnout Predictor"),
        (SCRIPT_DIR / "generate_profile.py", "Profile Generator"),
        (SCRIPT_DIR / "generate_report.py", "Report Generator"),
        (SCRIPT_DIR / "personalized_advisor.py", "Personalized Advisor"),
        (SCRIPT_DIR / "main_pipeline.py", "Main Pipeline"),
    ]
    
    for filepath, desc in required_files:
        if not check_file_exists(filepath, desc):
            all_good = False
    
    print()
    
    # Check main_pipeline.py has all required functions
    print("🔍 Checking main_pipeline.py functions...")
    pipeline_functions = [
        ("step_parse_form", "Step 1: Parse Form"),
        ("step_predict_burnout", "Step 2: Predict Burnout"),
        ("step_generate_profile", "Step 3: Generate Profile"),
        ("step_generate_report", "Step 4: Generate Report"),
        ("step_save_outputs", "Step 5: Save Outputs"),
        ("run_pipeline", "Pipeline Orchestrator"),
        ("parse_args", "CLI Parser"),
        ("main", "Main Entry Point"),
    ]
    
    pipeline_file = SCRIPT_DIR / "main_pipeline.py"
    for func_name, desc in pipeline_functions:
        if not check_function_exists(pipeline_file, func_name, desc):
            all_good = False
    
    print()
    
    # Check form_parser.py has GoogleFormParser class
    print("🔍 Checking form_parser.py structure...")
    form_parser_file = SCRIPT_DIR / "form_parser.py"
    try:
        with open(form_parser_file, 'r') as f:
            content = f.read()
            if "class GoogleFormParser:" in content:
                print("✓ GoogleFormParser class exists")
            else:
                print("✗ GoogleFormParser class NOT FOUND")
                all_good = False
            
            if "def parse_google_form_csv(" in content:
                print("✓ parse_google_form_csv() method exists")
            else:
                print("✗ parse_google_form_csv() NOT FOUND")
                all_good = False
    except Exception as e:
        print(f"✗ Error reading form_parser.py: {e}")
        all_good = False
    
    print()
    
    # Check test CSV exists
    print("📊 Checking test data...")
    test_csv = PROJECT_ROOT / "test_responses.csv"
    if check_file_exists(test_csv, "Test CSV"):
        with open(test_csv, 'r') as f:
            lines = f.readlines()
            print(f"   Rows: {len(lines) - 1} (excluding header)")
    else:
        all_good = False
    
    print()
    
    # Summary
    print("=" * 70)
    if all_good:
        print("✅ ALL CHECKS PASSED - Pipeline is ready!")
        print("\nYou can now run:")
        print("  python scripts/main_pipeline.py --csv test_responses.csv --user test@example.com")
    else:
        print("❌ SOME CHECKS FAILED - Please review above")
        sys.exit(1)
    print("=" * 70 + "\n")

if __name__ == "__main__":
    validate_pipeline()
