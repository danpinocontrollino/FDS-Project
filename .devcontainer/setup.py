#!/usr/bin/env python3
"""
Emergency setup script for dev container
Runs in Python instead of bash for better compatibility
Usage: python .devcontainer/setup.py
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, description):
    """Run a command and report status."""
    print(f"\n📍 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✓ Success")
            return True
        else:
            print(f"   ⚠️  {result.stderr[:100] if result.stderr else 'Error'}")
            return False
    except Exception as e:
        print(f"   ✗ Exception: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 FDS PROJECT - EMERGENCY DEV CONTAINER SETUP")
    print("=" * 60)
    
    # Check Python
    print(f"\n✓ Python: {sys.version.split()[0]}")
    
    # Step 1: Upgrade pip
    print("\n" + "=" * 60)
    print("STEP 1: Upgrade pip")
    print("=" * 60)
    
    run_cmd(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Install requirements
    print("\n" + "=" * 60)
    print("STEP 2: Install Dependencies")
    print("=" * 60)
    
    reqs_path = Path("requirements.txt")
    if reqs_path.exists():
        print("\n📦 Installing from requirements.txt...")
        run_cmd(f"{sys.executable} -m pip install -r requirements.txt -q", "Installing requirements")
    else:
        print("\n📦 Installing core packages...")
        packages = [
            "torch pandas numpy scikit-learn scipy",
            "streamlit plotly jupyter",
            "pytest black flake8 pylint"
        ]
        for pkg_group in packages:
            run_cmd(f"{sys.executable} -m pip install -q {pkg_group}", f"Installing {pkg_group.split()[0]}")
    
    # Step 3: Verify
    print("\n" + "=" * 60)
    print("STEP 3: Verification")
    print("=" * 60)
    
    print("\n📦 Checking imports...")
    try:
        import torch
        print(f"   ✓ torch {torch.__version__}")
    except ImportError:
        print("   ⚠️  torch")
    
    try:
        import pandas
        print(f"   ✓ pandas {pandas.__version__}")
    except ImportError:
        print("   ⚠️  pandas")
    
    try:
        import numpy
        print(f"   ✓ numpy {numpy.__version__}")
    except ImportError:
        print("   ⚠️  numpy")
    
    try:
        import streamlit
        print(f"   ✓ streamlit {streamlit.__version__}")
    except ImportError:
        print("   ⚠️  streamlit")
    
    # Step 4: Create demo model
    print("\n" + "=" * 60)
    print("STEP 4: Create Demo Model")
    print("=" * 60)
    
    demo_script = Path("scripts/create_demo_model.py")
    if demo_script.exists():
        run_cmd(f"{sys.executable} scripts/create_demo_model.py", "Creating demo model")
    else:
        print("   ⚠️  create_demo_model.py not found")
    
    # Step 5: Create sample data
    print("\n" + "=" * 60)
    print("STEP 5: Create Sample Data")
    print("=" * 60)
    
    sample_script = Path("scripts/create_sample_data.py")
    if sample_script.exists():
        run_cmd(f"{sys.executable} scripts/create_sample_data.py", "Creating sample data")
    else:
        print("   ⚠️  create_sample_data.py not found")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\n🎬 Next steps:")
    print("   1. streamlit run demo_app.py")
    print("   2. Or: python scripts/main_pipeline.py --csv test_responses.csv --user test@example.com")
    print()

if __name__ == "__main__":
    main()
