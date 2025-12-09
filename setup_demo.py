#!/usr/bin/env python3
"""
Setup script for Streamlit demo
Installs dependencies and creates demo model
Run with: python setup_demo.py
"""

import sys
import subprocess
from pathlib import Path
import json

def run_command(cmd, description):
    """Run a shell command and report results."""
    print(f"\n📍 {description}")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✓ {description} - Success")
            return True
        else:
            print(f"   ✗ {description} - Failed")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"   ✗ {description} - Exception: {e}")
        return False

def check_file(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"   ✓ {description} found: {path}")
        return True
    else:
        print(f"   ✗ {description} NOT FOUND: {path}")
        return False

def main():
    """Main setup function."""
    
    print("=" * 50)
    print("🚀 FDS PROJECT - STREAMLIT DEMO SETUP")
    print("=" * 50)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    
    # Step 1: Install dependencies
    print("\n" + "=" * 50)
    print("STEP 1: Install Dependencies")
    print("=" * 50)
    
    packages = [
        "torch",
        "pandas",
        "numpy",
        "scikit-learn",
        "streamlit",
        "plotly",
        "jupyter"
    ]
    
    print(f"\nInstalling {len(packages)} packages (this may take 2-3 minutes)...")
    
    for package in packages:
        print(f"  • Installing {package}...", end=" ", flush=True)
        if run_command(f"pip install -q {package}", f"Install {package}"):
            print("✓")
        else:
            print("⚠️  (may already be installed)")
    
    # Step 2: Create demo model
    print("\n" + "=" * 50)
    print("STEP 2: Create Demo LSTM Model")
    print("=" * 50)
    
    print("\nGenerating demo model...")
    result = subprocess.run(
        [sys.executable, "scripts/create_demo_model.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"✗ Failed to create demo model")
        print(result.stderr)
        return False
    
    # Step 3: Create sample data
    print("\n" + "=" * 50)
    print("STEP 3: Create Sample Data")
    print("=" * 50)
    
    print("\nGenerating sample profiles...")
    result = subprocess.run(
        [sys.executable, "scripts/create_sample_data.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"⚠️  Failed to create sample data (non-critical)")
    
    # Step 4: Verify configuration
    print("\n" + "=" * 50)
    print("STEP 4: Verify Configuration")
    print("=" * 50)
    
    all_good = True
    
    print("\nChecking configuration files...")
    all_good &= check_file("config/job_categories.json", "Job categories")
    all_good &= check_file("config/thresholds.json", "Thresholds")
    
    print("\nChecking model files...")
    all_good &= check_file("models/saved/mental_health_lstm.pt", "LSTM model")
    
    print("\nChecking demo app...")
    all_good &= check_file("demo_app.py", "Streamlit app")
    
    print("\nChecking documentation...")
    check_file("DEMO_README.md", "Demo README")
    
    # Final summary
    print("\n" + "=" * 50)
    if all_good:
        print("✅ SETUP COMPLETE!")
        print("=" * 50)
        print("\n🎬 To start the demo, run:\n")
        print("   streamlit run demo_app.py\n")
        print("📖 For more information, see: DEMO_README.md\n")
        return True
    else:
        print("⚠️  SETUP INCOMPLETE - Some files missing")
        print("=" * 50)
        print("\nPlease check the errors above and try again.\n")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
