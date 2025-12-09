#!/bin/bash
"""
Fast initialization script for dev container
Run this if post_create.sh doesn't run automatically
Usage: bash .devcontainer/init.sh
"""

echo "🚀 Initializing FDS Project Dev Container..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.11+"
    exit 1
fi

echo "✓ Python $(python --version | cut -d' ' -f2) found"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install pip"
    exit 1
fi

echo "✓ pip found"

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip -q

# Install all requirements
echo "📦 Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    echo "✓ Dependencies installed"
else
    echo "⚠️  requirements.txt not found, installing manually..."
    
    # Core ML
    pip install -q numpy pandas scikit-learn scipy
    
    # Deep Learning
    pip install -q torch torchvision
    
    # Interactive
    pip install -q streamlit plotly jupyter
    
    # Dev tools
    pip install -q pytest black flake8 pylint
    
    echo "✓ Manual installation complete"
fi

# Verify installation
echo ""
echo "✅ Verification:"
python -c "import torch, pandas, numpy, streamlit; print('  ✓ All core packages loaded successfully')" 2>/dev/null || echo "  ⚠️  Some packages may be missing"

echo ""
echo "🎉 Dev container initialized!"
echo ""
echo "Next steps:"
echo "  1. streamlit run demo_app.py"
echo "  2. python scripts/main_pipeline.py --csv test_responses.csv --user test@example.com"
echo ""
