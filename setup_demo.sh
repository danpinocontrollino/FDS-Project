#!/bin/bash
"""
Setup script for Streamlit demo
Installs dependencies and creates demo model
"""

echo "🚀 FDS Project - Demo Setup"
echo "================================"

# Check Python
echo ""
echo "✓ Checking Python..."
python --version

# Install requirements
echo ""
echo "📦 Installing dependencies..."
echo "   (This may take 2-3 minutes)"

pip install -q torch pandas numpy scikit-learn streamlit plotly jupyter

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Create demo model
echo ""
echo "🤖 Creating demo LSTM model..."
python scripts/create_demo_model.py

if [ $? -eq 0 ]; then
    echo "✓ Demo model ready"
else
    echo "✗ Failed to create demo model"
    exit 1
fi

# Verify config
echo ""
echo "📋 Verifying configuration..."

if [ -f "config/job_categories.json" ] && [ -f "config/thresholds.json" ]; then
    echo "✓ Config files found"
else
    echo "✗ Config files missing"
    exit 1
fi

# Final summary
echo ""
echo "================================"
echo "✅ Setup Complete!"
echo "================================"
echo ""
echo "🎬 To start the demo, run:"
echo ""
echo "   streamlit run demo_app.py"
echo ""
echo "The app will open at: http://localhost:8501"
echo ""
echo "📖 For more info, see: DEMO_README.md"
echo ""
