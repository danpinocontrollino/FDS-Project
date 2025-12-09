#!/bin/bash
"""
Post-create hook for dev container
Runs automatically after container is created
Installs all dependencies and prepares the environment
"""

set -e

echo "=================================================="
echo "🚀 FDS PROJECT - DEV CONTAINER SETUP"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Update pip
echo -e "\n${BLUE}📦 Step 1: Updating pip...${NC}"
python -m pip install --upgrade pip setuptools wheel -q

# Step 2: Install core dependencies
echo -e "\n${BLUE}📦 Step 2: Installing core ML dependencies...${NC}"
pip install -q \
  numpy==1.24.3 \
  pandas==2.1.3 \
  scikit-learn==1.3.2 \
  scipy==1.11.4

echo -e "${GREEN}✓ Core dependencies installed${NC}"

# Step 3: Install Deep Learning
echo -e "\n${BLUE}📦 Step 3: Installing PyTorch...${NC}"
pip install -q \
  torch==2.1.0 \
  torchvision==0.16.0

echo -e "${GREEN}✓ PyTorch installed${NC}"

# Step 4: Install Interactive & Demo
echo -e "\n${BLUE}📦 Step 4: Installing interactive tools...${NC}"
pip install -q \
  streamlit==1.29.0 \
  plotly==5.18.0 \
  jupyter==1.0.0 \
  ipykernel==6.27.1 \
  ipywidgets==8.1.1

echo -e "${GREEN}✓ Interactive tools installed${NC}"

# Step 5: Install utilities
echo -e "\n${BLUE}📦 Step 5: Installing utilities...${NC}"
pip install -q \
  tqdm==4.66.1 \
  python-dateutil==2.8.2 \
  joblib==1.3.2

echo -e "${GREEN}✓ Utilities installed${NC}"

# Step 6: Install development tools
echo -e "\n${BLUE}📦 Step 6: Installing development tools...${NC}"
pip install -q \
  pytest==7.4.3 \
  black==23.12.0 \
  flake8==6.1.0 \
  pylint==3.0.3

echo -e "${GREEN}✓ Development tools installed${NC}"

# Step 7: Create demo model (optional, non-critical)
echo -e "\n${BLUE}🤖 Step 7: Creating demo LSTM model...${NC}"
if [ -f "scripts/create_demo_model.py" ]; then
  python scripts/create_demo_model.py 2>/dev/null || echo -e "${YELLOW}⚠️  Demo model creation skipped (non-critical)${NC}"
else
  echo -e "${YELLOW}⚠️  create_demo_model.py not found${NC}"
fi

# Step 8: Create sample data (optional, non-critical)
echo -e "\n${BLUE}📊 Step 8: Creating sample test data...${NC}"
if [ -f "scripts/create_sample_data.py" ]; then
  python scripts/create_sample_data.py 2>/dev/null || echo -e "${YELLOW}⚠️  Sample data creation skipped (non-critical)${NC}"
else
  echo -e "${YELLOW}⚠️  create_sample_data.py not found${NC}"
fi

# Step 9: Verify installation
echo -e "\n${BLUE}✓ Step 9: Verifying installation...${NC}"

python -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  ⚠️  PyTorch"
python -c "import pandas; print(f'  Pandas: {pandas.__version__}')" 2>/dev/null || echo "  ⚠️  Pandas"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>/dev/null || echo "  ⚠️  NumPy"
python -c "import streamlit; print(f'  Streamlit: {streamlit.__version__}')" 2>/dev/null || echo "  ⚠️  Streamlit"
python -c "import sklearn; print(f'  Scikit-learn: {sklearn.__version__}')" 2>/dev/null || echo "  ⚠️  Scikit-learn"

echo ""
echo "=================================================="
echo -e "${GREEN}✅ DEV CONTAINER SETUP COMPLETE!${NC}"
echo "=================================================="
echo ""
echo -e "${BLUE}📚 Next Steps:${NC}"
echo ""
echo "1. Run the demo:"
echo "   streamlit run demo_app.py"
echo ""
echo "2. Or run the full pipeline:"
echo "   python scripts/main_pipeline.py --csv test_responses.csv --user test@example.com"
echo ""
echo "3. Or run tests:"
echo "   python scripts/validate_pipeline.py"
echo ""
echo "=================================================="
echo ""
