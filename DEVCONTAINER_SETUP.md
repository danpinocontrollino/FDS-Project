# Dev Container Setup Instructions

## 🚀 Automatic Setup (VS Code Dev Containers)

### Step 1: Install Extension
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Dev Containers"
4. Install `ms-vscode-remote.remote-containers`

### Step 2: Reopen in Container
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P`)
2. Type "Dev Containers: Reopen in Container"
3. Wait for automatic setup (~5-10 minutes)

**That's it!** Everything will be installed automatically.

---

## 🔧 Manual Setup (If Automatic Fails)

### Option A: Python Setup Script (Easiest)
```bash
python .devcontainer/setup.py
```
This runs everything automatically using Python (more compatible than bash).

### Option B: Bash Setup Script
```bash
chmod +x .devcontainer/init.sh
bash .devcontainer/init.sh
```

### Option C: Manual Installation
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Create demo model
python scripts/create_demo_model.py

# Create sample data
python scripts/create_sample_data.py
```

---

## ✅ Verify Installation

```bash
# Check Python
python --version

# Check pip
pip --version

# Check key packages
python -c "import torch, pandas, numpy, streamlit; print('✓ All packages OK')"
```

---

## 🎬 Run the Demo

Once setup is complete:

```bash
# Start Streamlit demo
streamlit run demo_app.py

# Then open: http://localhost:8501
```

---

## 🐳 Docker Setup (Alternative)

If you prefer Docker:

```bash
# Build image
docker build -f .devcontainer/Dockerfile -t fds-project .

# Run container
docker run -it -p 8501:8501 -v $(pwd):/workspaces/FDS-Project fds-project
```

---

## ⚠️ Troubleshooting

### "python: command not found"
- Python not installed in container
- Try: `apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev`

### "pip: command not found"
- pip not installed
- Try: `python -m ensurepip --upgrade`

### Streamlit not found
- Didn't install streamlit
- Run: `pip install streamlit`

### Port 8501 already in use
- Change port: `streamlit run demo_app.py --server.port 8502`

---

## 📁 What Gets Installed

### Data Science & ML
- PyTorch 2.1.0
- NumPy, Pandas, Scikit-learn

### Interactive & Demo
- Streamlit
- Plotly, Jupyter

### Development
- pytest, black, flake8, pylint

### Utilities
- tqdm, joblib, python-dateutil

---

## 📖 Full Documentation

See `.devcontainer/README.md` for complete information.

