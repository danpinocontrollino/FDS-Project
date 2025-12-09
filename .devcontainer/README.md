# 🐳 Dev Container Configuration

This directory contains configuration files for setting up a consistent development environment for the FDS Project.

## Files

### `devcontainer.json`
Main configuration for VS Code Dev Containers
- Specifies Python 3.11 image
- Configures port forwarding (8501 for Streamlit, 5000 for Flask, 8000 for dev server)
- Installs VS Code extensions
- Runs post-create setup script

### `post_create.sh`
Automatic setup script that runs after container is created
- Updates pip
- Installs all Python dependencies
- Creates demo LSTM model
- Generates sample test data
- Verifies installation

### `Dockerfile` (Optional)
Alternative Dockerfile for building custom image
- Based on official Python 3.11 image
- Pre-installs all dependencies
- Useful if Docker build is preferred

### `init.sh`
Fast initialization script (run manually if needed)
- Use if post_create.sh doesn't run automatically
- Quick dependency installation
- Verification checks

## Quick Start

### Option 1: Using VS Code Dev Containers (Recommended)

1. **Open project in VS Code**
   ```bash
   code /workspaces/FDS-Project
   ```

2. **Install Extension**
   - Install "Dev Containers" extension (ms-vscode-remote.remote-containers)

3. **Reopen in Container**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Dev Containers: Reopen in Container"
   - Wait for automatic setup (5-10 minutes)

4. **Done!**
   - Everything is installed automatically
   - Terminal is ready to use

### Option 2: Manual Setup

If post_create.sh doesn't run automatically:

```bash
# Make init script executable
chmod +x .devcontainer/init.sh

# Run initialization
bash .devcontainer/init.sh
```

### Option 3: Using Docker

```bash
# Build custom image
docker build -f .devcontainer/Dockerfile -t fds-project:latest .

# Run container
docker run -it -p 8501:8501 -v $(pwd):/workspaces/FDS-Project fds-project:latest
```

## Environment Variables

The dev container automatically sets:
- `PYTHONUNBUFFERED=1` - Real-time output
- `PYTHONDONTWRITEBYTECODE=1` - No .pyc files
- `PIP_NO_CACHE_DIR=1` - Smaller image size

## Port Forwarding

Auto-forwarding for:
- **8501** - Streamlit app
- **5000** - Flask development server
- **8000** - Generic development server

## Installed Tools

### Core ML/Data Science
- PyTorch 2.1.0
- NumPy 1.24.3
- Pandas 2.1.3
- Scikit-learn 1.3.2
- SciPy 1.11.4

### Interactive & Demo
- Streamlit 1.29.0
- Plotly 5.18.0
- Jupyter 1.0.0
- IPyKernel 6.27.1
- IPyWidgets 8.1.1

### Development Tools
- pytest 7.4.3
- black 23.12.0
- flake8 6.1.0
- pylint 3.0.3

### Utilities
- tqdm 4.66.1
- python-dateutil 2.8.2
- joblib 1.3.2

## VS Code Extensions

Automatically installed:
- Python
- Pylance
- Debugpy
- Ruff
- Jupyter
- GitLens

## Troubleshooting

### Port already in use
```bash
# Use different port
streamlit run demo_app.py --server.port 8502
```

### Container creation failed
```bash
# Clean up and rebuild
docker system prune -a
# Then reopen in container
```

### Dependencies not installed
```bash
# Run manual setup
bash .devcontainer/init.sh
```

### Python/pip not found
```bash
# Verify Python installation
python --version
pip --version

# If not found, install system Python first
```

## File Structure

```
.devcontainer/
├── devcontainer.json    # VS Code Dev Container config
├── post_create.sh       # Auto-run setup script
├── init.sh             # Manual setup script
├── Dockerfile          # Optional Docker build
└── README.md           # This file
```

## After Setup

Once container is ready:

1. **Run the demo:**
   ```bash
   streamlit run demo_app.py
   ```

2. **Run the pipeline:**
   ```bash
   python scripts/main_pipeline.py --csv test_responses.csv --user test@example.com
   ```

3. **Run tests:**
   ```bash
   python scripts/validate_pipeline.py
   ```

4. **Develop:**
   - All Python tools are available
   - Git integration works
   - SSH keys available for GitHub

## Documentation

For more information:
- See `DEMO_README.md` for demo documentation
- See `DEMO_SETUP_COMPLETE.md` for setup details
- See `MAIN_PIPELINE_README.md` for pipeline documentation

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review VS Code Dev Containers documentation
3. Check Python and pip are installed correctly
4. Verify internet connection for package downloads

---

**Status:** ✅ Dev Container fully configured and ready to use
