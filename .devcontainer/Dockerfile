FROM mcr.microsoft.com/devcontainers/python:3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir \
    # Core ML
    numpy==1.24.3 \
    pandas==2.1.3 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    # Deep Learning
    torch==2.1.0 \
    torchvision==0.16.0 \
    # Interactive & Demo
    streamlit==1.29.0 \
    plotly==5.18.0 \
    jupyter==1.0.0 \
    ipykernel==6.27.1 \
    ipywidgets==8.1.1 \
    # Utilities
    tqdm==4.66.1 \
    python-dateutil==2.8.2 \
    joblib==1.3.2 \
    # Development
    pytest==7.4.3 \
    black==23.12.0 \
    flake8==6.1.0 \
    pylint==3.0.3 \
    jupyter-contrib-nbextensions

# Create working directory
WORKDIR /workspaces/FDS-Project

# Copy project files (if building from local)
# COPY . .

# Expose ports
EXPOSE 8501 5000 8000

# Set up the environment
CMD ["/bin/bash"]
