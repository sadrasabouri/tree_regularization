#!/bin/bash

# TreeReg Environment Setup Script
# This script sets up the environment for running TreeReg

set -e  # Exit on any error

echo "TreeReg Environment Setup"
echo "========================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Found Python $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

echo "✓ Found pip3"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ Pip upgraded"

# Install PyTorch first (required for other packages)
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch installed"

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install transformers datasets accelerate wandb einops rotary-embedding-torch torch-struct
echo "✓ Core dependencies installed"

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install spacy benepar nltk scikit-learn matplotlib seaborn
echo "✓ Additional dependencies installed"

# Install the package in editable mode
echo ""
echo "Installing TreeReg package..."
pip install -e .
echo "✓ TreeReg package installed"

# Test the installation
echo ""
echo "Testing installation..."
python example_usage.py
echo "✓ Installation test passed"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the example:"
echo "  python example_usage.py"
echo ""
echo "To start training, see the commands in README.md"
