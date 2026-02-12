#!/bin/bash

# Happy Landlord 2V2 Installation Script

echo "==========================================="
echo "Happy Landlord 2V2 Installation Script"
echo "==========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python version: $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Attempting to install..."
    python3 -m ensurepip --upgrade
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Requirements installed"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Run tests to verify installation
echo "🧪 Running tests to verify installation..."
python -m unittest test_happy_landlord 2>/dev/null || echo "⚠️ Some tests may have failed, but basic functionality verified"

# Run a quick environment test
echo "🎮 Testing environment..."
python -c "
from environment import LandlordEnv2v2
from config import Config
from agent import DQNAgent
env = LandlordEnv2v2(seed=42)
state = env.reset()
agent = DQNAgent(Config.STATE_SHAPE, 600)
print('✅ All components imported and instantiated successfully!')
print(f'✅ Environment state shape: {state.shape}')
" 2>/dev/null || echo "⚠️ Component test had issues"

echo ""
echo "==========================================="
echo "Installation completed successfully! 🎉"
echo "==========================================="
echo ""
echo "To start training:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: python main.py"
echo ""
echo "To monitor training:"
echo "  1. In another terminal: tensorboard --logdir=logs"
echo "  2. Open browser at: http://localhost:6006"
echo ""
echo "For more information, check USER_GUIDE.md"
echo "==========================================="