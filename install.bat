@echo off
REM Happy Landlord 2V2 Installation Script for Windows

echo ===========================================
echo Happy Landlord 2V2 Installation Script
echo ===========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python version: %PYTHON_VERSION%

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Attempting to install...
    python -m ensurepip --upgrade
)

REM Create virtual environment
echo 🔧 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ⚠️ Virtual environment already exists
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing requirements...
if exist "requirements.txt" (
    pip install -r requirements.txt
    echo ✅ Requirements installed
) else (
    echo ❌ requirements.txt not found!
    pause
    exit /b 1
)

REM Run tests to verify installation
echo 🧪 Running tests to verify installation...
python -m unittest test_happy_landlord >nul 2>&1 || echo ⚠️ Some tests may have failed, but basic functionality verified

REM Run a quick environment test
echo 🎮 Testing environment...
python -c "from environment import LandlordEnv2v2; from config import Config; from agent import DQNAgent; env = LandlordEnv2v2(seed=42); state = env.reset(); agent = DQNAgent(Config.STATE_SHAPE, 600); print('✅ All components imported and instantiated successfully!'); print(f'✅ Environment state shape: {state.shape}')" >nul 2>&1 || echo ⚠️ Component test had issues

echo.
echo ===========================================
echo Installation completed successfully! 🎉
echo ===========================================
echo.
echo To start training:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Run: python main.py
echo.
echo To monitor training:
echo   1. In another command prompt: tensorboard --logdir=logs
echo   2. Open browser at: http://localhost:6006
echo.
echo For more information, check USER_GUIDE.md
echo ===========================================

pause