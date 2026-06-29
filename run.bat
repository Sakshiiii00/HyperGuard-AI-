@echo off
echo ============================================================
echo   HyperGuard AI - Hybrid Malware Detection Framework
echo ============================================================

:: Check if venv exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Run: python -m venv .venv
    echo Then: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Activate venv and run
echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install TF if missing
python -c "import tensorflow" 2>nul || (
    echo [*] TensorFlow not found in venv. Installing...
    pip install tensorflow --quiet
)

echo [*] Starting Flask API server...
echo [*] Open browser at: http://127.0.0.1:5000
echo.
python app/app.py

pause
