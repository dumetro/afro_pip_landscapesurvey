@echo off
echo WHO AFRO Influenza Landscape Survey Dashboard
echo ============================================

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Installing requirements...
pip install -r requirements.txt

echo Starting Streamlit dashboard...
echo.
echo The dashboard will open in your default browser
echo Press Ctrl+C to stop the dashboard
echo.

streamlit run app.py