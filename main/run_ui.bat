@echo off
rem Windows batch script to run the Streamlit dashboard

rem Determine script directory
set SCRIPT_DIR=%~dp0
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

rem Get project root directory
for %%I in ("%SCRIPT_DIR%\..") do set PROJECT_ROOT=%%~fI

rem Change directory to project root
cd /d "%PROJECT_ROOT%"

rem Set environment variables
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONPATH=main

echo Checking Python environment...

rem Check if python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not found on PATH.
    pause
    exit /b 1
)

rem Run Streamlit dashboard using Python
echo Starting Streamlit dashboard...
python -m streamlit run main/src/ui/dashboard.py %*
