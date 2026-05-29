@echo off
title VANGUARD INTELLIGENCE SYSTEM BOOTSTRAPPER
color 0B
cd /d "%~dp0"

echo ============================================================
echo   Vanguard Intelligence System: Bootstrapping & Launcher
echo ============================================================
echo.

:: 1. Run Python Bootstrap script
python bootstrap.py
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo [ERROR] Bootstrapping failed! Please fix the errors above and retry.
    echo.
    pause
    exit /b %ERRORLEVEL%
)

:: 2. Prompt user to launch
echo.
echo ============================================================
echo   Environment successfully verified and prepared!
echo ============================================================
echo.
set /p launch="Do you want to start the Vanguard Intelligence System now? (Y/N): "
if /i "%launch%"=="Y" goto START_SYSTEM
if /i "%launch%"=="" goto START_SYSTEM
echo.
echo Exiting without launching. You can launch manually anytime using:
echo    Start Dashboard:  env\Scripts\python scripts/vanguard_dashboard.py
echo    Start Engine:     env\Scripts\python scripts/vanguard_signal_engine.py
echo.
pause
exit /b 0

:START_SYSTEM
echo.
echo Launching Vanguard Dashboard in background window...
start "Vanguard Dashboard" cmd /k "color 0E && env\Scripts\python scripts/vanguard_dashboard.py"

echo.
echo Launching Vanguard Signal Engine in this window...
echo Press Ctrl+C to terminate both components.
echo.
timeout /t 2 >nul
env\Scripts\python scripts/vanguard_signal_engine.py

pause
