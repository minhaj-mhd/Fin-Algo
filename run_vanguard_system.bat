@echo off
title FINALGO VANGUARD SYSTEM
echo Starting FinalGo Vanguard Intelligence...
echo ========================================

:: Ensure we are in the right directory
cd /d "%~dp0"

:: Activate Environment
call env\Scripts\activate

:: Start Dashboard in Background Terminal
start "Vanguard Dashboard" cmd /k "env\Scripts\python scripts/vanguard_dashboard.py"

:: Start Signal Engine in Main Window
echo Launching Vanguard Signal Engine (15m Candle-Aligned Scan + PrimoGPT)...
env\Scripts\python scripts/vanguard_signal_engine.py

pause
