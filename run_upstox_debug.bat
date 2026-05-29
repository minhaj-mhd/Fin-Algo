@echo off
title UPSTOX SANDBOX — Debug Console
cd /d "%~dp0"
call env\Scripts\activate
echo.
echo  Starting Upstox Debug Console...
echo  Type 'm' at any time to re-show the menu.
echo.
env\Scripts\python scripts\upstox_debug.py
pause
