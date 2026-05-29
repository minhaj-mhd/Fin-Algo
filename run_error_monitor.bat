@echo off
title VANGUARD SYSTEM — Real-Time Error & Warning Monitor
cd /d "%~dp0"
call env\Scripts\activate
cls
env\Scripts\python scripts/stream_errors.py
pause
