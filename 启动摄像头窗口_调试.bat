@echo off
rem ============================================================
rem  Same as the silent launcher, but KEEPS the console open so
rem  you can read any error message. Use this one when the window
rem  does not appear or something crashes.
rem ============================================================

set CAMERA=0

cd /d "%~dp0"

echo Starting camera window (debug mode)...
echo If the window does not appear, the error is printed below.
echo.

"%~dp0venv\Scripts\python.exe" "%~dp0camera_gui.py" %CAMERA%

echo.
echo Program exited. Press any key to close this window.
pause >nul
