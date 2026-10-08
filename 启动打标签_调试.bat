@echo off
rem ============================================================
rem  Same as the silent launcher, but keeps the console open so
rem  you can read error messages. Use when the window does not
rem  appear or something crashes.
rem ============================================================

cd /d "%~dp0"

echo Starting labelling window (debug mode)...
echo If the window does not appear, the error is printed below.
echo.

"%~dp0venv\Scripts\python.exe" "%~dp0label_gui.py"

echo.
echo Program exited. Press any key to close this window.
pause >nul
