@echo off
rem ============================================================
rem  Launch the desktop app (Electron window).
rem  Double-click this file.
rem
rem  What it does:
rem    1. clears ELECTRON_RUN_AS_NODE (otherwise Electron starts in
rem       plain Node mode and the window never appears)
rem    2. runs the Electron shell, which in turn starts the Python
rem       backend (web_server.py) and points the window at it
rem
rem  Requirements:
rem    - Ollama running            (double-click 启动Ollama服务.bat)
rem    - desktop\node_modules installed (npm install inside desktop\)
rem ============================================================

cd /d "%~dp0desktop"

rem This variable makes Electron behave like plain Node and break startup.
set ELECTRON_RUN_AS_NODE=

set ELECTRON_EXE=%~dp0desktop\node_modules\electron\dist\electron.exe

if not exist "%ELECTRON_EXE%" (
    echo [ERROR] Electron not found:
    echo         %ELECTRON_EXE%
    echo.
    echo Install it first:
    echo     cd desktop
    echo     npm install --registry=https://registry.npmmirror.com
    pause
    exit /b 1
)

echo Starting desktop app...
echo.

"%ELECTRON_EXE%" .
