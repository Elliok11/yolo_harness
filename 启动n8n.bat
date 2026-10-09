@echo off
rem ============================================================
rem  Start n8n (the low-code workflow / agent builder).
rem
rem  Web UI:  http://localhost:5678
rem  Data:    %USERPROFILE%\.n8n
rem
rem  Keep this window open while using n8n.
rem  Press Ctrl+C to stop the service.
rem ============================================================

cd /d "%~dp0"

rem Turn off telemetry and version checks: this machine is old and
rem the extra network calls only slow startup down.
set N8N_DIAGNOSTICS_ENABLED=false
set N8N_VERSION_NOTIFICATIONS_ENABLED=false
set N8N_DEFAULT_BINARY_DATA_MODE=filesystem

rem Local access is plain http, so the secure-cookie check must be off
rem or the browser will refuse to log in.
set N8N_SECURE_COOKIE=false

set N8N_HOST=127.0.0.1
set N8N_PORT=5678
set GENERIC_TIMEZONE=Asia/Shanghai

set N8N_CMD=%APPDATA%\npm\n8n.cmd

if not exist "%N8N_CMD%" (
    echo [ERROR] n8n not found at:
    echo         %N8N_CMD%
    echo.
    echo Install it with:
    echo   npm install -g n8n --ignore-scripts --registry=https://registry.npmmirror.com
    pause
    exit /b 1
)

echo Starting n8n...
echo.
echo   Web UI:  http://localhost:5678
echo.
echo Keep this window open. Press Ctrl+C to stop.
echo.

call "%N8N_CMD%" start
