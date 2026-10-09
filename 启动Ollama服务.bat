@echo off
rem ============================================================
rem  Start the local Ollama service (CPU mode).
rem
rem  Both agent.py and n8n need this running to use the local
rem  qwen2.5 model. It listens on http://127.0.0.1:11434
rem
rem  WHY OLLAMA_VULKAN=false  (important, do not remove)
rem    Ollama auto-detected the AMD Radeon R7 M260 (2013, 2GB) via
rem    Vulkan and tried to run inference on it. The GPU driver is
rem    from 2020 and the combination crashes the inference process:
rem        0xc0000005  (access violation)
rem    Every request returned HTTP 500 until Vulkan was disabled.
rem    With the GPU off, inference works on CPU (~8s for a short reply).
rem
rem    If you later update the AMD driver and want to try the GPU
rem    again, just delete the OLLAMA_VULKAN line below and test.
rem
rem  Keep this window open while using the model.
rem  Close it (or press Ctrl+C) to stop the service.
rem ============================================================

cd /d "%~dp0"

set OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
set OLLAMA_VULKAN=false

if not exist "%OLLAMA_EXE%" (
    echo [ERROR] Ollama not found at:
    echo         %OLLAMA_EXE%
    echo.
    echo Install it from https://ollama.com/download
    pause
    exit /b 1
)

echo Starting Ollama service (CPU mode)...
echo Endpoint: http://127.0.0.1:11434
echo.
echo Keep this window open. Press Ctrl+C to stop.
echo.

"%OLLAMA_EXE%" serve
