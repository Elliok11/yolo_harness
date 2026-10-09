@echo off
rem ============================================================
rem  Start the local Ollama service (CPU mode).
rem
rem  Both agent.py and the desktop app need this running to use
rem  the local qwen2.5 model. It listens on http://127.0.0.1:11434
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
rem  HOW TO TELL IT IS READY:
rem    This window stays open with no further output once the service
rem    is up. The startup line "Listening on 127.0.0.1:11434" means
rem    it is serving requests. To double-check, open a browser at
rem    http://127.0.0.1:11434  -- it should say "Ollama is running".
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

echo ============================================================
echo   Starting Ollama service (CPU mode)
echo ============================================================
echo   Endpoint : http://127.0.0.1:11434
echo   Model    : qwen2.5:1.5b
echo.
echo   When it is ready you will see a line like:
echo       Listening on 127.0.0.1:11434
echo   After that this window just sits there quietly -- that is
echo   NORMAL. It is not frozen. Do not close it while using the app.
echo.
echo   Press Ctrl+C to stop the service.
echo ============================================================
echo.

"%OLLAMA_EXE%" serve
