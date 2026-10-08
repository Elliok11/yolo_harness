@echo off
rem ============================================================
rem  Launch the camera GUI window WITHOUT any black console box.
rem  Double-click this file to start.
rem
rem  Optional: edit the CAMERA line below to change the source.
rem    0                     = built-in camera
rem    1                     = second camera
rem    http://192.168.1.5:8080/video   = phone camera
rem ============================================================

set CAMERA=0

cd /d "%~dp0"

rem pythonw.exe runs Python with no console window.
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0camera_gui.py" %CAMERA%
