@echo off
rem ============================================================
rem  Launch the labelling window (draw boxes on images).
rem  Double-click this file to start.
rem
rem  Labels are saved to the "labels" folder as standard YOLO
rem  txt files, one per image. No existing feature is touched.
rem ============================================================

cd /d "%~dp0"

start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0label_gui.py"
