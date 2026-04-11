@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo ===========================================
echo Building SIMPLE_AI_WEB from agent_web.py
echo ===========================================

REM Pick Python interpreter (prefer project venv)
set "PYEXE=%CD%\venv311_3.11\Scripts\python.exe"
if not exist "%PYEXE%" (
  set "PYEXE=python"
)

echo Using Python: %PYEXE%

REM Ensure pyinstaller is available
"%PYEXE%" -m pip install --upgrade pyinstaller
if errorlevel 1 (
  echo ERROR: Failed to install/upgrade pyinstaller.
  exit /b 1
)

REM Clean previous build artifacts
if exist build\SIMPLE_AI_WEB rmdir /s /q build\SIMPLE_AI_WEB
if exist dist\SIMPLE_AI_WEB rmdir /s /q dist\SIMPLE_AI_WEB
if exist SIMPLE_AI_WEB.spec del /q SIMPLE_AI_WEB.spec

REM Build executable (onedir)
"%PYEXE%" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onedir ^
  --name SIMPLE_AI_WEB ^
  --add-data "web;web" ^
  --add-data "instruction_templates.json;." ^
  --hidden-import webview ^
  --hidden-import requests ^
  --hidden-import bs4 ^
  --hidden-import pandas ^
  --hidden-import openpyxl ^
  --hidden-import reportlab ^
  --hidden-import docx ^
  --hidden-import fitz ^
  --hidden-import pytesseract ^
  --hidden-import PIL ^
  --hidden-import PIL.Image ^
  --hidden-import duckdb ^
  --hidden-import sklearn ^
  --hidden-import joblib ^
  --hidden-import numpy ^
  --hidden-import scipy ^
  --hidden-import scipy.sparse ^
  --hidden-import pyttsx3 ^
  --hidden-import pyttsx3.drivers ^
  --hidden-import pyttsx3.drivers.sapi5 ^
  --hidden-import pyttsx3.voice ^
  --hidden-import comtypes ^
  --hidden-import comtypes.client ^
  --hidden-import win32com ^
  --hidden-import win32com.client ^
  --collect-all llama_cpp ^
  --collect-all sklearn ^
  agent_web.py

if errorlevel 1 (
  echo ERROR: PyInstaller build failed.
  exit /b 1
)

REM Copy optional runtime folders if present
if exist models xcopy /E /I /Y models dist\SIMPLE_AI_WEB\models >nul
if exist rag_databases xcopy /E /I /Y rag_databases dist\SIMPLE_AI_WEB\rag_databases >nul
if exist plugins xcopy /E /I /Y plugins dist\SIMPLE_AI_WEB\plugins >nul
if exist processed_files xcopy /E /I /Y processed_files dist\SIMPLE_AI_WEB\processed_files >nul
if exist exports xcopy /E /I /Y exports dist\SIMPLE_AI_WEB\exports >nul
if exist saved_chats xcopy /E /I /Y saved_chats dist\SIMPLE_AI_WEB\saved_chats >nul

REM Bundle Tesseract OCR (if installed)
set "TESSDIR=%ProgramFiles%\Tesseract-OCR"
if exist "%TESSDIR%\tesseract.exe" (
  echo Bundling Tesseract OCR...
  if not exist "dist\SIMPLE_AI_WEB\Tesseract-OCR" mkdir "dist\SIMPLE_AI_WEB\Tesseract-OCR"
  if not exist "dist\SIMPLE_AI_WEB\Tesseract-OCR\tessdata" mkdir "dist\SIMPLE_AI_WEB\Tesseract-OCR\tessdata"
  copy /Y "%TESSDIR%\tesseract.exe" "dist\SIMPLE_AI_WEB\Tesseract-OCR\" >nul
  copy /Y "%TESSDIR%\*.dll" "dist\SIMPLE_AI_WEB\Tesseract-OCR\" >nul
  if exist "%TESSDIR%\tessdata\eng.traineddata" copy /Y "%TESSDIR%\tessdata\eng.traineddata" "dist\SIMPLE_AI_WEB\Tesseract-OCR\tessdata\" >nul
  if exist "%TESSDIR%\tessdata\osd.traineddata" copy /Y "%TESSDIR%\tessdata\osd.traineddata" "dist\SIMPLE_AI_WEB\Tesseract-OCR\tessdata\" >nul
  echo Tesseract bundled.
) else (
  echo NOTE: Tesseract not found at %TESSDIR% - OCR for scanned PDFs/images will not be available.
)

echo.
echo Build successful.
echo Executable: dist\SIMPLE_AI_WEB\SIMPLE_AI_WEB.exe
echo.
pause
exit /b 0
