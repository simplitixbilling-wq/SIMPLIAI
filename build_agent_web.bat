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

REM Generate logo / icon if tools\create_logo.py exists
if exist tools\create_logo.py (
  echo Generating app icon...
  "%PYEXE%" tools\create_logo.py
  if errorlevel 1 (
    echo WARNING: Logo generation failed, continuing without icon.
  )
)

REM Clean previous build artifacts
if exist build\SIMPLE_AI_WEB rmdir /s /q build\SIMPLE_AI_WEB
if exist dist\SIMPLE_AI_WEB rmdir /s /q dist\SIMPLE_AI_WEB

REM Build executable using the checked-in spec file
"%PYEXE%" -m PyInstaller --noconfirm --clean SIMPLE_AI_WEB.spec

if errorlevel 1 (
  echo ERROR: PyInstaller build failed.
  exit /b 1
)

REM Copy optional runtime folders if present (models intentionally excluded)
if exist rag_databases xcopy /E /I /Y rag_databases dist\SIMPLE_AI_WEB\rag_databases >nul
if exist plugins xcopy /E /I /Y plugins dist\SIMPLE_AI_WEB\plugins >nul
if exist processed_files xcopy /E /I /Y processed_files dist\SIMPLE_AI_WEB\processed_files >nul
if exist exports xcopy /E /I /Y exports dist\SIMPLE_AI_WEB\exports >nul
if exist saved_chats xcopy /E /I /Y saved_chats dist\SIMPLE_AI_WEB\saved_chats >nul

REM Tesseract OCR is bundled automatically by SIMPLE_AI_WEB.spec (datas block)

echo.
echo Build successful.
echo Executable: dist\SIMPLE_AI_WEB\SIMPLE_AI_WEB.exe
echo.
pause
exit /b 0
