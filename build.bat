@echo off
echo ============================================================
echo   SIMPLE_AI - Build Script
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/4] Cleaning previous build...
if exist "dist\SIMPLE_AI" rmdir /s /q "dist\SIMPLE_AI"
if exist "build\SIMPLE_AI" rmdir /s /q "build\SIMPLE_AI"

echo [2/5] Generating app icon...
if exist tools\create_logo.py (
  ".\venv311_3.11\Scripts\python.exe" tools\create_logo.py
  if errorlevel 1 (
    echo   [WARN] Logo generation failed, continuing without icon.
  )
)

echo [3/5] Building executable...
".\venv311_3.11\Scripts\python.exe" -m PyInstaller SIMPLE_AI.spec --clean --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: Build failed. Check the output above.
    pause
    exit /b 1
)

echo [4/5] Copying user-data folders to dist...
if exist "models" (
    echo   - Copying models\  (this may take a while for large .gguf files)
    xcopy /E /I /Y "models" "dist\SIMPLE_AI\models" >nul
)
if exist "rag_databases" (
    echo   - Copying rag_databases\
    xcopy /E /I /Y "rag_databases" "dist\SIMPLE_AI\rag_databases" >nul
)

echo [5/5] Verifying bundled components...
if exist "dist\SIMPLE_AI\Tesseract-OCR\tesseract.exe" (
    echo   [OK] Tesseract OCR bundled
) else (
    echo   [WARN] Tesseract OCR not bundled - scanned PDF/image OCR will be disabled
)
if exist "dist\SIMPLE_AI\llama_cpp" (
    echo   [OK] llama-cpp-python bundled
) else (
    echo   [WARN] llama-cpp-python not bundled
)
if exist "dist\SIMPLE_AI\web\index.html" (
    echo   [OK] Web UI bundled
) else (
    echo   [WARN] Web UI not found!
)

echo.
echo ============================================================
echo   BUILD COMPLETE!
echo   Output: dist\SIMPLE_AI\SIMPLE_AI.exe
echo ============================================================
echo.
echo NOTE: Ollama models work if the user has Ollama installed
echo       and running.  Local GGUF models work out-of-the-box.
echo.
pause
