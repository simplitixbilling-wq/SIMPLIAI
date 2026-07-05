@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

REM Prefer project venv Python when available.
set "PYTHON_EXE=python"
if exist ".\venv311_3.11\Scripts\python.exe" set "PYTHON_EXE=.\venv311_3.11\Scripts\python.exe"

set "OUT_FILE=generated_activation_key.txt"
set "DEFAULT_APP_SECRET=SIMPLIAI-FULL-ACCESS"
set "GENERATOR_SCRIPT=tools\generate_passkey.py"

:main
cls
echo ================================================
echo        SIMPLIAI Activation Key Generator
echo ================================================
echo.

REM Auto-pick master secret from environment first, then from activation_passkey.txt.
set "MASTER_SECRET=%SIMPLIAI_PASSKEY%"
if not defined MASTER_SECRET (
    if exist "activation_passkey.txt" (
        set /p MASTER_SECRET=<"activation_passkey.txt"
    )
)

if not defined MASTER_SECRET (
    set "MASTER_SECRET=%DEFAULT_APP_SECRET%"
    echo No master secret configured in env/file.
    echo Using app default secret: %DEFAULT_APP_SECRET%
    echo.
    echo Tip: For custom secret, set SIMPLIAI_PASSKEY or create activation_passkey.txt
    echo.
)

set /p SYSTEM_CODE=Enter user system code: 
if "%SYSTEM_CODE%"=="" (
    echo ERROR: System code is required.
    goto :wait_exit
)

set "SYSTEM_CODE=%SYSTEM_CODE: =%"
if not "%SYSTEM_CODE:~11,1%"=="" if "%SYSTEM_CODE:~12,1%"=="" (
    rem length is exactly 12, continue
) else (
    echo ERROR: System code must be exactly 12 characters.
    echo Example format: 7A1B2C3D4E5F
    goto :wait_exit
)
if "%SYSTEM_CODE:~11,1%"=="" (
    echo ERROR: System code must be exactly 12 characters.
    echo Example format: 7A1B2C3D4E5F
    goto :wait_exit
)
echo(%SYSTEM_CODE%| findstr /R /I "^[0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F][0-9A-F]$" >nul
if errorlevel 1 (
    echo ERROR: Invalid system code format.
    echo Only 0-9 and A-F are allowed, 12 chars total.
    goto :wait_exit
)

set "SIMPLIAI_PASSKEY=%MASTER_SECRET%"
if not exist "%GENERATOR_SCRIPT%" (
    echo ERROR: Activation generator script not found: %GENERATOR_SCRIPT%
    goto :wait_exit
)
"%PYTHON_EXE%" "%GENERATOR_SCRIPT%" --system-code "%SYSTEM_CODE%" --output "%OUT_FILE%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Failed to generate activation key. (Exit code: %EXIT_CODE%)
    goto :wait_exit
)

set "GENERATED_KEY="
if exist "%OUT_FILE%" (
    set /p GENERATED_KEY=<"%OUT_FILE%"
)

echo.
echo Activation key generated successfully.
echo Saved to: %CD%\%OUT_FILE%
if defined GENERATED_KEY echo Activation key: !GENERATED_KEY!
echo.
choice /c RE /n /m "Press R to generate another key or E to exit: "
if errorlevel 2 exit /b 0
if errorlevel 1 goto :main

:wait_exit
echo.
set /p DUMMY=Press ENTER to close this window... 
exit /b 1
