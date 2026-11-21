@echo off
setlocal
pushd "%~dp0" || exit /b 1

set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "DEEPSEEK_ENABLED=0"
set "BGE_MODEL=off"
set "RAG_ALLOW_MODEL_DOWNLOAD=0"
set "MACBERT_MODEL_DIR=.runtime/demo-macbert"

set "DEMO_PYTHON=runtime\python.exe"
if exist "%DEMO_PYTHON%" goto run_demo

set "DEMO_VENV=.runtime\source-demo"
set "DEMO_PYTHON=%DEMO_VENV%\Scripts\python.exe"
if exist "%DEMO_PYTHON%" if exist "%DEMO_VENV%\.ready" goto run_demo

echo Preparing the source demo. The first run requires an internet connection.
echo For offline use, download the Windows x64 ZIP from GitHub Releases.

set "DEMO_UV=uv.exe"
where.exe uv.exe >nul 2>nul
if not errorlevel 1 goto install_packages

set "DEMO_TOOLS=.runtime\launcher"
set "DEMO_UV=%DEMO_TOOLS%\uv.exe"
set "DEMO_UV_ZIP=%DEMO_TOOLS%\uv.zip"
if exist "%DEMO_UV%" goto install_packages
if not exist "%DEMO_TOOLS%\." (
    mkdir "%DEMO_TOOLS%"
    if errorlevel 1 goto setup_failed
)
curl.exe --fail --location --retry 2 --output "%DEMO_UV_ZIP%" "https://github.com/astral-sh/uv/releases/download/0.11.16/uv-x86_64-pc-windows-msvc.zip"
if errorlevel 1 goto setup_failed
powershell.exe -NoProfile -NonInteractive -Command "$ErrorActionPreference = 'Stop'; if ((Get-FileHash -LiteralPath '.runtime/launcher/uv.zip' -Algorithm SHA256).Hash -ne 'DD9D6D6554BFAB265BFA98AA8E8A406C5C3A7B97582F93DE1F4D48D9154A0395') { throw 'The uv download failed verification.' }"
if errorlevel 1 goto setup_failed
tar.exe -xf "%DEMO_UV_ZIP%" -C "%DEMO_TOOLS%" uv.exe
if errorlevel 1 goto setup_failed

:install_packages
set "UV_SYSTEM_CERTS=true"
if not exist "%DEMO_PYTHON%" (
    "%DEMO_UV%" venv --managed-python --python 3.11 "%DEMO_VENV%"
    if errorlevel 1 goto setup_failed
)
"%DEMO_UV%" pip install --python "%DEMO_PYTHON%" -r requirements.txt
if errorlevel 1 goto setup_failed
type nul > "%DEMO_VENV%\.ready"
if errorlevel 1 goto setup_failed

:run_demo
"%DEMO_PYTHON%" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false %*
set "DEMO_EXIT=%ERRORLEVEL%"
if not "%DEMO_EXIT%"=="0" pause
popd
exit /b %DEMO_EXIT%

:setup_failed
echo.
echo Setup could not finish. Check the internet connection and run again.
echo The portable ZIP on GitHub Releases runs without setup.
pause
popd
exit /b 1
