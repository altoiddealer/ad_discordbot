@echo off

REM Step 1: Perform git pull
git pull
if %errorlevel% neq 0 (
    echo Failed to pull from the repository.
    pause
    exit /b %errorlevel%
)

REM Step 2: Go up one directory level
cd ..
if %errorlevel% neq 0 (
    echo Failed to navigate to the parent directory.
    pause
    exit /b %errorlevel%
)

REM Set the working directory explicitly to the parent directory
set WORKING_DIR=%cd%

REM Change directory to where the script should execute
cd /D "%WORKING_DIR%"
if %errorlevel% neq 0 (
    echo Failed to change directory to %WORKING_DIR%.
    pause
    exit /b %errorlevel%
)

set PATH=%PATH%;%SystemRoot%\system32

echo "%WORKING_DIR%" | findstr /C:" " >nul
if %errorlevel% neq 0 (
    echo This script relies on Miniconda which can not be silently installed under a path with spaces.
    pause
    goto end
)

REM Fix failed install when installing to a separate drive
set TMP=%WORKING_DIR%\installer_files
set TEMP=%WORKING_DIR%\installer_files

REM Deactivate existing conda envs as needed to avoid conflicts
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul

REM Config
set CONDA_ROOT_PREFIX=%WORKING_DIR%\installer_files\conda
set INSTALL_ENV_DIR=%WORKING_DIR%\installer_files\env

REM Environment isolation
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=
set "CUDA_PATH=%INSTALL_ENV_DIR%"
set "CUDA_HOME=%CUDA_PATH%"

REM Check if conda.bat exists
if not exist "%CONDA_ROOT_PREFIX%\condabin\conda.bat" (
    echo Conda.bat not found at %CONDA_ROOT_PREFIX%\condabin\conda.bat
    pause
    goto end
)

REM Activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
if %errorlevel% neq 0 (
    echo Failed to activate the conda environment.
    pause
    goto end
)

REM Step 4: Install dependencies
pip install -r ad_discordbot\requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    goto end
)

echo ad_discordbot has been updated.

REM Enter commands
cmd /k "%*"

:end
pause
