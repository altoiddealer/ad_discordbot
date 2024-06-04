@echo off

Go up one directory level
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

REM Set the path to WSL executable
set WSL_EXEC=wsl

echo "%WORKING_DIR%" | findstr /C:" " >nul
if %errorlevel% neq 0 (
    echo This script relies on Miniconda which cannot be silently installed under a path with spaces.
    pause
    goto end
)

REM Deactivate existing conda envs as needed to avoid conflicts
call %WSL_EXEC% -e bash -lic "conda deactivate && conda deactivate && conda deactivate" 2>nul

REM Config
set CONDA_ROOT_PREFIX=%WORKING_DIR%/installer_files/conda
set INSTALL_ENV_DIR=%WORKING_DIR%/installer_files/env

REM Environment isolation
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=
set "CUDA_PATH=%INSTALL_ENV_DIR%"
set "CUDA_HOME=%CUDA_PATH%"

REM Check if conda.sh exists
if not exist "%CONDA_ROOT_PREFIX%/etc/profile.d/conda.sh" (
    echo Conda.sh not found at %CONDA_ROOT_PREFIX%/etc/profile.d/conda.sh
    pause
    goto end
)

REM Activate installer env
call %WSL_EXEC% -e bash -lic "source %CONDA_ROOT_PREFIX%/etc/profile.d/conda.sh && conda activate %INSTALL_ENV_DIR%"
if %errorlevel% neq 0 (
    echo Failed to activate the conda environment.
    pause
    goto end
)

REM Check if bot.py is in the root directory
if exist "bot.py" (
    echo bot.py found in the root directory.
    echo bot.py is now expected to be in the ad_discordbot directory.
    echo Please move bot.py to the ad_discordbot directory and try again.
    pause
    goto end
)

REM Read command flags from CMD_FLAGS.txt
set "CMD_FLAGS="
if not exist "ad_discordbot/CMD_FLAGS.txt" (
    echo CMD_FLAGS.txt is not found.
) else (
    rem Read each line from CMD_FLAGS.txt, skipping comments
    for /f "usebackq delims=" %%i in (`sed '/^\s*#/d' ad_discordbot/CMD_FLAGS.txt`) do set "CMD_FLAGS=%%i" & goto flags_found
    echo CMD_FLAGS.txt is empty.
)

:flags_found

REM Launch ad_discordbot with flags from CMD_FLAGS.txt
python ad_discordbot/bot.py %CMD_FLAGS%
if %errorlevel% neq 0 (
    echo bot.py execution failed
    pause
    goto end
)

:end
pause
