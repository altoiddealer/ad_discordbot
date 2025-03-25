@echo off
setlocal enabledelayedexpansion

cd /D "%~dp0"

set PATH=%PATH%;%SystemRoot%\system32

echo "%CD%"| findstr /C:" " >nul && echo This script relies on Miniconda which can not be silently installed under a path with spaces. && goto end

@rem Check for special characters in installation path
set "SPCHARMESSAGE="WARNING: Special characters were detected in the installation path!" "         This can cause the installation to fail!""
echo "%CD%"| findstr /R /C:"[!#\$%&()\*+,;<=>?@\[\]\^`{|}~]" >nul && (
	call :PrintBigMessage %SPCHARMESSAGE%
)
set SPCHARMESSAGE=

@rem fix failed install when installing to a separate drive
set TMP=%cd%\installer_files
set TEMP=%cd%\installer_files

@rem deactivate existing conda envs as needed to avoid conflicts
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul


REM Store the initial directory before any changes
set "HOME_DIR=%cd%"
set "PARENT_DIR=%HOME_DIR%\.."

REM Set Conda paths
set "CONDA_HOME=%HOME_DIR%\installer_files\conda"
set "CONDA_PARENT=%PARENT_DIR%\installer_files\conda"
set "ENV_HOME=%HOME_DIR%\installer_files\env"
set "ENV_PARENT=%PARENT_DIR%\installer_files\env"
set "ENV_FLAG=%HOME_DIR%\installer_files\user_env.txt"

REM Check for existing environment flag
if exist "%ENV_FLAG%" (
    for /f "usebackq delims=" %%i in ("%ENV_FLAG%") do (
    set "CONDA_ROOT_PREFIX=%%i"
    set "CONDA_ROOT_PREFIX=!CONDA_ROOT_PREFIX:~0,-1!"
    )
    echo Running the bot from conda environment: "!CONDA_ROOT_PREFIX!"

    REM Determine the correct INSTALL_ENV_DIR based on CONDA_ROOT_PREFIX
    set "STRIPPED_CONDA_ROOT_PREFIX=%CONDA_ROOT_PREFIX:"=%"
    set "STRIPPED_CONDA_HOME=%CONDA_HOME:"=%"
    set "STRIPPED_CONDA_PARENT=%CONDA_PARENT:"=%"

    if "%STRIPPED_CONDA_ROOT_PREFIX%"=="%STRIPPED_CONDA_HOME%" (
        set "INSTALL_ENV_DIR=%ENV_HOME%"
    ) else if "%STRIPPED_CONDA_ROOT_PREFIX%"=="%STRIPPED_CONDA_PARENT%" (
        set "INSTALL_ENV_DIR=%ENV_PARENT%"
    ) else (
        echo Warning: CONDA_ROOT_PREFIX does not match expected paths. Using default.
        set "INSTALL_ENV_DIR=%CONDA_ROOT_PREFIX%\env"
    )

    goto activate_conda
)

REM Welcome message for first run
echo Welcome to ad_discordbot
echo.

REM Check if conda environment exists
:check_conda
if exist %CONDA_PARENT%\condabin\conda.bat (
    echo The bot can be integrated with your existing text-generation-webui environment.
    echo [A] Integrate with TGWUI *Recommended*
    echo [B] Use own environment
    echo [N] Nothing, exit script
    set /p USER_CHOICE="Enter A, B, or N: "
    set USER_CHOICE=!USER_CHOICE:~0,1!

    if /I "!USER_CHOICE!"=="A" (
        set "CONDA_ROOT_PREFIX=%CONDA_PARENT%"
        set "INSTALL_ENV_DIR=%ENV_PARENT%"
        goto activate_conda
    ) 

    if /I "!USER_CHOICE!"=="B" (
        call :setup_conda
        goto activate_conda
    ) 

    if /I "!USER_CHOICE!"=="N" (
        echo Exiting script.
        exit /b
    )

    echo Invalid input. Please enter A, B, or N.
    goto check_conda
) else (
    echo This bot can be integrated with text-generation-webui, but it was not detected.
    echo Install the bot as standalone? This option can be changed later via the update-wizard script.
    echo.
    echo [Y] Yes, install standalone
    echo [N] No, exit
    set /p USER_CHOICE="Enter Y or N: "
    set USER_CHOICE=!USER_CHOICE:~0,1!

    if /I "!USER_CHOICE!"=="Y" (
        call :setup_conda
        goto activate_conda
    ) 

    if /I "!USER_CHOICE!"=="N" (
        echo Exiting script.
        exit /b
    )

    echo Invalid input. Please enter Y or N.
    goto check_conda
)


REM Function to install conda and setup environment
:setup_conda
set "INSTALL_DIR=%HOME_DIR%\installer_files"
set "CONDA_ROOT_PREFIX=%CONDA_HOME%"
set "INSTALL_ENV_DIR=%ENV_HOME%"
set "MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe"
set "MINICONDA_CHECKSUM=307194e1f12bbeb52b083634e89cc67db4f7980bd542254b43d3309eaf7cb358"

mkdir "%INSTALL_DIR%"
echo Downloading Miniconda...
curl -Lk "%MINICONDA_DOWNLOAD_URL%" -o "%INSTALL_DIR%\miniconda_installer.exe"

for /f %%a in ('CertUtil -hashfile "%INSTALL_DIR%\miniconda_installer.exe" SHA256 ^| find /i /v " " ^| find /i "%MINICONDA_CHECKSUM%"') do (
    set "output=%%a"
)
if not defined output (
    echo Miniconda checksum verification failed.
    del "%INSTALL_DIR%\miniconda_installer.exe"
    exit /b
)

echo Installing Miniconda...
start /wait "" "%INSTALL_DIR%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%

if not exist "%CONDA_ROOT_PREFIX%\condabin\conda.bat" (
    echo Miniconda installation failed.
    exit /b
)

echo Creating conda environment...
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.11
if not exist "%INSTALL_ENV_DIR%\python.exe" (
    echo Conda environment creation failed.
    exit /b
)

echo %CONDA_ROOT_PREFIX% > "%ENV_FLAG%"

goto activate_conda

REM Function to activate conda and run script
:activate_conda
echo Trying to activate Conda from: "%CONDA_ROOT_PREFIX%\condabin\conda.bat"
if not exist "%CONDA_ROOT_PREFIX%\condabin\conda.bat" (
    echo Conda activation script not found! Please check your environment and try running the script again.
    del "%ENV_FLAG%"
    goto end
)

call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
if %errorlevel% neq 0 (
    echo Failed to activate the conda environment. Exiting...
    exit /b
)

echo Conda activated successfully.
echo %CONDA_ROOT_PREFIX% > "%ENV_FLAG%"
call python "%HOME_DIR%\one_click.py" --conda-env-path "%INSTALL_ENV_DIR%" --update-wizard-windows %*

@rem below are functions for the script   next line skips these during normal execution
goto end

:PrintBigMessage
echo. && echo.
echo *******************************************************************
for %%M in (%*) do echo * %%~M
echo *******************************************************************
echo. && echo.
exit /b

:end
pause