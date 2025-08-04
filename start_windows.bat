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

@rem --- Fix TMP/TEMP and ensure installer_files exists ---
set "TMP=%cd%\installer_files"
set "TEMP=%cd%\installer_files"
if not exist "%TMP%" mkdir "%TMP%" >nul 2>nul

@rem deactivate existing conda envs as needed to avoid conflicts
(call conda deactivate && call conda deactivate && call conda deactivate) 2>nul


@rem root directories
set "HOME_DIR=%cd%"
set "PARENT_DIR=%HOME_DIR%\.."

@rem resolve %PARENT_DIR% to absolute path (strips '..\')
for /f "delims=" %%i in ("%PARENT_DIR%") do set "PARENT_DIR=%%~fi"

@rem configs
set "CONDA_HOME=%HOME_DIR%\installer_files\conda"
set "ENV_HOME=%HOME_DIR%\installer_files\env"
set "CONDA_PARENT=%PARENT_DIR%\installer_files\conda"
set "ENV_PARENT=%PARENT_DIR%\installer_files\env"


@rem Read user_env.txt into ENV_FLAG
set "ENV_FLAG="
if exist "%HOME_DIR%\internal\user_env.txt" (
    set /p ENV_FLAG=<"%HOME_DIR%\internal\user_env.txt"
)

@rem If env flag exists, assign paths and activate
if "%ENV_FLAG%"=="%ENV_HOME%" (
    set "CONDA_ROOT_PREFIX=%CONDA_HOME%"
    set "INSTALL_ENV_DIR=%ENV_HOME%"
    goto activate_conda
) 
if "%ENV_FLAG%"=="%ENV_PARENT%" (
    set "CONDA_ROOT_PREFIX=%CONDA_PARENT%"
    set "INSTALL_ENV_DIR=%ENV_PARENT%"
    goto activate_conda
)


@rem Welcome message for first run
echo Welcome to ad_discordbot
echo.

@rem Check if conda environment exists
:check_conda
if exist %CONDA_PARENT%\condabin\conda.bat (
    echo The bot can be integrated with your existing text-generation-webui environment.
    echo [A] Integrate with TGWUI *Recommended*
    echo [B] Create and use own environment
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


@rem Function to install conda and setup environment
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

echo checking if conda environment was created successfully
echo install env dir is %INSTALL_ENV_DIR%

if not exist "%INSTALL_ENV_DIR%\python.exe" (
    echo Conda environment creation failed.
    exit /b
)

echo Conda environment created!

goto activate_conda

@rem Function to activate conda and run script
:activate_conda
echo.
echo ==== Conda Activation Stage ====
echo CONDA ROOT: "%CONDA_ROOT_PREFIX%"
echo TARGET ENV: "%INSTALL_ENV_DIR%"
echo.

rem --- Determine the best activation script (Miniforge vs Miniconda) ---
set "CONDA_ACTIVATE_BAT=%CONDA_ROOT_PREFIX%\condabin\conda.bat"
if exist "%CONDA_ROOT_PREFIX%\Scripts\activate.bat" (
    set "CONDA_ACTIVATE_BAT=%CONDA_ROOT_PREFIX%\Scripts\activate.bat"
)

if not exist "%CONDA_ACTIVATE_BAT%" (
    echo ERROR: Could not find conda activation script at "%CONDA_ACTIVATE_BAT%"
    echo Make sure Miniconda/Miniforge is installed correctly.
    del "%ENV_FLAG%" 2>nul
    goto end
)

echo Trying to activate Conda from: "%CONDA_ACTIVATE_BAT%"

rem --- Activate the environment ---
call "%CONDA_ACTIVATE_BAT%" "%INSTALL_ENV_DIR%"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to activate the conda environment.
    echo This is often due to TMP/TEMP pointing to a non‑existent folder or missing permissions.
    echo.
    exit /b
)

echo Conda activated successfully.

rem --- Run the Python script with inherited environment ---
call python "%HOME_DIR%\one_click.py" --conda-env-path "%INSTALL_ENV_DIR%" %*


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