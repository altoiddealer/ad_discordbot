@echo off

REM Store the initial directory before any changes
set LAUNCH_DIR=%cd%

REM Go up one directory level
cd ..
if %errorlevel% neq 0 (
    echo Failed to navigate to the parent directory.
    goto run_standalone
)

REM Set the working directory explicitly to the parent directory
set WORKING_DIR=%cd%

REM Change directory to where the script should execute
cd /D "%WORKING_DIR%"
if %errorlevel% neq 0 (
    echo Failed to change directory to %WORKING_DIR%.
    goto run_standalone
)

set PATH=%PATH%;%SystemRoot%\system32

echo "%WORKING_DIR%" | findstr /C:" " >nul
if %errorlevel% neq 0 (
    echo This script relies on Miniconda which can not be silently installed under a path with spaces.
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
    echo text-generation-webui environment not detected. Proceeding as standalone...
    goto run_standalone
)

REM Activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
if %errorlevel% neq 0 (
    echo Failed to activate the conda environment.
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
if not exist "ad_discordbot\CMD_FLAGS.txt" (
    echo CMD_FLAGS.txt is not found.
) else (
    for /f "usebackq delims=" %%i in ("ad_discordbot\CMD_FLAGS.txt") do (
        rem Check if the line is not empty and doesn't start with #
        echo %%i | findstr /r "^#" > nul
        if errorlevel 1 (
            set "CMD_FLAGS=%%i"
            goto flags_found
        )
    )
    echo CMD_FLAGS.txt contains only comments or is empty.
)

:flags_found

REM Launch ad_discordbot with flags from CMD_FLAGS.txt
python ad_discordbot\bot.py %CMD_FLAGS%
if %errorlevel% neq 0 (
    echo bot.py execution failed
    pause
    goto end
)

goto end


:run_standalone

cd /D "%LAUNCH_DIR%"

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

@rem config
set INSTALL_DIR=%cd%\installer_files
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\installer_files\env
set MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Windows-x86_64.exe
set MINICONDA_CHECKSUM=307194e1f12bbeb52b083634e89cc67db4f7980bd542254b43d3309eaf7cb358
set conda_exists=F

@rem figure out whether git and conda needs to be installed
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" set conda_exists=T

@rem (if necessary) install git and conda into a contained environment
@rem download conda
if "%conda_exists%" == "F" (
    REM Prompt the user to install as standalone
    echo Install the bot without TGWUI integration?
    echo This action can be changed at any time via the update-wizard script for your OS.
    set /p USER_CHOICE="Continue with standalone installation? (Y/N): "

    if /I "%USER_CHOICE:~0,1%"=="Y" (
        REM continue to install as standalone
    ) else (
        echo Exiting script.
        exit /b
    )


	echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL% to %INSTALL_DIR%\miniconda_installer.exe

	mkdir "%INSTALL_DIR%"
	call curl -Lk "%MINICONDA_DOWNLOAD_URL%" > "%INSTALL_DIR%\miniconda_installer.exe" || ( echo. && echo Miniconda failed to download. && goto end )

	for /f %%a in ('CertUtil -hashfile "%INSTALL_DIR%\miniconda_installer.exe" SHA256 ^| find /i /v " " ^| find /i "%MINICONDA_CHECKSUM%"') do (
		set "output=%%a"
	)

	if not defined output (
		echo The checksum verification for miniconda_installer.exe has failed.
		del "%INSTALL_DIR%\miniconda_installer.exe"
		goto end
	) else (
		echo The checksum verification for miniconda_installer.exe has passed successfully.
	)

	echo Installing Miniconda to %CONDA_ROOT_PREFIX%
	start /wait "" "%INSTALL_DIR%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%

	@rem test the conda binary
	echo Miniconda version:
	call "%CONDA_ROOT_PREFIX%\_conda.exe" --version || ( echo. && echo Miniconda not found. && goto end )

	@rem delete the Miniconda installer
	del "%INSTALL_DIR%\miniconda_installer.exe"
)

@rem create the installer env
if not exist "%INSTALL_ENV_DIR%" (
	echo Packages to install: %PACKAGES_TO_INSTALL%
	call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.11 || ( echo. && echo Conda environment creation failed. && goto end )
)

@rem check if conda environment was actually created
if not exist "%INSTALL_ENV_DIR%\python.exe" ( echo. && echo Conda environment is empty. && goto end )

@rem environment isolation
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=
set "CUDA_PATH=%INSTALL_ENV_DIR%"
set "CUDA_HOME=%CUDA_PATH%"

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

@rem setup installer env
call python one_click.py %*

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
