@echo off

cd /D "%~dp0"

set PATH=%PATH%;%SystemRoot%\system32

echo "%CD%"| findstr /C:" " >nul && echo This script relies on Miniconda which can not be silently installed under a path with spaces. && goto end

@rem fix failed install when installing to a separate drive
set TMP=%cd%\installer_files
set TEMP=%cd%\installer_files

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
set "CONDA_ROOT_PREFIX=%CONDA_HOME%"
set "INSTALL_ENV_DIR=%ENV_HOME%"


@rem Read user_env.txt into ENV_FLAG
set ENV_FLAG=""
if exist "%HOME_DIR%\installer_files\user_env.txt" (
    set /p ENV_FLAG=<"%HOME_DIR%\installer_files\user_env.txt"
)

@rem if TGWUI integration flag, run from its env
if "%ENV_FLAG%"=="%ENV_PARENT%" (
    set "CONDA_ROOT_PREFIX=%CONDA_PARENT%"
    set "INSTALL_ENV_DIR=%ENV_PARENT%"
)


@rem environment isolation
set PYTHONNOUSERSITE=1
set PYTHONPATH=
set PYTHONHOME=


@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

@rem update installer env
call python one_click.py --update-wizard-windows --conda-env-path "%INSTALL_ENV_DIR%" && (
    echo.
    echo Have a great day!
)

:end
pause
