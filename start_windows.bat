@echo off

REM Store the initial directory before any changes
set "HOME_DIR=%cd%"
set "PARENT_DIR=%HOME_DIR%\.."

REM Set Conda paths
set "CONDA_HOME=%HOME_DIR%\installer_files\conda"
set "CONDA_PARENT=%PARENT_DIR%\installer_files\conda"
set "ENV_HOME=%HOME_DIR%\installer_files\env"
set "ENV_PARENT=%PARENT_DIR%\installer_files\env"

REM Function to check if conda environment exists
:check_conda
if exist "%CONDA_HOME%\condabin\conda.bat" (
    set "CONDA_ROOT_PREFIX=%CONDA_HOME%"
    set "INSTALL_ENV_DIR=%ENV_HOME%"
    goto activate_conda
) else if exist "%CONDA_PARENT%\condabin\conda.bat" (
    set "CONDA_ROOT_PREFIX=%CONDA_PARENT%"
    set "INSTALL_ENV_DIR=%ENV_PARENT%"
    goto activate_conda
)

REM No conda environment found, ask the user to install as standalone
set /p USER_CHOICE="No conda environment found. Proceed with standalone installation? (Y/N): "
if /I "%USER_CHOICE:~0,1%"=="Y" (
    call :setup_conda
    goto activate_conda
) else (
    echo Exiting script.
    exit /b
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

goto activate_conda

REM Function to activate conda and run script
:activate_conda
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%"
if %errorlevel% neq 0 (
    echo Failed to activate the conda environment.
    exit /b
)

echo Running one_click.py...
python "%HOME_DIR%\one_click.py"

goto end

:end
pause
