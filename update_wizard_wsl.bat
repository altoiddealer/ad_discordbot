@echo off

cd /D "%~dp0"

set PATH=%PATH%;%SystemRoot%\system32

@rem Check if wsl.sh exists before attempting to modify it
if not exist "./wsl.sh" (
    echo Error: wsl.sh not found. Exiting...
    exit /b
)

@rem Convert newlines in wsl.sh to Unix format
call wsl -e bash -lic "sed -i 's/\x0D$//' ./wsl.sh; source ./wsl.sh update-wizard"

:end
pause
