@echo off
REM Check if a file is dragged and dropped onto the batch file
if "%~1"=="" (
    echo Please drag and drop a JSON file onto this batch script.
    pause
    exit /b 1
)

REM Change directory to the location of this batch file
cd /d "%~dp0"

REM Get the full path of the dropped file
set input_file=%~1

REM Call the Python script with the input file
python script.py "%input_file%"

pause
