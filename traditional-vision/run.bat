
@echo off
setlocal

REM Always prefer the MSYS2 UCRT64 runtime DLLs over other MinGW runtimes.
set "MSYS2_BIN=D:\ProgramData\msys64\ucrt64\bin"
set "PATH=%MSYS2_BIN%;%PATH%"

REM Project directory is the directory containing this batch file.
cd /d "%~dp0"

set "SOURCE=%~dp0traditional_vision.cpp"
set "OUTPUT=%~dp0traditional_vision.exe"
set "DATA_PATH=D:\Projects\welding-droplet-vision-data"
set "GXX=%MSYS2_BIN%\g++.exe"

REM Video name from the first command-line argument.
REM Default to "_" if no argument is provided.
set "VIDEO_NAME=%~1"
if "%VIDEO_NAME%"=="" set "VIDEO_NAME=_"

if not exist "%GXX%" (
    echo ERROR: g++.exe not found: "%GXX%"
    pause
    exit /b 1
)

if not exist "%SOURCE%" (
    echo ERROR: traditional_vision.cpp not found in "%~dp0"
    pause
    exit /b 1
)

echo Video: %VIDEO_NAME%
echo.

echo [1/2] Compiling traditional_vision.cpp ...
"%GXX%" -std=c++17 -g -O0 -Wall -Wextra "%SOURCE%" -o "%OUTPUT%" ^
    -I"D:\ProgramData\msys64\ucrt64\include\opencv5" ^
    -L"D:\ProgramData\msys64\ucrt64\lib" ^
    -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_geometry -lopencv_core

if errorlevel 1 (
    echo.
    echo ERROR: Compilation failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Running traditional_vision.exe ...
"%OUTPUT%" "%DATA_PATH%" "%VIDEO_NAME%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo Program exited with code %EXIT_CODE%.
) else (
    echo Program finished.
)
pause
exit /b %EXIT_CODE%

