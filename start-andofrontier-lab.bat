@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_EXE="

if exist "%SCRIPT_DIR%AndoFrontier Lab.exe" (
  set "APP_EXE=%SCRIPT_DIR%AndoFrontier Lab.exe"
) else if exist "%SCRIPT_DIR%release_build\AndoFrontier-Lab-v0.3-Windows\AndoFrontier Lab.exe" (
  set "APP_EXE=%SCRIPT_DIR%release_build\AndoFrontier-Lab-v0.3-Windows\AndoFrontier Lab.exe"
)

if "%APP_EXE%"=="" (
  echo AndoFrontier Lab executable was not found.
  echo.
  echo Expected one of:
  echo   %SCRIPT_DIR%AndoFrontier Lab.exe
  echo   %SCRIPT_DIR%release_build\AndoFrontier-Lab-v0.3-Windows\AndoFrontier Lab.exe
  echo.
  echo Build the Windows release first or unzip the portable package.
  pause
  exit /b 1
)

echo Starting AndoFrontier Lab...
echo %APP_EXE%
start "" "%APP_EXE%"

if errorlevel 1 (
  echo.
  echo Failed to start AndoFrontier Lab.
  pause
  exit /b 1
)

exit /b 0
