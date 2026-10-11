@echo off
setlocal
set "USB_ROOT=%~dp0"
set "PACKAGE=%USB_ROOT%MemoryLane-Windows-x64.zip"
set "ARCHIVE=%USB_ROOT%Gallery"
set "INSTALL=%LOCALAPPDATA%\MemoryLane"
set "APP=%INSTALL%\WindowsSearch"
set "VERSION_FILE=%INSTALL%\package-version.txt"

if not exist "%ARCHIVE%\index.html" (
  echo The prepared Memory Lane Gallery was not found.
  echo.
  echo Expected: %ARCHIVE%\index.html
  pause
  exit /b 2
)

if not exist "%PACKAGE%" (
  echo MemoryLane-Windows-x64.zip was not found on this USB drive.
  pause
  exit /b 2
)

for /f "usebackq delims=" %%I in (`powershell.exe -NoProfile -Command ^
  "$f=Get-Item -LiteralPath '%PACKAGE%'; $f.Length.ToString() + ':' + $f.LastWriteTimeUtc.Ticks.ToString()"`) do set "PACKAGE_VERSION=%%I"
if not defined PACKAGE_VERSION (
  echo Memory Lane could not inspect the Windows package.
  pause
  exit /b 2
)

set "INSTALLED_VERSION="
if exist "%VERSION_FILE%" set /p INSTALLED_VERSION=<"%VERSION_FILE%"

if not exist "%APP%\runtime\python.exe" goto prepare
if not "%INSTALLED_VERSION%"=="%PACKAGE_VERSION%" goto prepare
goto launch

:prepare
  echo Preparing or updating Memory Lane on this computer...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -LiteralPath '%PACKAGE%' -DestinationPath '%INSTALL%' -Force"
  if errorlevel 1 (
    echo Memory Lane could not be prepared on this computer.
    pause
    exit /b 2
  )
  >"%VERSION_FILE%" echo %PACKAGE_VERSION%

:launch
pushd "%APP%"
runtime\python.exe -m src.serve --archive "%ARCHIVE%"
set "RESULT=%ERRORLEVEL%"
popd

if not "%RESULT%"=="0" pause
exit /b %RESULT%
