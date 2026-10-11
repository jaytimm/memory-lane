@echo off
setlocal
set "USB_ROOT=%~dp0"
set "PACKAGE=%USB_ROOT%MemoryLane-Windows-x64.zip"
set "ARCHIVE=%USB_ROOT%Gallery"
set "PHOTOS=%USB_ROOT%..\Family Photos"
set "INSTALL=%LOCALAPPDATA%\MemoryLane"
set "PACKAGES=%INSTALL%\packages"
set "MEMORY_LANE_PACKAGE=%PACKAGE%"

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

if not exist "%PHOTOS%\" (
  echo The Family Photos folder was not found beside Memory Lane.
  echo.
  echo Expected: %PHOTOS%
  pause
  exit /b 2
)

for /f "usebackq delims=" %%I in (`powershell.exe -NoProfile -Command ^
  "$f=Get-Item -LiteralPath $env:MEMORY_LANE_PACKAGE; $f.Length.ToString() + '-' + $f.LastWriteTimeUtc.Ticks.ToString()"`) do set "PACKAGE_VERSION=%%I"
if not defined PACKAGE_VERSION (
  echo Memory Lane could not inspect the Windows package.
  pause
  exit /b 2
)

set "RELEASE=%PACKAGES%\%PACKAGE_VERSION%"
set "STAGING=%PACKAGES%\%PACKAGE_VERSION%.partial"
set "APP=%RELEASE%\WindowsSearch"
set "READY=%RELEASE%\ready.txt"
set "MEMORY_LANE_PACKAGES=%PACKAGES%"
set "MEMORY_LANE_RELEASE=%RELEASE%"
set "MEMORY_LANE_STAGING=%STAGING%"
set "MEMORY_LANE_APP=%APP%"
set "MEMORY_LANE_READY=%READY%"

if not exist "%APP%\runtime\python.exe" goto prepare
if not exist "%READY%" goto prepare
goto launch

:prepare
  echo Preparing or updating Memory Lane on this computer...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "New-Item -ItemType Directory -Force -Path $env:MEMORY_LANE_PACKAGES | Out-Null;" ^
    "if (Test-Path -LiteralPath $env:MEMORY_LANE_STAGING) { Remove-Item -LiteralPath $env:MEMORY_LANE_STAGING -Recurse -Force };" ^
    "if (Test-Path -LiteralPath $env:MEMORY_LANE_RELEASE) { Remove-Item -LiteralPath $env:MEMORY_LANE_RELEASE -Recurse -Force };" ^
    "Expand-Archive -LiteralPath $env:MEMORY_LANE_PACKAGE -DestinationPath $env:MEMORY_LANE_STAGING;" ^
    "if (-not (Test-Path -LiteralPath ($env:MEMORY_LANE_STAGING + '\WindowsSearch\runtime\python.exe'))) { throw 'The Windows package is incomplete.' };" ^
    "Move-Item -LiteralPath $env:MEMORY_LANE_STAGING -Destination $env:MEMORY_LANE_RELEASE;" ^
    "Set-Content -LiteralPath $env:MEMORY_LANE_READY -Value 'ready' -Encoding ASCII"
  if errorlevel 1 (
    echo Memory Lane could not be prepared on this computer.
    pause
    exit /b 2
  )

:launch
pushd "%APP%"
runtime\python.exe -m src.serve --archive "%ARCHIVE%" --photos "%PHOTOS%"
set "RESULT=%ERRORLEVEL%"
popd

if not "%RESULT%"=="0" pause
exit /b %RESULT%
