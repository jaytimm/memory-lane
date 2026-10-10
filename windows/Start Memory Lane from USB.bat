@echo off
setlocal
set "USB_ROOT=%~dp0"
set "PACKAGE=%USB_ROOT%MemoryLane-Windows-x64.zip"
set "ARCHIVE=%USB_ROOT%FamilyPhotoArchive"
set "INSTALL=%LOCALAPPDATA%\MemoryLane"
set "APP=%INSTALL%\WindowsSearch"

if not exist "%ARCHIVE%\index.html" (
  echo FamilyPhotoArchive was not found on this USB drive.
  pause
  exit /b 2
)

if not exist "%PACKAGE%" (
  echo MemoryLane-Windows-x64.zip was not found on this USB drive.
  pause
  exit /b 2
)

if not exist "%APP%\runtime\python.exe" (
  echo Preparing Memory Lane on this computer. This happens only once...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "Expand-Archive -LiteralPath '%PACKAGE%' -DestinationPath '%INSTALL%' -Force"
  if errorlevel 1 (
    echo Memory Lane could not be prepared on this computer.
    pause
    exit /b 2
  )
)

pushd "%APP%"
runtime\python.exe -m src.serve --archive "%ARCHIVE%"
set "RESULT=%ERRORLEVEL%"
popd

if not "%RESULT%"=="0" pause
exit /b %RESULT%
