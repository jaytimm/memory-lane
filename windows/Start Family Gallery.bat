@echo off
setlocal
set "ROOT=%~dp0"

if not exist "%ROOT%FamilyPhotoArchive\index.html" (
  echo FamilyPhotoArchive was not found next to this launcher.
  echo.
  echo Expected: %ROOT%FamilyPhotoArchive\index.html
  pause
  exit /b 2
)

pushd "%ROOT%WindowsSearch"
runtime\python.exe -m src.serve --archive "%ROOT%FamilyPhotoArchive"
set "RESULT=%ERRORLEVEL%"
popd

if not "%RESULT%"=="0" pause
exit /b %RESULT%

