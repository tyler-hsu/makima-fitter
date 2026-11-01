@echo off
setlocal
set "APPDIR=%~dp0"
if exist "%APPDIR%Makima Fitter.exe" (
  start "" "%APPDIR%Makima Fitter.exe" %*
) else (
  start "" pythonw "%APPDIR%source\gui.py" %*
)
exit /b 0
