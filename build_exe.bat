@echo off
setlocal
cd /d "%~dp0source"
pyinstaller --clean --noconfirm gui.spec
copy /y "dist\Makima Fitter.exe" "..\Makima Fitter.exe" >nul
