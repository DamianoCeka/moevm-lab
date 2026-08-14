@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_olmoe_demo.ps1" %*
exit /b %ERRORLEVEL%
