@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0启动训练.ps1"
if errorlevel 1 (
  echo.
  echo [失败] 请查看上方报错信息。
)
pause
