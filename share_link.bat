@echo off
chcp 65001 > nul
python share.py
if errorlevel 1 (
    echo.
    echo Chay truc tiep bang cloudflared...
    .\cloudflared.exe tunnel --url http://127.0.0.1:8000
)
pause

