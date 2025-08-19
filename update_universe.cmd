@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python -m aktien_oop.update_universe --save-dir aktien_oop
pause
