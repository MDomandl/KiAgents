@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python -m aktien_oop.main --tickers aktien_oop\sp500_tickers.txt --sector-meta aktien_oop\sp500_meta.csv --save-dir aktien_oop --force
pause
