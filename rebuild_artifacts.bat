@echo off
setlocal
cd /d "%~dp0"
echo Recalcul complet des artefacts hors ligne. Cette operation peut prendre plusieurs minutes.
call "C:\Users\user\miniconda3\condabin\conda.bat" run -n pfa-rag python backend\scripts\fast_build_mvp.py --force
exit /b %errorlevel%

