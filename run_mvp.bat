@echo off
setlocal
cd /d "%~dp0"
set "CONDA_BAT=C:\Users\user\miniconda3\condabin\conda.bat"

echo [1/4] Verification des artefacts et de Neo4j Aura...
call "%CONDA_BAT%" run -n pfa-rag python backend\scripts\fast_build_mvp.py
if errorlevel 1 goto :error

echo [2/4] Lancement du backend FastAPI...
start "PFA FastAPI" /min cmd /k "cd /d %~dp0 && call %CONDA_BAT% activate pfa-rag && python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"

echo [3/4] Lancement du frontend React...
start "PFA React" /min cmd /k "cd /d %~dp0frontend && npm.cmd run dev"

echo [4/4] Ouverture de l'application...
timeout /t 5 /nobreak >nul
start "" "http://127.0.0.1:5173"
echo Backend: http://127.0.0.1:8000/docs
echo Frontend: http://127.0.0.1:5173
exit /b 0

:error
echo Echec pendant la verification des artefacts.
exit /b 1

