@echo off
REM ====================================================================
REM  ETAPE 0B - Creation de l'environnement Python dedie (Windows)
REM  L'environnement Python global n'est JAMAIS modifie.
REM
REM  Lancer depuis un "Anaconda Prompt", a la racine du projet :
REM      backend\scripts\setup_env.bat
REM ====================================================================

echo.
echo [1/5] Creation de l'environnement conda pfa-rag (Python 3.11)...
REM Canal conda-forge uniquement : communautaire, sans CGU a accepter
REM (les canaux repo.anaconda.com exigent un "conda tos accept" prealable).
call conda create -y -n pfa-rag python=3.11 -c conda-forge --override-channels
if errorlevel 1 goto :erreur

echo.
echo [2/5] Activation...
call conda activate pfa-rag
if errorlevel 1 goto :erreur

echo.
echo [3/5] Mise a jour de pip...
python -m pip install --upgrade pip

echo.
echo [4/5] Installation des dependances de la phase 1...
echo       (environ 1 Go de telechargement, torch inclus - patienter)
pip install -r backend\requirements.txt
if errorlevel 1 goto :erreur

echo.
echo [5/5] Gel des versions installees...
pip freeze > backend\requirements.lock.txt

echo.
echo ====================================================================
echo  Environnement pret. Etape suivante :
echo      python backend\scripts\00_env_check.py
echo ====================================================================
goto :fin

:erreur
echo.
echo  ECHEC. Verifier que conda est bien dans le PATH
echo  (utiliser "Anaconda Prompt" et non l'invite de commandes standard).
exit /b 1

:fin
