"""Rend le paquet `app` importable depuis les tests, sans installation.

Les tests sont lances depuis la racine du projet (`pytest backend/tests -v`) ;
le dossier `backend/` doit donc etre ajoute au chemin d'import.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
