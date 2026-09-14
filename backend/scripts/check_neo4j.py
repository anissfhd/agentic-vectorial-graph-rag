"""ETAPE 0E - Diagnostic Neo4j, sans creer le moindre noeud.

    python backend/scripts/check_neo4j.py

Determine laquelle des trois situations s'applique :
  A. une instance Neo4j locale ecoute sur bolt://localhost:7687 ;
  B. des identifiants Neo4j Aura sont presents dans backend/.env ;
  C. aucun Neo4j disponible.

Aucun identifiant n'est affiche : seule leur PRESENCE est signalee.
Le driver neo4j n'est pas encore installe en phase 1 : ce script se limite
donc a un test de socket TCP, ce qui suffit pour trancher entre A, B et C.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

BACKEND_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env")
except Exception:
    pass


def port_ouvert(host: str, port: int, timeout: float = 3.0) -> bool:
    """Teste si un service ecoute sur host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> None:
    print("=" * 60)
    print("ETAPE 0E - DIAGNOSTIC NEO4J")
    print("=" * 60)

    env_file = BACKEND_DIR / ".env"
    print(f"\nFichier .env : {'present' if env_file.exists() else 'ABSENT'}")

    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USER", "").strip()
    pwd = os.environ.get("NEO4J_PASSWORD", "").strip()

    # On ne montre jamais la valeur, seulement si elle est renseignee.
    print(f"  NEO4J_URI      : {'renseigne' if uri else 'vide'}")
    print(f"  NEO4J_USER     : {'renseigne' if user else 'vide'}")
    print(f"  NEO4J_PASSWORD : {'renseigne' if pwd else 'vide'}")

    local_bolt = port_ouvert("127.0.0.1", 7687)
    local_http = port_ouvert("127.0.0.1", 7474)
    print(f"\nPort local 7687 (bolt) : {'OUVERT' if local_bolt else 'ferme'}")
    print(f"Port local 7474 (http) : {'OUVERT' if local_http else 'ferme'}")

    aura_ok = False
    if uri and not any(h in uri for h in ("localhost", "127.0.0.1")):
        parsed = urlparse(uri)
        host = parsed.hostname
        port = parsed.port or 7687
        if host:
            aura_ok = port_ouvert(host, port, timeout=8.0)
            print(f"Instance distante joignable ({port}) : {'OUI' if aura_ok else 'NON'}")

    print("\n--- SITUATION ---")
    if local_bolt:
        print("A : instance Neo4j LOCALE detectee et joignable.")
        print("    -> rien a installer, renseigner .env et poursuivre.")
    elif uri and user and pwd and aura_ok:
        print("B : identifiants Aura presents et instance joignable.")
        print("    -> rien a installer, poursuivre.")
    elif uri and user and pwd and not aura_ok:
        print("B (degrade) : identifiants presents mais instance INJOIGNABLE.")
        print("    -> instance Aura probablement en pause : la reveiller depuis")
        print("       console.neo4j.io, puis relancer ce script.")
    else:
        print("C : aucun Neo4j disponible.")
        print("    -> voir la section Neo4j du README pour le chemin le plus rapide.")


if __name__ == "__main__":
    main()
