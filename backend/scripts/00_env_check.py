"""ETAPE 0B + 0D - Controle de l'environnement Python et test d'embedding cross-lingue.

A lancer APRES creation de l'environnement conda et installation des dependances :

    conda activate pfa-rag
    python backend/scripts/00_env_check.py

Ce script :
  1. verifie la version de Python ;
  2. importe reellement les 7 modules critiques et affiche leurs versions ;
  3. telecharge (une seule fois) le modele multilingue leger du prototype ;
  4. encode 3 phrases de controle et calcule leurs similarites cosinus ;
  5. verifie que le retrieval cross-lingue FR -> EN fonctionne ;
  6. ecrit le resultat dans backend/data/metrics/environment_embedding_test.json.

Aucune valeur n'est codee en dur : tous les chiffres du JSON sont calcules ici.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Chemins du projet ------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
METRICS_DIR = BACKEND_DIR / "data" / "metrics"
OUT_FILE = METRICS_DIR / "environment_embedding_test.json"

# --- Cache Hugging Face : dossier stable, lu depuis .env si present ---------
# Doit etre positionne AVANT tout import de sentence_transformers / transformers.
try:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env")
except Exception:  # python-dotenv absent : on continue avec l'environnement courant
    pass

if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str((BACKEND_DIR / "data" / "hf_cache").resolve())
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42

SENTENCES = [
    "Les vagues de froid extremes en Europe",          # FR - sujet de la these
    "Extreme cold spells in Europe",                    # EN - meme sens
    "Comment fonctionne une voiture electrique ?",      # FR - hors sujet
]
LABELS = ["fr_sujet", "en_sujet", "fr_hors_sujet"]

REQUIRED = [
    ("fitz", "pymupdf"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("sentence_transformers", "sentence-transformers"),
    ("faiss", "faiss-cpu"),
]


def check_imports() -> dict:
    """Importe chaque module critique et retourne son statut et sa version."""
    import importlib

    results = {}
    print("\n--- Verification des imports ---")
    for module_name, pip_name in REQUIRED:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", None) or getattr(mod, "VersionBind", "n/a")
            results[module_name] = {"ok": True, "version": str(version), "pip": pip_name}
            print(f"  [OK]    {module_name:24s} {version}")
        except Exception as exc:
            results[module_name] = {"ok": False, "erreur": str(exc), "pip": pip_name}
            print(f"  [ECHEC] {module_name:24s} -> pip install {pip_name}")
            print(f"          {exc}")
    return results


def run_embedding_test() -> dict:
    """Charge le modele multilingue et mesure les similarites cosinus reelles."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    # Reproductibilite (contrainte du cahier des charges : seed fixe partout)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"\n--- Chargement du modele : {MODEL_NAME} ---")
    print(f"    cache HF : {os.environ['HF_HOME']}")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    load_s = time.perf_counter() - t0
    print(f"    charge en {load_s:.1f} s")

    t0 = time.perf_counter()
    emb = model.encode(SENTENCES, normalize_embeddings=True, convert_to_numpy=True)
    encode_s = time.perf_counter() - t0

    # Vecteurs normalises : le produit scalaire EST la similarite cosinus.
    sim = emb @ emb.T

    fr_en = float(sim[0, 1])   # FR sujet  <-> EN sujet   : doit etre eleve
    fr_hs = float(sim[0, 2])   # FR sujet  <-> hors sujet : doit etre bas
    en_hs = float(sim[1, 2])   # EN sujet  <-> hors sujet : doit etre bas

    print("\n--- Similarites cosinus mesurees ---")
    print(f"  FR sujet  <-> EN sujet       : {fr_en:.4f}")
    print(f"  FR sujet  <-> FR hors sujet  : {fr_hs:.4f}")
    print(f"  EN sujet  <-> FR hors sujet  : {en_hs:.4f}")

    # Critere de validation : le cross-lingue doit primer sur le hors-sujet.
    passed = (fr_en > fr_hs) and (fr_en > en_hs) and (fr_en > 0.5)
    marge = fr_en - max(fr_hs, en_hs)
    print(f"\n  Marge cross-lingue : {marge:.4f}")
    print(f"  VERDICT : {'REUSSI' if passed else 'ECHEC'}")
    if not passed:
        print("  -> Le retrieval FR->EN ne fonctionne pas : ne pas poursuivre l'ingestion.")

    return {
        "modele": MODEL_NAME,
        "dimension": int(emb.shape[1]),
        "device": "cpu",
        "temps_chargement_s": round(load_s, 3),
        "temps_encodage_s": round(encode_s, 4),
        "phrases": {lab: txt for lab, txt in zip(LABELS, SENTENCES)},
        "matrice_similarite": [[round(float(x), 4) for x in row] for row in sim],
        "similarites": {
            "fr_sujet__en_sujet": round(fr_en, 4),
            "fr_sujet__fr_hors_sujet": round(fr_hs, 4),
            "en_sujet__fr_hors_sujet": round(en_hs, 4),
        },
        "marge_cross_lingue": round(marge, 4),
        "critere": "sim(FR,EN) > sim(FR,hors-sujet) ET sim(FR,EN) > sim(EN,hors-sujet) ET sim(FR,EN) > 0.5",
        "reussi": bool(passed),
    }


def main() -> int:
    print("=" * 68)
    print("ETAPE 0 - CONTROLE DE L'ENVIRONNEMENT PYTHON")
    print("=" * 68)
    print(f"Python     : {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"Plateforme : {platform.platform()}")
    print(f"Executable : {sys.executable}")

    if sys.version_info[:2] != (3, 11):
        print(f"\n  ATTENTION : Python {sys.version_info[0]}.{sys.version_info[1]} "
              f"au lieu de 3.11. L'environnement conda 'pfa-rag' est-il active ?")

    imports = check_imports()
    all_ok = all(v["ok"] for v in imports.values())

    payload = {
        "genere_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etape": "0 - audit environnement",
        "python": sys.version.split()[0],
        "plateforme": platform.platform(),
        "seed": SEED,
        "imports": imports,
        "tous_imports_ok": all_ok,
    }

    if not all_ok:
        print("\nDes imports ont echoue : test d'embedding non execute.")
        payload["test_embedding"] = None
    else:
        try:
            payload["test_embedding"] = run_embedding_test()
        except Exception as exc:  # reseau, quota HF, disque plein...
            print(f"\n[ECHEC] Test d'embedding impossible : {exc}")
            payload["test_embedding"] = {"reussi": False, "erreur": str(exc)}

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultat ecrit dans : {OUT_FILE}")

    test = payload.get("test_embedding") or {}
    return 0 if (all_ok and test.get("reussi")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
