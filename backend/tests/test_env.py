"""Tests d'environnement (etape 0). Aucun test d'algorithme a ce stade.

Lancer :  pytest backend/tests -v
"""
import importlib
import sys

import pytest

MODULES = ["fitz", "numpy", "pandas", "sklearn", "torch", "sentence_transformers", "faiss"]


def test_python_version():
    """Le projet cible Python 3.11 (wheels Windows verifiees pour toute la stack)."""
    assert sys.version_info[:2] == (3, 11), f"Python {sys.version_info[:2]} au lieu de 3.11"


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    """Chaque dependance critique doit s'importer sans erreur."""
    importlib.import_module(name)
