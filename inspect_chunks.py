"""
AstraExec — Utilitaire d'inspection du corpus indexé
======================================================

Affiche pour chaque chunk indexé par DocumentManager :
    - son identifiant global (chunk_id) ;
    - le document source ;
    - un aperçu de son contenu (~200 caractères).

Cet utilitaire sert de référence pour l'alignement du ground truth
d'évaluation : les chunk_id affichés sont exactement ceux que voit
FusionSearch lors d'une recherche.

Usage :
    python inspect_chunks.py
"""

import sys

from app.retrieval.document_manager import DocumentManager

# Taille de l'aperçu demandé (~200 caractères).
PREVIEW_SIZE = 200
SEP = "-" * 49


def main():
    """Charge le corpus indexé et affiche chaque chunk (ID, source, aperçu)."""
    # Force l'UTF-8 sur la sortie console (affichage correct des accents
    # sous Windows, dont la console par défaut utilise cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # certains environnements ne permettent pas la reconfiguration

    dm = DocumentManager("app/api/data")
    chunks = dm.load_documents()

    print(f"{len(chunks)} chunks indexés depuis app/api/data/")
    print(SEP)

    for chunk in chunks:
        # Aperçu : 200 premiers caractères, retours à la ligne écrasés.
        preview = chunk["content"][:PREVIEW_SIZE].replace("\n", " ")
        print(f"Chunk {chunk['chunk_id']}")
        print(f"Source : {chunk.get('source')}")
        print()
        print(preview)
        print(SEP)


if __name__ == "__main__":
    main()
