"""
AstraExec — Génération de la Base Documentaire (Livrable 4)
=============================================================

Orchestration de bout en bout de la base documentaire ChromaDB :

    DocumentManager (app/api/data)
        ↓  chunks
    EmbeddingGenerator (all-MiniLM-L6-v2, 384 dims)
        ↓  embeddings
    ChromaManager.build()  →  storage/chroma/  (collection astra_docs, cosine)
        ↓
    close()  (libère les verrous avant compression)
        ↓
    BaseExporter.export()  →  exports/base_documentaire_v1.zip

Le binôme reçoit base_documentaire_v1.zip, le dézippe, puis :

    chromadb.PersistentClient(path="storage/chroma")

Aucun format propriétaire : l'archive est une copie brute du dossier Chroma.

Usage :
    python demo_database.py
"""

import sys
import time

# ── Modules maison ──────────────────────────────────────────────────
from app.retrieval.document_manager import DocumentManager
from app.storage.embedding_generator import EmbeddingGenerator
from app.storage.chroma_manager import DEFAULT_COLLECTION, ChromaManager
from app.storage.base_export import BaseExporter

# ── Constantes ─────────────────────────────────────────────────────
DATA_FOLDER = "app/api/data"

# ── Couleurs ANSI ──────────────────────────────────────────────────
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
SEP = f"{BOLD}{'=' * 72}{RESET}"


def print_step(label: str):
    print(f"  {BLUE}>{RESET}  {label}")


def print_ok():
    print(f"  {GREEN}OK{RESET}")


def print_info(text: str):
    print(f"    {DIM}{text}{RESET}")


# ════════════════════════════════════════════════════════════════════
#  Pipeline de génération
# ════════════════════════════════════════════════════════════════════

def main():
    """Génère la base documentaire et l'archive pour le binôme."""
    # UTF-8 sur la console Windows (affichage correct des accents).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"""
  {BOLD}{CYAN}+----------------------------------------------------+{RESET}
  {BOLD}{CYAN}|  AstraExec - Base Documentaire (Livrable 4)        |{RESET}
  {BOLD}{CYAN}|  DocumentManager → Embeddings → ChromaDB → ZIP     |{RESET}
  {BOLD}{CYAN}+----------------------------------------------------+{RESET}
""")

    start_total = time.time()

    # ── 1. Segmentation (DocumentManager → SmartSeg) ──────────────
    print_step("Segmentation du corpus (DocumentManager) ...")
    dm = DocumentManager(DATA_FOLDER)
    chunks = dm.load_documents()
    if not chunks:
        # Résilience : tous les fichiers ont pu être illisibles → arrêt clair.
        print(f"  {RED}{BOLD}ERREUR{RESET} : aucun chunk généré depuis {DATA_FOLDER}.")
        sys.exit(1)
    print_ok()
    print_info(f"{len(chunks)} chunks générés")

    # ── 2. Vectorisation (EmbeddingGenerator) ─────────────────────
    # Premier chargement du modèle (~30 s), ensuite en cache.
    print()
    print_step("Vectorisation des chunks (all-MiniLM-L6-v2) ...")
    generator = EmbeddingGenerator()
    print_info(generator.info()["engine"])
    t0 = time.time()
    contents = [c["content"] for c in chunks]
    embeddings = generator.embed_texts(contents)
    print_ok()
    print_info(f"{embeddings.shape[0]} vecteurs × {embeddings.shape[1]} dims "
               f"({time.time() - t0:.1f}s)")

    # ── 3. Indexation ChromaDB (try/finally : close() garanti) ────
    print()
    print_step("Indexation dans ChromaDB (astra_docs, cosine) ...")
    chroma = ChromaManager(collection_name=DEFAULT_COLLECTION)
    try:
        chroma.build(chunks, embeddings)
        print_ok()
        info = chroma.info()
        print_info(f"Base : {info['path']}")
        print_info(f"Collection : {info['collection_name']} ({info['space']})")
        print_info(f"Chunks indexés : {info['count']}")
    finally:
        # Libère les verrous de fichiers AVANT la compression (Windows).
        chroma.close()
        print_info("Client Chroma fermé (verrous libérés)")

    # ── 4. Export (BaseExporter) ──────────────────────────────────
    print()
    print_step("Export de l'archive pour le binôme ...")
    zip_path = BaseExporter.export()
    print_ok()
    print_info(f"Archive : {zip_path}")

    # ── Bilan ─────────────────────────────────────────────────────
    elapsed = time.time() - start_total
    print()
    print(SEP)
    print(f"  {BOLD}{GREEN}Base documentaire générée !{RESET}")
    print(f"  {BOLD}  Temps total : {elapsed:.1f}s{RESET}")
    print(f"  {BOLD}  Binôme : dézipper {zip_path} puis "
          f"PersistentClient(path=\"storage/chroma\"){RESET}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
