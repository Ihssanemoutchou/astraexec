"""
DocumentManager — Gestionnaire de documents
=============================================

Charge automatiquement tous les fichiers .txt d'un dossier
et construit une collection unique de chunks.

Support : .txt (UTF-8, Latin-1, CP1252)
"""

from pathlib import Path
from typing import List, Dict

from app.retrieval.smart_seg import SmartSeg


class DocumentManager:
    """
    DocumentManager

    Parcourt un dossier et indexe tous les fichiers .txt.
    """

    def __init__(self, data_folder: str):
        self.data_folder = Path(data_folder)
        self.segmenter = SmartSeg()

    def load_documents(self) -> List[Dict]:
        all_chunks = []

        txt_files = sorted(self.data_folder.glob("*.txt"))

        if not txt_files:
            raise FileNotFoundError(
                f"Aucun fichier .txt trouvé dans {self.data_folder}"
            )

        global_chunk = 0

        for txt in txt_files:
            print(f"Lecture : {txt.name}")
            chunks = self.segmenter.process(str(txt))

            for chunk in chunks:
                chunk["chunk_id"] = global_chunk
                chunk["source"] = txt.name
                global_chunk += 1
                all_chunks.append(chunk)

        print(f"{len(txt_files)} documents indexés.")
        print(f"{len(all_chunks)} chunks générés.")

        return all_chunks

    def available_sources(self) -> List[str]:
        """Retourne la liste des fichiers .txt disponibles."""
        return sorted(
            str(f.name) for f in self.data_folder.glob("*.txt")
        )

    def source_count(self) -> int:
        return len(list(self.data_folder.glob("*.txt")))