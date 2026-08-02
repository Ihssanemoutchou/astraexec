"""
DocumentManager — Gestionnaire de documents
=============================================

Charge automatiquement tous les fichiers supportés d'un dossier
(.txt et .pdf) et construit une collection unique de chunks.

Pipeline :
    ReaderFactory
    ↓
    TXTDocumentReader ou PDFDocumentReader
    ↓
    SmartSeg.process_text()
    ↓
    chunks

Support : .txt (UTF-8, Latin-1, CP1252), .pdf (PyMuPDF)

Résilience : un fichier illisible (PDF corrompu, protégé, scanné, ...)
est ignoré avec un avertissement sans interrompre l'indexation du lot.
"""

from pathlib import Path
from typing import List, Dict

from app.retrieval.smart_seg import SmartSeg
from app.retrieval.reader_factory import ReaderFactory
from app.retrieval.pdf_reader import PDFReadError


class DocumentManager:
    """
    DocumentManager

    Parcourt un dossier et indexe tous les fichiers supportés
    (.txt, .pdf) via la ReaderFactory et SmartSeg.
    """

    def __init__(self, data_folder: str):
        self.data_folder = Path(data_folder)
        self.segmenter = SmartSeg()

    def load_documents(self) -> List[Dict]:
        all_chunks = []

        files = ReaderFactory.supported_files(self.data_folder)

        if not files:
            raise FileNotFoundError(
                f"Aucun fichier ({', '.join(ReaderFactory.supported_extensions())}) "
                f"trouvé dans {self.data_folder}"
            )

        global_chunk = 0
        indexed = 0

        for file_path in files:
            # Lecture via le lecteur adapté à l'extension.
            try:
                reader = ReaderFactory.create(file_path)
                text = reader.read(str(file_path))
            except (PDFReadError, FileNotFoundError, ValueError) as exc:
                # Résilience : le fichier fautif est ignoré, le lot continue.
                print(f"Avertissement : {file_path.name} ignoré ({exc})")
                continue

            print(f"Lecture : {file_path.name}")
            chunks = self.segmenter.process_text(text, str(file_path))

            for chunk in chunks:
                chunk["chunk_id"] = global_chunk
                chunk["source"] = file_path.name
                global_chunk += 1
                all_chunks.append(chunk)

            indexed += 1

        print(f"{indexed} documents indexés.")
        print(f"{len(all_chunks)} chunks générés.")

        return all_chunks

    def available_sources(self) -> List[str]:
        """Retourne la liste des fichiers supportés disponibles."""
        return sorted(
            str(f.name) for f in ReaderFactory.supported_files(self.data_folder)
        )

    def source_count(self) -> int:
        return len(ReaderFactory.supported_files(self.data_folder))