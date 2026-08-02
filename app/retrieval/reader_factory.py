"""
ReaderFactory — Fabrique de lecteurs de documents
====================================================

Sélectionne le lecteur adapté à l'extension d'un fichier :

    .pdf  → PDFDocumentReader
    .txt  → TXTDocumentReader

Toute la logique de sélection est centralisée ici
(principe ouvert/fermé) : le reste du projet ne contient
aucun `if extension` éparpillé. Ajouter un format futur
se réduit à enregistrer un lecteur dans `READERS`.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Type

# Permet d'exécuter le module directement (python app/retrieval/reader_factory.py)
# en rendant le paquet « app » importable, comme dans les tests du projet.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.retrieval.pdf_reader import PDFDocumentReader
from app.retrieval.txt_reader import TXTDocumentReader


class ReaderFactory:
    """
    ReaderFactory

    Registre central des lecteurs de documents,
    sélectionnés selon l'extension du fichier.
    """

    READERS: Dict[str, Type] = {
        ".pdf": PDFDocumentReader,
        ".txt": TXTDocumentReader,
    }

    # =====================================================
    # Création d'un lecteur
    # =====================================================

    @classmethod
    def create(cls, file_path: str):
        """
        Retourne une instance du lecteur adapté au fichier.

        Paramètres :
            file_path : chemin vers le document.

        Retour :
            instance de PDFDocumentReader ou TXTDocumentReader.

        Exceptions :
            ValueError : extension non supportée.
        """
        extension = Path(file_path).suffix.lower()
        reader_class: Optional[Type] = cls.READERS.get(extension)

        if reader_class is None:
            supported = ", ".join(sorted(cls.READERS))
            raise ValueError(
                f"Format non supporté : '{extension}'. "
                f"Formats pris en charge : {supported}."
            )

        return reader_class()

    # =====================================================
    # Interrogation du registre
    # =====================================================

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """
        Retourne True si l'extension du fichier est prise en charge.
        """
        return Path(file_path).suffix.lower() in cls.READERS

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """
        Retourne la liste triée des extensions supportées.
        """
        return sorted(cls.READERS)

    # =====================================================
    # Énumération des fichiers supportés d'un dossier
    # =====================================================

    @classmethod
    def supported_files(cls, folder: str) -> List[Path]:
        """
        Retourne, triés alphabétiquement, tous les fichiers du dossier
        dont l'extension est prise en charge.

        Paramètres :
            folder : chemin du dossier à parcourir.

        Retour :
            liste de chemins Path (fichiers supportés uniquement).
        """
        folder_path = Path(folder)
        files: List[Path] = []

        for extension in cls.READERS:
            files.extend(folder_path.glob(f"*{extension}"))

        return sorted(set(files))


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    # Affichage correct des caractères Unicode sous Windows (cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp()

    try:
        pdf_path = os.path.join(tmp_dir, "doc.pdf")
        txt_path = os.path.join(tmp_dir, "note.txt")
        other_path = os.path.join(tmp_dir, "image.png")

        # Création de faux fichiers (contenu non lu ici)
        with open(pdf_path, "w") as f:
            f.write("x")
        with open(txt_path, "w") as f:
            f.write("x")
        with open(other_path, "w") as f:
            f.write("x")

        # Sélection par extension
        pdf_reader = ReaderFactory.create(pdf_path)
        txt_reader = ReaderFactory.create(txt_path)
        print(f".pdf → {type(pdf_reader).__name__}")
        print(f".txt → {type(txt_reader).__name__}")

        # Extension inconnue
        try:
            ReaderFactory.create(other_path)
        except ValueError as exc:
            print(f".png → rejeté : {exc}")

        # Interrogation du registre
        print(f"Extensions supportées : {ReaderFactory.supported_extensions()}")
        print(f"is_supported(doc.pdf) : {ReaderFactory.is_supported(pdf_path)}")

        # Énumération du dossier
        files = ReaderFactory.supported_files(tmp_dir)
        print(f"Fichiers supportés : {[f.name for f in files]}")
    finally:
        shutil.rmtree(tmp_dir)
