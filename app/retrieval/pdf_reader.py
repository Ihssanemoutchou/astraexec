"""
PDFDocumentReader — Lecture des fichiers PDF
==============================================

Lecture seule d'un PDF avec PyMuPDF (fitz) :

    1. ouverture du fichier
    2. lecture de toutes les pages
    3. concaténation du texte extrait
    4. retour d'une chaîne unique

Responsabilité unique : la LECTURE.

Hors périmètre (traités par d'autres composants) :
    - nettoyage du texte      → TextCleaner / SmartSeg
    - segmentation / chunks   → ChunkGenerator / SmartSeg
    - métadonnées             → MetadataBuilder / SmartSeg

Cas gérés :
    - PDF vide            → PDFReadError explicite
    - PDF protégé         → PDFReadError (mot de passe requis)
    - PDF corrompu        → PDFReadError (message de cause)
    - PDF scanné          → PDFReadError (OCR hors périmètre)
    - PDF multi-pages     → concaténation de toutes les pages
    - caractères Unicode  → gérés nativement par PyMuPDF
"""

from pathlib import Path
from typing import List

# PyMuPDF s'importe via « pymupdf » (>= 1.24) ou « fitz » (ancien alias).
try:
    import pymupdf as fitz
except ImportError:
    import fitz


class PDFReadError(Exception):
    """
    Erreur de lecture d'un PDF.

    Porte un message explicite destiné à l'utilisateur
    (fichier concerné + cause), afin que l'appelant puisse
    décider d'ignorer le fichier sans interrompre le lot.
    """


class PDFDocumentReader:
    """
    PDFDocumentReader

    Lecture seule des fichiers PDF via PyMuPDF (fitz).
    Retourne le texte brut concaténé de toutes les pages.
    """

    EXTENSIONS = (".pdf",)

    # =====================================================
    # Constructeur
    # =====================================================

    def __init__(self, password: str = ""):
        """
        Initialise le lecteur.

        Paramètres :
            password : mot de passe optionnel pour les PDF protégés.
        """
        self.password = password

    # =====================================================
    # Lecture
    # =====================================================

    def read(self, file_path: str) -> str:
        """
        Lit un PDF et retourne le texte concaténé de toutes les pages.

        Paramètres :
            file_path : chemin vers le fichier PDF.

        Retour :
            chaîne de caractères contenant le texte extrait.

        Exceptions :
            FileNotFoundError : fichier inexistant.
            ValueError        : extension non supportée.
            PDFReadError      : PDF vide, protégé, corrompu ou scanné.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document introuvable : {file_path}"
            )

        if path.suffix.lower() not in self.EXTENSIONS:
            raise ValueError(
                f"Format non supporté : {path.suffix}. "
                f"Seuls les fichiers {', '.join(self.EXTENSIONS)} "
                "sont acceptés."
            )

        # ── Ouverture (les fichiers corrompus échouent ici) ────────
        try:
            document = fitz.open(str(path))
        except Exception as exc:
            raise PDFReadError(
                f"PDF illisible ou corrompu : {path.name} ({exc})"
            ) from exc

        try:
            # ── Protection par mot de passe ────────────────────────
            if document.needs_pass:
                if not self.password:
                    raise PDFReadError(
                        f"PDF protégé par mot de passe : {path.name}. "
                        "Un mot de passe est requis."
                    )
                if not document.authenticate(self.password):
                    raise PDFReadError(
                        f"PDF protégé : mot de passe invalide pour "
                        f"{path.name}."
                    )

            # ── PDF vide ───────────────────────────────────────────
            if document.page_count == 0:
                raise PDFReadError(
                    f"PDF vide : {path.name} (aucune page)."
                )

            # ── Extraction de toutes les pages ─────────────────────
            try:
                pages: List[str] = []
                for page in document:
                    pages.append(page.get_text("text"))
                text = "\n".join(pages)
            except Exception as exc:
                # Corruption survenant en cours de lecture
                # (fichier ouvert mais pages illisibles).
                raise PDFReadError(
                    f"PDF illisible ou corrompu : {path.name} ({exc})"
                ) from exc

            # ── PDF scanné (aucun texte numérique extractible) ─────
            if not text.strip():
                raise PDFReadError(
                    f"PDF scanné : aucun texte extractible dans "
                    f"{path.name}. L'OCR est hors périmètre."
                )

            return text

        finally:
            document.close()


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    import tempfile
    import os

    reader = PDFDocumentReader()

    # 1. PDF normal multi-pages
    tmp_pdf = tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False
    )
    tmp_pdf.close()

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Machine learning et intelligence artificielle.")
    page = doc.new_page()
    page.insert_text((72, 72), "La recherche lexicale avec BM25 et TF-IDF.")
    doc.save(tmp_pdf.name)
    doc.close()

    try:
        text = reader.read(tmp_pdf.name)
        print(f"PDF multi-pages lu ({len(text)} caractères) :")
        print(text)
    finally:
        os.unlink(tmp_pdf.name)

    # 2. PDF vide (0 page : document minimal écrit en bytes bruts,
    #    car PyMuPDF refuse de sauvegarder un document sans page)
    tmp_empty = tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False
    )
    tmp_empty.write(
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"trailer\n<< /Root 1 0 R /Size 2 >>\n%%EOF\n"
    )
    tmp_empty.close()

    try:
        try:
            reader.read(tmp_empty.name)
        except PDFReadError as exc:
            print(f"\nPDF vide détecté : {exc}")
    finally:
        os.unlink(tmp_empty.name)

    # 3. Fichier corrompu
    tmp_bad = tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False
    )
    tmp_bad.write(b"ceci n'est pas un pdf")
    tmp_bad.close()

    try:
        try:
            reader.read(tmp_bad.name)
        except PDFReadError as exc:
            print(f"PDF corrompu détecté : {exc}")
    finally:
        os.unlink(tmp_bad.name)
