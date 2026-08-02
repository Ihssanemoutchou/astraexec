"""
Tests pour PDFDocumentReader (support PDF - Livrable 3).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval.pdf_reader import PDFDocumentReader, PDFReadError
from tests.pdf_fixtures import (
    make_pdf,
    make_empty_pdf,
    make_corrupted_pdf,
    make_scanned_pdf,
    make_big_pdf,
    find_emoji_font,
)


class TestPDFDocumentReader:
    """Cas nominaux et cas limites du lecteur PDF."""

    def test_read_normal_pdf(self, tmp_path):
        path = str(tmp_path / "normal.pdf")
        make_pdf(path, texts=("Machine learning et intelligence artificielle.",))

        text = PDFDocumentReader().read(path)
        assert "Machine learning" in text
        assert "intelligence artificielle" in text

    def test_read_multi_page_pdf(self, tmp_path):
        path = str(tmp_path / "multi.pdf")
        make_pdf(
            path,
            texts=(
                "Première page : la recherche lexicale.",
                "Deuxième page : BM25 et TF-IDF.",
                "Troisième page : la recherche vectorielle.",
            ),
        )

        text = PDFDocumentReader().read(path)
        assert "Première page" in text
        assert "Deuxième page" in text
        assert "Troisième page" in text

    def test_read_empty_pdf(self, tmp_path):
        path = str(tmp_path / "empty.pdf")
        make_empty_pdf(path)

        with pytest.raises(PDFReadError) as excinfo:
            PDFDocumentReader().read(path)
        assert "vide" in str(excinfo.value).lower()

    def test_read_protected_pdf(self, tmp_path):
        path = str(tmp_path / "protected.pdf")
        make_pdf(path, texts=("Contenu protégé.",), password="secret")

        with pytest.raises(PDFReadError) as excinfo:
            PDFDocumentReader().read(path)
        assert "mot de passe" in str(excinfo.value).lower()

    def test_read_protected_pdf_with_password(self, tmp_path):
        path = str(tmp_path / "protected.pdf")
        make_pdf(path, texts=("Contenu protégé débloqué.",), password="secret")

        text = PDFDocumentReader(password="secret").read(path)
        assert "Contenu protégé débloqué" in text

    def test_read_protected_pdf_wrong_password(self, tmp_path):
        path = str(tmp_path / "protected.pdf")
        make_pdf(path, texts=("Contenu protégé.",), password="secret")

        with pytest.raises(PDFReadError) as excinfo:
            PDFDocumentReader(password="mauvais").read(path)
        assert "mot de passe" in str(excinfo.value).lower()

    def test_read_corrupted_pdf(self, tmp_path):
        path = str(tmp_path / "corrupted.pdf")
        make_corrupted_pdf(path)

        with pytest.raises(PDFReadError) as excinfo:
            PDFDocumentReader().read(path)
        assert "corrompu" in str(excinfo.value).lower()

    def test_read_scanned_pdf(self, tmp_path):
        path = str(tmp_path / "scanned.pdf")
        make_scanned_pdf(path)

        with pytest.raises(PDFReadError) as excinfo:
            PDFDocumentReader().read(path)
        assert "ocr" in str(excinfo.value).lower()

    def test_read_unicode_accents(self, tmp_path):
        path = str(tmp_path / "accents.pdf")
        make_pdf(
            path,
            texts=("Évaluation de la pertinence : naïve, précisément, déjà, où, çà.",),
        )

        text = PDFDocumentReader().read(path)
        assert "Évaluation" in text
        assert "naïve" in text
        assert "précisément" in text

    def test_read_emoji_if_font_available(self, tmp_path):
        font = find_emoji_font()
        if font is None:
            pytest.skip("Aucune police emoji disponible sur cette machine")

        # Même repli d'import que pdf_reader.py (compatibilité anciennes
        # versions de PyMuPDF qui n'exposent que le nom `fitz`).
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        path = str(tmp_path / "emoji.pdf")
        document = fitz.open()
        page = document.new_page()
        page.insert_text(
            (72, 72), "Texte avec emoji 😀", fontsize=12,
            fontname="emoji", fontfile=font,
        )
        document.save(path)
        document.close()

        # La lecture ne doit pas échouer et doit retourner du texte.
        text = PDFDocumentReader().read(path)
        assert len(text.strip()) > 0

    def test_read_big_pdf(self, tmp_path):
        path = str(tmp_path / "big.pdf")
        make_big_pdf(path, page_count=50)

        text = PDFDocumentReader().read(path)
        assert len(text) > 1000
        assert text.count("Page de contenu") >= 50

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PDFDocumentReader().read("fichier_inexistant.pdf")

    def test_wrong_extension(self, tmp_path):
        path = str(tmp_path / "note.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("pas un pdf")

        with pytest.raises(ValueError):
            PDFDocumentReader().read(path)
