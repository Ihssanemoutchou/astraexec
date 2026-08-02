"""
Tests pour TXTDocumentReader, ReaderFactory et DocumentManager
(support PDF - Livrable 3).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval.txt_reader import TXTDocumentReader
from app.retrieval.pdf_reader import PDFDocumentReader
from app.retrieval.reader_factory import ReaderFactory
from app.retrieval.smart_seg import SmartSeg
from tests.pdf_fixtures import make_pdf


class TestTXTDocumentReader:
    """Cas nominaux et cas limites du lecteur TXT."""

    def test_read_utf8(self, tmp_path):
        path = str(tmp_path / "note.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("Évaluation de la pertinence avec accents.")

        text = TXTDocumentReader().read(path)
        assert "Évaluation" in text
        assert "pertinence" in text

    def test_read_cp1252(self, tmp_path):
        path = str(tmp_path / "legacy.txt")
        with open(path, "w", encoding="cp1252") as handle:
            handle.write("Fichier encodé en cp1252 : déjà, où, çà.")

        text = TXTDocumentReader().read(path)
        assert "déjà" in text
        assert "où" in text

    def test_read_emoji_utf8(self, tmp_path):
        path = str(tmp_path / "emoji.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("Texte avec emoji 😀 en utf-8.")

        text = TXTDocumentReader().read(path)
        assert "😀" in text

    def test_identity_with_smartseg(self):
        """
        Non-régression Livrable 2 : TXTDocumentReader.read doit produire
        exactement le même texte que SmartSeg.load_text (même boucle
        d'encodages), pour garantir des chunks bit-à-bit identiques.
        """
        reader = TXTDocumentReader()
        segmenter = SmartSeg()
        for name in [
            "astra_platform.txt",
            "machine_learning.txt",
            "recherche_lexicale.txt",
            "sample.txt",
        ]:
            path = os.path.join("app", "api", "data", name)
            assert reader.read(path) == segmenter.load_text(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            TXTDocumentReader().read("fichier_inexistant.txt")

    def test_wrong_extension(self, tmp_path):
        path = str(tmp_path / "image.png")
        with open(path, "wb") as handle:
            handle.write(b"x")

        with pytest.raises(ValueError):
            TXTDocumentReader().read(path)


class TestReaderFactory:
    """Sélection centralisée des lecteurs par extension."""

    def test_create_pdf_reader(self, tmp_path):
        path = str(tmp_path / "doc.pdf")
        with open(path, "wb") as handle:
            handle.write(b"%PDF")
        reader = ReaderFactory.create(path)
        assert isinstance(reader, PDFDocumentReader)

    def test_create_txt_reader(self, tmp_path):
        path = str(tmp_path / "note.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
        reader = ReaderFactory.create(path)
        assert isinstance(reader, TXTDocumentReader)

    def test_uppercase_extension(self, tmp_path):
        path = str(tmp_path / "DOC.PDF")
        with open(path, "wb") as handle:
            handle.write(b"%PDF")
        reader = ReaderFactory.create(path)
        assert isinstance(reader, PDFDocumentReader)

    def test_unknown_extension(self, tmp_path):
        path = str(tmp_path / "image.png")
        with open(path, "wb") as handle:
            handle.write(b"x")

        with pytest.raises(ValueError):
            ReaderFactory.create(path)

    def test_is_supported(self, tmp_path):
        assert ReaderFactory.is_supported(str(tmp_path / "a.pdf")) is True
        assert ReaderFactory.is_supported(str(tmp_path / "a.txt")) is True
        assert ReaderFactory.is_supported(str(tmp_path / "a.png")) is False

    def test_supported_extensions(self):
        extensions = ReaderFactory.supported_extensions()
        assert ".pdf" in extensions
        assert ".txt" in extensions

    def test_supported_files(self, tmp_path):
        for name in ["a.txt", "b.pdf", "c.txt", "d.png"]:
            with open(str(tmp_path / name), "wb") as handle:
                handle.write(b"x")

        files = ReaderFactory.supported_files(str(tmp_path))
        names = sorted(f.name for f in files)
        assert names == ["a.txt", "b.pdf", "c.txt"]


class TestDocumentManagerMixed:
    """Indexation mixte TXT + PDF par DocumentManager."""

    def test_mixed_txt_pdf(self, tmp_path):
        # Fichier TXT
        txt_path = tmp_path / "note.txt"
        txt_path.write_text(
            "La recherche lexicale avec BM25 est efficace. "
            "Elle repose sur le calcul de fréquences de termes. "
            "Ce modèle statistique demeure une référence en recherche d'information.",
            encoding="utf-8",
        )

        # Fichier PDF
        pdf_path = tmp_path / "doc.pdf"
        make_pdf(
            str(pdf_path),
            texts=(
                "Le deep learning utilise des réseaux de neurones. "
                "Ces modèles apprennent des représentations hiérarchiques. "
                "Ils obtiennent d'excellents résultats en vision et en langage.",
            ),
        )

        from app.retrieval.document_manager import DocumentManager

        manager = DocumentManager(str(tmp_path))
        chunks = manager.load_documents()

        sources = {c["source"] for c in chunks}
        assert "note.txt" in sources
        assert "doc.pdf" in sources

        # Numérotation globale continue
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

        # Chaque chunk possède les métadonnées attendues
        for chunk in chunks:
            assert "content" in chunk
            assert "length" in chunk
            assert "word_count" in chunk

    def test_txt_only_non_regression(self, tmp_path):
        """
        Non-régression Livrable 2 : sur un corpus TXT seul, DocumentManager
        doit produire exactement les mêmes chunks que SmartSeg.process
        (même contenu, même longueur, même nombre de mots).
        """
        txt_path = tmp_path / "sample.txt"
        txt_path.write_text(
            "Python est un langage de programmation. "
            "Le machine learning est une branche de l'intelligence artificielle. "
            "La recherche hybride combine les approches lexicales et vectorielles.",
            encoding="utf-8",
        )

        from app.retrieval.document_manager import DocumentManager

        # Ancien chemin : SmartSeg.process directement
        segmenter = SmartSeg()
        expected = segmenter.process(str(txt_path))

        # Nouveau chemin : DocumentManager via ReaderFactory + process_text
        manager = DocumentManager(str(tmp_path))
        actual = manager.load_documents()

        assert len(actual) == len(expected)
        for actual_chunk, expected_chunk in zip(actual, expected):
            assert actual_chunk["content"] == expected_chunk["content"]
            assert actual_chunk["length"] == expected_chunk["length"]
            assert actual_chunk["word_count"] == expected_chunk["word_count"]
            assert actual_chunk["source"] == expected_chunk["source"]

    def test_no_supported_files(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"x")

        from app.retrieval.document_manager import DocumentManager

        manager = DocumentManager(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            manager.load_documents()

    def test_resilience_bad_pdf_skipped(self, tmp_path):
        """Un PDF illisible est ignoré, le TXT est quand même indexé."""
        txt_path = tmp_path / "note.txt"
        txt_path.write_text(
            "Contenu texte valide. Ce document contient plusieurs phrases. "
            "Il doit être segmenté correctement par SmartSeg malgré la "
            "présence d'un PDF illisible dans le même dossier.",
            encoding="utf-8",
        )

        bad_pdf = tmp_path / "broken.pdf"
        bad_pdf.write_bytes(b"pas un pdf")

        from app.retrieval.document_manager import DocumentManager

        manager = DocumentManager(str(tmp_path))
        chunks = manager.load_documents()

        sources = {c["source"] for c in chunks}
        assert "note.txt" in sources
        assert "broken.pdf" not in sources
        assert len(chunks) > 0

    def test_available_sources(self, tmp_path):
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        (tmp_path / "b.pdf").write_bytes(b"%PDF")
        (tmp_path / "c.png").write_bytes(b"x")

        from app.retrieval.document_manager import DocumentManager

        manager = DocumentManager(str(tmp_path))
        sources = manager.available_sources()
        assert sources == ["a.txt", "b.pdf"]
        assert manager.source_count() == 2
