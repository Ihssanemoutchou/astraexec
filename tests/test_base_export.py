"""
Tests pour BaseExporter (archive brute de la base Chroma).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from zipfile import ZipFile

import pytest

from app.storage.base_export import BaseExporter


def _fake_chroma_dir(root: Path, name: str = "chroma") -> Path:
    """
    Crée un dossier Chroma factice reproduisant la structure persistante
    réelle : chroma.sqlite3 + segments HNSW dans un sous-dossier uuid.
    """
    chroma_dir = root / "storage" / name
    (chroma_dir / "5f2a1c9e").mkdir(parents=True, exist_ok=True)
    (chroma_dir / "chroma.sqlite3").write_bytes(b"fake sqlite header")
    (chroma_dir / "5f2a1c9e" / "data_level0.bin").write_bytes(b"hnsw segment")
    return chroma_dir


@pytest.fixture()
def fake_chroma(tmp_path):
    return _fake_chroma_dir(tmp_path)


class TestBaseExporterExport:
    def test_export_cree_l_archive(self, fake_chroma, tmp_path):
        zip_path = BaseExporter.export(
            str(fake_chroma), str(tmp_path / "exports" / "base_documentaire_v1.zip")
        )
        assert Path(zip_path).is_file()
        assert zip_path.endswith("base_documentaire_v1.zip")

    def test_export_contient_storage_chroma(self, fake_chroma, tmp_path):
        zip_path = BaseExporter.export(
            str(fake_chroma), str(tmp_path / "exports" / "base_documentaire_v1.zip")
        )
        with ZipFile(zip_path) as archive:
            names = archive.namelist()

        # Le binôme doit retrouver le dossier storage/chroma tel quel.
        assert "storage/chroma/chroma.sqlite3" in names
        assert "storage/chroma/5f2a1c9e/data_level0.bin" in names

    def test_export_aucun_format_proprietaire(self, fake_chroma, tmp_path):
        """Aucun fichier hors de storage/chroma dans l'archive."""
        zip_path = BaseExporter.export(
            str(fake_chroma), str(tmp_path / "exports" / "base_documentaire_v1.zip")
        )
        with ZipFile(zip_path) as archive:
            for name in archive.namelist():
                assert name.startswith("storage/chroma/"), name

    def test_export_erreur_dossier_introuvable(self, tmp_path):
        with pytest.raises(ValueError):
            BaseExporter.export(
                str(tmp_path / "inexistant"),
                str(tmp_path / "exports" / "base_documentaire_v1.zip"),
            )

    def test_export_erreur_sans_chroma_sqlite(self, tmp_path):
        empty_dir = tmp_path / "storage" / "chroma"
        empty_dir.mkdir(parents=True)
        with pytest.raises(ValueError):
            BaseExporter.export(
                str(empty_dir), str(tmp_path / "exports" / "base_documentaire_v1.zip")
            )

    def test_export_ecrase_l_archive_existante(self, fake_chroma, tmp_path):
        zip_path = str(tmp_path / "exports" / "base_documentaire_v1.zip")
        BaseExporter.export(str(fake_chroma), zip_path)
        BaseExporter.export(str(fake_chroma), zip_path)  # pas d'erreur
        assert Path(zip_path).is_file()


class TestBaseExporterUnzip:
    def test_unzip_restaure_la_base(self, fake_chroma, tmp_path):
        zip_path = BaseExporter.export(
            str(fake_chroma), str(tmp_path / "exports" / "base_documentaire_v1.zip")
        )
        restored = BaseExporter.unzip(zip_path, str(tmp_path / "restaure"))

        sqlite = Path(restored) / "chroma.sqlite3"
        assert sqlite.is_file()
        assert sqlite.read_bytes() == b"fake sqlite header"
        assert (Path(restored) / "5f2a1c9e" / "data_level0.bin").is_file()

    def test_unzip_round_trip_identique(self, fake_chroma, tmp_path):
        """Export → unzip → les fichiers sont bit-à-bit identiques."""
        zip_path = BaseExporter.export(
            str(fake_chroma), str(tmp_path / "exports" / "base_documentaire_v1.zip")
        )
        restored = BaseExporter.unzip(zip_path, str(tmp_path / "restaure"))
        restored_dir = Path(restored)

        for original in fake_chroma.rglob("*"):
            if not original.is_file():
                continue
            relative = original.relative_to(fake_chroma)
            assert (restored_dir / relative).read_bytes() == original.read_bytes()

    def test_unzip_erreur_archive_introuvable(self, tmp_path):
        with pytest.raises(ValueError):
            BaseExporter.unzip(str(tmp_path / "absent.zip"), str(tmp_path / "out"))

    def test_unzip_erreur_sans_chroma_sqlite(self, tmp_path):
        # Archive sous storage/chroma/ mais sans chroma.sqlite3 → rejetée
        # par la validation de la base (et non par la garde de préfixe).
        bad_zip = tmp_path / "incomplete.zip"
        with ZipFile(bad_zip, "w") as archive:
            archive.writestr("storage/chroma/autre.bin", "pas une base chroma")
        with pytest.raises(ValueError):
            BaseExporter.unzip(str(bad_zip), str(tmp_path / "out"))

    def test_unzip_par_defaut_pas_de_double_imbrication(self, fake_chroma, tmp_path, monkeypatch):
        """Unzip par défaut (cible ".") → storage/chroma/ à la racine, jamais
        storage/chroma/storage/chroma/."""
        zip_path = BaseExporter.export(
            str(fake_chroma), str(tmp_path / "exports" / "base_documentaire_v1.zip")
        )
        monkeypatch.chdir(tmp_path)
        restored = BaseExporter.unzip(zip_path)  # cible par défaut : "."
        # Chemin retourné relatif à la cible ("storage/chroma") → résolution.
        assert Path(restored).resolve() == (tmp_path / "storage" / "chroma").resolve()
        assert (tmp_path / "storage" / "chroma" / "chroma.sqlite3").is_file()
        assert not (tmp_path / "storage" / "chroma" / "storage").exists()

    def test_unzip_rejette_membre_hors_storage_chroma(self, tmp_path):
        """Garde anti-traversal : un membre ../ ou hors storage/chroma est rejeté."""
        evil_zip = tmp_path / "malveillant.zip"
        with ZipFile(evil_zip, "w") as archive:
            archive.writestr("storage/chroma/chroma.sqlite3", b"ok")
            archive.writestr("../evil.txt", "traversal")
        with pytest.raises(ValueError):
            BaseExporter.unzip(str(evil_zip), str(tmp_path / "out"))
