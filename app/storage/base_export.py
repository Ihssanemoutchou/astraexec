"""
BaseExporter — Préparation de la base documentaire pour le binôme
==================================================================

MON outil d'export : compresse le dossier ChromaDB persistant
(`storage/chroma/`) dans une archive ZIP brute.

Aucun format propriétaire : ni manifest.json, ni embeddings.npy, ni
chunks.jsonl. L'archive contient strictement le dossier `storage/chroma/`
(chroma.sqlite3 + segments HNSW). Le binôme dézippe puis ouvre
immédiatement :

    chromadb.PersistentClient(path="storage/chroma")

Pipeline :
    storage/chroma/
        ↓
    export()
        ↓
    base_documentaire_v1.zip   (contenu : storage/chroma/...)
        ↓
    unzip()
        ↓
    storage/chroma/  (réutilisable tel quel)

Règles d'encapsulation :
  - aucun import de chromadb ici (ChromaManager est le seul habilité)
  - l'archive est une copie brute du dossier, rien de plus.
"""

from pathlib import Path
from typing import Optional
from zipfile import ZIP_DEFLATED, ZipFile

ARCHIVE_ROOT = "storage/chroma"
DEFAULT_SOURCE = "storage/chroma"
DEFAULT_OUTPUT = "exports/base_documentaire_v1.zip"
DEFAULT_TARGET = "."


class BaseExporter:
    """
    BaseExporter

    Compresse / restaure la base documentaire Chroma telle quelle.

    API :
      - export(source_dir, output_zip)  -> chemin de l'archive créée
      - unzip(zip_path, target_dir)     -> chemin du dossier Chroma restauré
    """

    @staticmethod
    def export(
        source_dir: Optional[str] = None,
        output_zip: Optional[str] = None,
    ) -> str:
        """
        Compresse `storage/chroma/` dans `base_documentaire_v1.zip`.

        Le préfixe d'archive est volontairement `storage/chroma/` : le
        binôme dézippe à la racine de son projet et obtient directement le
        dossier attendu par `PersistentClient(path="storage/chroma")`.
        """
        source = Path(source_dir or DEFAULT_SOURCE)
        output = Path(output_zip or DEFAULT_OUTPUT)

        if not source.is_dir():
            raise ValueError(f"Dossier Chroma introuvable : {source}")
        if not (source / "chroma.sqlite3").is_file():
            raise ValueError(
                f"Base Chroma invalide : chroma.sqlite3 absent de {source}"
            )

        output.parent.mkdir(parents=True, exist_ok=True)

        with ZipFile(output, "w", ZIP_DEFLATED) as archive:
            for file_path in sorted(source.rglob("*")):
                if not file_path.is_file():
                    continue
                arcname = Path(ARCHIVE_ROOT) / file_path.relative_to(source)
                archive.write(file_path, arcname.as_posix())

        return str(output)

    @staticmethod
    def unzip(
        zip_path: Optional[str] = None,
        target_dir: Optional[str] = None,
    ) -> str:
        """
        Restaure la base depuis l'archive et valide la présence de
        `storage/chroma/chroma.sqlite3` après extraction.
        """
        archive = Path(zip_path or DEFAULT_OUTPUT)
        target = Path(target_dir or DEFAULT_TARGET)

        if not archive.is_file():
            raise ValueError(f"Archive introuvable : {archive}")

        with ZipFile(archive, "r") as zf:
            # Garde anti-traversal : seuls les membres sous storage/chroma/
            # sans segment ".." sont extraits (copie brute, rien d'autre).
            for name in zf.namelist():
                if not name.startswith(ARCHIVE_ROOT + "/"):
                    raise ValueError(f"Membre inattendu dans l'archive : {name}")
                if ".." in name.split("/"):
                    raise ValueError(f"Membre suspect dans l'archive : {name}")
            zf.extractall(target)

        chroma_sqlite = target / ARCHIVE_ROOT / "chroma.sqlite3"
        if not chroma_sqlite.is_file():
            raise ValueError(
                "Archive invalide : storage/chroma/chroma.sqlite3 "
                "absent après extraction."
            )

        return str(chroma_sqlite.parent)


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Faux dossier Chroma (structure réelle : sqlite + segments HNSW).
        fake_chroma = root / "storage" / "chroma"
        fake_chroma.mkdir(parents=True)
        (fake_chroma / "chroma.sqlite3").write_bytes(b"fake sqlite")
        (fake_chroma / "data_level0.bin").write_bytes(b"fake hnsw")

        exporter = BaseExporter()
        zip_path = exporter.export(
            str(fake_chroma), str(root / "exports" / "base_documentaire_v1.zip")
        )
        print(f"Archive créée : {zip_path}")

        restored = exporter.unzip(zip_path, str(root / "restaure"))
        print(f"Base restaurée : {restored}")
        print((Path(restored) / "chroma.sqlite3").is_file())
