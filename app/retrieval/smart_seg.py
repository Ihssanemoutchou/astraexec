"""
SmartSeg — Segmentation intelligente maison
============================================

Moteur de segmentation agnostique au format d'entrée : le pipeline
(nettoyage → phrases → sections → chunks) s'applique à du texte brut,
qu'il provienne d'un fichier .txt (via load_text, conservé pour la
rétrocompatibilité) ou de tout autre lecteur externe (ex. PDF via
PDFDocumentReader puis process_text).

Fonctionnalités :
  - Segmentation de texte brut : process_text(text, source)
  - Lecture .txt conservée      : load_text + process(path)
  - Nettoyage avancé du texte
  - Segmentation en phrases
  - Chunking avec overlap
  - Métadonnées enrichies (longueur, mots, source)
"""

import re
from pathlib import Path
from typing import List, Dict


class SmartSeg:
    """
    Segmentation intelligente de documents texte.

    Paramètres :
      - chunk_size : nombre de caractères par chunk
      - overlap    : nombre de caractères de chevauchement
      - min_chunk_size : taille minimale d'un chunk
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def load_text(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document introuvable : {file_path}")
        if path.suffix.lower() not in (".txt",):
            raise ValueError(
                f"Format non supporte : {path.suffix}. "
                f"Seuls les fichiers .txt sont acceptes."
            )
        for enc in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

    def clean_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = text.replace("\r", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def remove_noise(self, text: str) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
        text = re.sub(r"^[=\-_*]{3,}$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9A-ÿ\"'(«])", text)
        return [s.strip() for s in raw if s.strip()]

    def split_sections(self, sentences: List[str]) -> List[str]:
        sections = []
        current = []
        for sentence in sentences:
            current.append(sentence)
            joined = " ".join(current)
            if len(joined) >= self.chunk_size:
                sections.append(joined)
                current = []
        if current:
            sections.append(" ".join(current))
        return sections

    def split_into_chunks(self, section: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(section):
            end = start + self.chunk_size
            chunk = section[start:end]
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks

    def build_metadata(self, chunks: List[str], source: str) -> List[Dict]:
        filename = Path(source).name
        return [
            {
                "chunk_id": idx,
                "source": filename,
                "length": len(chunk),
                "word_count": len(chunk.split()),
                "content": chunk,
            }
            for idx, chunk in enumerate(chunks)
        ]

    def process_text(self, text: str, source: str) -> List[Dict]:
        """
        Segmente un texte déjà chargé en mémoire et retourne les chunks.

        Entrée générique pour tout format (TXT, PDF, ...) : le texte est
        fourni brut par un lecteur externe (ex. ReaderFactory), la source
        sert uniquement aux métadonnées.

        Réutilise exactement la même logique que `process` :
        nettoyage → suppression du bruit → phrases → sections → chunks.
        """
        text = self.clean_text(text)
        text = self.remove_noise(text)
        sentences = self.split_sentences(text)
        sections = self.split_sections(sentences)
        all_chunks = []
        for section in sections:
            all_chunks.extend(self.split_into_chunks(section))
        return self.build_metadata(all_chunks, source)

    def process(self, file_path: str) -> List[Dict]:
        text = self.load_text(file_path)
        return self.process_text(text, file_path)

    def info(self) -> Dict:
        return {
            "engine": "SmartSeg (segmentation texte pur)",
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "min_chunk_size": self.min_chunk_size,
        }


if __name__ == "__main__":
    processor = SmartSeg()
    print(processor.info())

    import tempfile, os

    test = (
        "Python est un langage de programmation interprete.\n\n"
        "Il est utilise dans de nombreux domaines : le developpement web, "
        "l'analyse de donnees, le machine learning et bien d'autres.\n\n"
        "L'un de ses avantages principaux est sa syntaxe claire et concise."
    )
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(test)
    tp = tmp.name
    tmp.close()
    try:
        chunks = processor.process(tp)
        print(f"\n{len(chunks)} chunks")
        for c in chunks[:3]:
            print(f"  #{c['chunk_id']} ({c['length']}c, {c['word_count']}mots): {c['content'][:80]}...")
    finally:
        os.unlink(tp)
