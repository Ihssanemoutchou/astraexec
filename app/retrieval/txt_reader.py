"""
TXTDocumentReader — Lecture des fichiers texte
================================================

Lecture seule d'un fichier .txt avec détection automatique
de l'encodage :

    1. ouverture du fichier
    2. détection de l'encodage (utf-8, latin-1, cp1252, iso-8859-1)
    3. retour de la chaîne de caractères

Responsabilité unique : la LECTURE.

Hors périmètre (traités par d'autres composants) :
    - nettoyage du texte      → TextCleaner / SmartSeg
    - segmentation / chunks   → ChunkGenerator / SmartSeg
    - métadonnées             → MetadataBuilder / SmartSeg

NB : la boucle d'encodages est volontairement IDENTIQUE à celle de
`SmartSeg.load_text` afin de garantir une sortie bit-à-bit identique
sur les fichiers .txt existants (non-régression du Livrable 2).
"""

from pathlib import Path


class TXTDocumentReader:
    """
    TXTDocumentReader

    Lecture seule des fichiers .txt avec détection d'encodage.
    Retourne le texte brut du fichier.
    """

    EXTENSIONS = (".txt",)
    ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

    # =====================================================
    # Lecture
    # =====================================================

    def read(self, file_path: str) -> str:
        """
        Lit un fichier .txt et retourne son contenu brut.

        Paramètres :
            file_path : chemin vers le fichier texte.

        Retour :
            chaîne de caractères contenant le texte du fichier.

        Exceptions :
            FileNotFoundError : fichier inexistant.
            ValueError        : extension non supportée.
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

        # Détection de l'encodage (même stratégie que SmartSeg).
        for encoding in self.ENCODINGS:
            try:
                with open(path, "r", encoding=encoding) as handle:
                    return handle.read()
            except (UnicodeDecodeError, UnicodeError):
                continue

        # Repli ultime : latin-1 décodera toujours (aucune erreur).
        with open(path, "r", encoding="latin-1") as handle:
            return handle.read()


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    import tempfile
    import os

    reader = TXTDocumentReader()

    # 1. Fichier UTF-8 avec accents
    tmp_utf8 = tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", encoding="utf-8", delete=False
    )
    tmp_utf8.write(
        "Évaluation de la pertinence : BM25, TF-IDF et deep learning."
    )
    tmp_utf8.close()

    try:
        text = reader.read(tmp_utf8.name)
        print(f"TXT UTF-8 lu ({len(text)} caractères) :")
        print(text)
    finally:
        os.unlink(tmp_utf8.name)

    # 2. Fichier CP1252 (encodage historique Windows)
    tmp_cp = tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", encoding="cp1252", delete=False
    )
    tmp_cp.write("Recherche lexicale avec accents : déjà, où, çà.")
    tmp_cp.close()

    try:
        text = reader.read(tmp_cp.name)
        print(f"\nTXT CP1252 lu : {text}")
    finally:
        os.unlink(tmp_cp.name)

    # 3. Fichier inexistant
    try:
        reader.read("fichier_inexistant.txt")
    except FileNotFoundError as exc:
        print(f"\nFichier introuvable géré : {exc}")
