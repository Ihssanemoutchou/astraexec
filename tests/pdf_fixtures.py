"""
Fixtures PDF pour les tests du support PDF (Livrable 3).

Tous les PDF sont générés programmatiquement avec PyMuPDF afin de
ne dépendre d'aucun fichier externe (le corpus ne contient pas de PDF
de référence pour les tests unitaires).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pymupdf as fitz
except ImportError:
    import fitz


def _insert_wrapped(page, text, fontsize=12):
    """
    Insère un texte sur une page PDF en gérant le retour à la ligne.

    `insert_text` de PyMuPDF ne retourne pas à la ligne : un texte trop
    long dépasse le bord droit de la page et est rogné à l'extraction
    (le contenu hors page n'est pas récupérable). On découpe donc le
    texte en segments qui tiennent sur une ligne A4 (~70 caractères
    à fontsize 12) et on insère chaque segment sur une ligne dédiée.
    """
    line_width = 70
    y = 72
    for index in range(0, len(text), line_width):
        segment = text[index:index + line_width]
        page.insert_text((72, y), segment, fontsize=fontsize)
        y += 16


def make_pdf(path, texts=("Texte de test PDF.",), password=None):
    """
    Crée un PDF contenant une page par chaîne de `texts`.

    Paramètres :
        path     : chemin du fichier PDF à créer.
        texts    : liste des textes (un par page).
        password : mot de passe utilisateur (PDF protégé) ou None.
    """
    document = fitz.open()
    for text in texts:
        page = document.new_page()
        _insert_wrapped(page, text)

    if password:
        document.save(
            str(path),
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw=password,
            user_pw=password,
        )
    else:
        document.save(str(path))
    document.close()


def make_empty_pdf(path):
    """
    Crée un PDF valide à 0 page en bytes bruts.

    PyMuPDF refuse de sauvegarder un document sans page
    (« cannot save with zero pages »), d'où l'écriture manuelle.
    """
    with open(path, "wb") as handle:
        handle.write(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
            b"trailer\n<< /Root 1 0 R /Size 2 >>\n%%EOF\n"
        )


def make_corrupted_pdf(path):
    """Crée un fichier .pdf volontairement invalide (bytes quelconques)."""
    with open(path, "wb") as handle:
        handle.write(b"ceci n'est pas un fichier pdf valide")


def make_scanned_pdf(path):
    """
    Crée un PDF « scanné » : une page contenant une image,
    sans aucun texte numérique extractible.
    """
    document = fitz.open()
    page = document.new_page()

    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100))
    pixmap.clear_with(200)  # remplissage gris clair
    page.insert_image(page.rect, pixmap=pixmap)

    document.save(str(path))
    document.close()


def make_big_pdf(path, page_count=50, text="Page de contenu de test. " * 20):
    """
    Crée un PDF volumineux (par défaut 50 pages) pour vérifier
    la robustesse de l'extraction sur de gros documents.
    """
    document = fitz.open()
    for _ in range(page_count):
        page = document.new_page()
        _insert_wrapped(page, text)
    document.save(str(path))
    document.close()


def find_emoji_font():
    """
    Retourne le chemin d'une police capable d'afficher des emojis,
    ou None si aucune n'est disponible sur la machine.
    """
    candidates = [
        r"C:\Windows\Fonts\seguiemj.ttf",      # Segoe UI Emoji (Windows)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None
