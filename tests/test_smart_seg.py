"""
Tests pour SmartSeg.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.retrieval.smart_seg import SmartSeg


class TestSmartSeg:
    def test_clean_text(self):
        seg = SmartSeg()
        cleaned = seg.clean_text("  Hello   World  \n\n\nTest  ")
        assert "  " not in cleaned

    def test_remove_noise(self):
        seg = SmartSeg()
        cleaned = seg.remove_noise("Hello\x00World")
        assert "\x00" not in cleaned

    def test_split_sentences(self):
        seg = SmartSeg()
        sentences = seg.split_sentences("Première phrase. Deuxième phrase.")
        assert len(sentences) >= 2

    def test_info(self):
        seg = SmartSeg()
        info = seg.info()
        assert info["engine"] == "SmartSeg (segmentation texte pur)"
        assert info["chunk_size"] == 500
