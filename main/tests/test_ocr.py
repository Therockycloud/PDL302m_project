"""Unit tests for the PlateOCR text cleaning pipeline."""

import unittest
import sys

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from src.models.ocr import PlateOCR


@unittest.skipIf(not EASYOCR_AVAILABLE, "EasyOCR not installed — skipping OCR tests")
class TestPlateOCR(unittest.TestCase):
    """Tests for PlateOCR._clean_text behaviour."""

    @classmethod
    def setUpClass(cls):
        """Instantiate PlateOCR once for all text-cleaning tests."""
        cls.ocr = PlateOCR(languages=["en"], gpu=False)

    def test_clean_text_strips_spaces(self):
        """Verify spaces are removed from raw plate text."""
        result = self.ocr._clean_text("A B C")
        self.assertEqual(result, "ABC")

    def test_clean_text_strips_dashes(self):
        """Verify dashes are removed from raw plate text."""
        result = self.ocr._clean_text("30F-12345")
        self.assertEqual(result, "30F12345")

    def test_clean_text_strips_dots(self):
        """Verify dots are removed from raw plate text."""
        result = self.ocr._clean_text("30F.12345")
        self.assertEqual(result, "30F12345")

    def test_clean_text_uppercase(self):
        """Verify lowercase input is uppercased."""
        result = self.ocr._clean_text("abc123")
        self.assertEqual(result, "ABC123")

    def test_clean_text_combined(self):
        """Verify mixed noise (spaces, dashes, dots, lowercase) is fully cleaned."""
        result = self.ocr._clean_text("30f - 123.45")
        self.assertEqual(result, "30F12345")


if __name__ == "__main__":
    unittest.main()
