"""Unit tests for the PlateOCR text cleaning pipeline."""

import unittest
import sys

try:
    import easyocr
    # src.models.ocr itself raises ImportError when easyocr is absent (it is
    # train/eval-only and not installed in the runtime Docker image), so this
    # import must live inside the same guard for the skipIf below to work.
    from src.models.ocr import PlateOCR
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    PlateOCR = None


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


@unittest.skipIf(not EASYOCR_AVAILABLE, "EasyOCR not installed — skipping OCR tests")
class TestPlateOCRRowClustering(unittest.TestCase):
    """Edge cases for PlateOCR._sort_and_merge row clustering.

    Exercises the /3 row-threshold logic on synthetic easyocr-style
    results (no real OCR model is run).
    """

    @classmethod
    def setUpClass(cls):
        """Instantiate PlateOCR once for all row-clustering tests."""
        cls.ocr = PlateOCR(languages=["en"], gpu=False)

    @staticmethod
    def _box(cx, cy, w=20, h=10):
        """Build an easyocr-style 4-corner bbox centred at ``(cx, cy)``."""
        x0, x1 = cx - w / 2, cx + w / 2
        y0, y1 = cy - h / 2, cy + h / 2
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    def test_single_line_plate_merges_left_to_right(self):
        """Boxes at the same y, given in shuffled x-order, form one row
        joined left-to-right."""
        results = [
            (self._box(60, 20), "F", 0.9),
            (self._box(10, 20), "30", 0.9),
            (self._box(110, 20), "12345", 0.9),
        ]
        self.assertEqual(self.ocr._sort_and_merge(results), "30F12345")

    def test_two_line_plate_top_row_precedes_bottom_row(self):
        """Two clear y-clusters (VN two-line plate): the whole top row
        comes before the bottom row, each row still left-to-right."""
        results = [
            (self._box(30, 60), "12345", 0.9),  # bottom row
            (self._box(50, 15), "F", 0.9),      # top row, right
            (self._box(15, 15), "30", 0.9),     # top row, left
        ]
        self.assertEqual(self.ocr._sort_and_merge(results), "30F12345")


if __name__ == "__main__":
    unittest.main()
