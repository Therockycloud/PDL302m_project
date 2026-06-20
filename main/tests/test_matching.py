import unittest
import os
import pandas as pd
from src.utils.matching import DatabaseMatcher


class TestDatabaseMatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_db_path = "tests_temp_database.csv"
        db_data = {
            'license_plate': ['30F-12345', '51G-67890'],
            'car_brand': ['Toyota Vios', 'Hyundai Accent'],
            'car_color': ['White', 'Black'],
        }
        pd.DataFrame(db_data).to_csv(cls.temp_db_path, index=False)
        cls.matcher = DatabaseMatcher(cls.temp_db_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_db_path):
            os.remove(cls.temp_db_path)

    def test_exact_match(self):
        result = self.matcher.verify_vehicle("30F-12345", "White")
        self.assertEqual(result['status'], 'AUTHORIZED')
        self.assertEqual(result['action'], 'ALLOW')

    def test_case_and_symbol_insensitivity(self):
        result = self.matcher.verify_vehicle("30f - 123.45", "white")
        self.assertEqual(result['status'], 'AUTHORIZED')

    def test_unregistered_plate(self):
        result = self.matcher.verify_vehicle("30H-99999", "Blue")
        self.assertEqual(result['status'], 'UNREGISTERED')
        self.assertEqual(result['action'], 'DENY_ALERT')

    def test_color_warning_is_authorized_not_denied(self):
        # Plate-primary: a correct plate with a differing colour is AUTHORIZED
        # but flagged (soft warning), not hard-denied.
        result = self.matcher.verify_vehicle("30F-12345", "Black")
        self.assertEqual(result['status'], 'AUTHORIZED')
        self.assertEqual(result['action'], 'ALLOW_WARN')
        self.assertTrue(result['color_warning'])
        self.assertIn("colour", result['message'].lower())


class TestRealDatabaseDemoPlate(unittest.TestCase):
    """WS-1 Task 6: the live demo plate (30M71854, locked at frame 510 of
    sample_parking.mp4) must be registered AUTHORIZED with no colour
    warning against the REAL main/data/database.csv (not a temp fixture).
    Colour ground truth: TorchColorClassifier (color_MobileNetV3Small.pt)
    predicted Yellow at conf 0.82-0.95 across 8/9 sampled frames
    (492-534) of the vehicle's own crop.
    """

    @classmethod
    def setUpClass(cls):
        db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "database.csv"
        )
        cls.matcher = DatabaseMatcher(db_path)

    def test_demo_plate_is_authorized_with_correct_color(self):
        result = self.matcher.verify_vehicle("30M71854", "Yellow")
        self.assertEqual(result['status'], 'AUTHORIZED')
        self.assertFalse(result['color_warning'])

    def test_demo_plate_matches_with_dashes_and_dots(self):
        # Matcher strips space/dash/dot, so the CSV's dashed form must
        # round-trip with the OCR's bare-digits form.
        result = self.matcher.verify_vehicle("30M-718.54", "Yellow")
        self.assertEqual(result['status'], 'AUTHORIZED')


if __name__ == '__main__':
    unittest.main()
