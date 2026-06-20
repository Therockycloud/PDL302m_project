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
        # WS-2: White and Black are BOTH in the neutral cluster, so under the
        # new neutral-cluster-equivalence logic this is no longer a warning
        # (previously this asserted color_warning=True; the cluster merge is
        # the whole point of WS-2 — see test_cross_cluster_high_conf_warns
        # below for a genuine cross-cluster case, which DOES still warn).
        result = self.matcher.verify_vehicle("30F-12345", "Black")
        self.assertEqual(result['status'], 'AUTHORIZED')
        self.assertEqual(result['action'], 'ALLOW')
        self.assertFalse(result['color_warning'])


class TestNeutralClusterAndConfidenceGating(unittest.TestCase):
    """WS-2: neutral-cluster colour equivalence + colour-confidence gating
    to cut the false-alarm rate (Report 4 §4.3 measured 14.5% before this)."""

    @classmethod
    def setUpClass(cls):
        cls.temp_db_path = "tests_temp_database_ws2.csv"
        db_data = {
            'license_plate': ['51A-001', '51A-002'],
            'car_brand': ['Honda Civic', 'Kia Morning'],
            'car_color': ['Red', 'Grey'],
        }
        pd.DataFrame(db_data).to_csv(cls.temp_db_path, index=False)
        cls.matcher = DatabaseMatcher(cls.temp_db_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_db_path):
            os.remove(cls.temp_db_path)

    def test_neutral_cluster_no_warning(self):
        # registered GREY, detected SILVER (both neutral) -> NO warning even
        # at high confidence: the clusters are treated as equivalent.
        r = self.matcher.verify_vehicle("51A-002", "SILVER", 0.95)
        self.assertEqual(r["status"], "AUTHORIZED")
        self.assertFalse(r["color_warning"])

    def test_cross_cluster_high_conf_warns(self):
        r = self.matcher.verify_vehicle("51A-001", "BLUE", 0.95)  # RED reg, BLUE det
        self.assertTrue(r["color_warning"])
        self.assertEqual(r["action"], "ALLOW_WARN")

    def test_low_conf_no_warning(self):
        # WS-2 gate lowered 0.60 -> 0.40 (docs/benchmarks/security_eval.md):
        # use 0.20, safely below the deployed 0.40 threshold, so this still
        # exercises "mismatch but low confidence -> no warning" regardless of
        # the exact gate value configured.
        r = self.matcher.verify_vehicle("51A-001", "BLUE", 0.20)  # mismatch but conf<0.40
        self.assertFalse(r["color_warning"])
        self.assertEqual(r["action"], "ALLOW")

    def test_exact_match_no_warning(self):
        r = self.matcher.verify_vehicle("51A-001", "RED", 0.95)
        self.assertFalse(r["color_warning"])

    def test_color_conf_none_defaults_to_warn_like_legacy_caller(self):
        # Legacy 2-arg callers (color_conf=None) must keep warning behaviour
        # for cross-cluster mismatches -- None means "no confidence info
        # available", which is treated the same as "trust it" so old callers
        # don't silently lose their warning.
        r = self.matcher.verify_vehicle("51A-001", "BLUE")
        self.assertTrue(r["color_warning"])


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
