import unittest
import os
import pandas as pd
from src.utils.matching import DatabaseMatcher

class TestDatabaseMatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary database CSV file for testing
        cls.temp_db_path = "tests_temp_database.csv"
        db_data = {
            'license_plate': ['30F-12345', '51G-67890'],
            'car_brand': ['Toyota Vios', 'Hyundai Accent'],
            'car_color': ['White', 'Black']
        }
        pd.DataFrame(db_data).to_csv(cls.temp_db_path, index=False)
        cls.matcher = DatabaseMatcher(cls.temp_db_path)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary database CSV
        if os.path.exists(cls.temp_db_path):
            os.remove(cls.temp_db_path)

    def test_exact_match(self):
        result = self.matcher.verify_vehicle("30F-12345", "Toyota Vios", "White")
        self.assertEqual(result['status'], 'AUTHORIZED')
        self.assertEqual(result['action'], 'ALLOW')

    def test_case_and_symbol_insensitivity(self):
        result = self.matcher.verify_vehicle("30f - 123.45", "toyota vios", "white")
        self.assertEqual(result['status'], 'AUTHORIZED')

    def test_unregistered_plate(self):
        result = self.matcher.verify_vehicle("30H-99999", "Honda Civic", "Blue")
        self.assertEqual(result['status'], 'UNREGISTERED')
        self.assertEqual(result['action'], 'DENY_ALERT')

    def test_brand_mismatch(self):
        result = self.matcher.verify_vehicle("30F-12345", "Hyundai Accent", "White")
        self.assertEqual(result['status'], 'MISMATCH')
        self.assertEqual(result['action'], 'DENY_ALERT')
        self.assertIn("Brand Mismatch", result['message'])

    def test_color_mismatch(self):
        result = self.matcher.verify_vehicle("30F-12345", "Toyota Vios", "Black")
        self.assertEqual(result['status'], 'MISMATCH')
        self.assertEqual(result['action'], 'DENY_ALERT')
        self.assertIn("Color Mismatch", result['message'])

if __name__ == '__main__':
    unittest.main()
