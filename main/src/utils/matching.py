import pandas as pd
import os

class DatabaseMatcher:
    def __init__(self, db_path: str):
        """
        Initializes the DatabaseMatcher with the path to the registration CSV file.
        """
        self.db_path = db_path
        self.db = None
        self.load_database()

    def load_database(self):
        """
        Loads the CSV database into a pandas DataFrame.
        """
        if os.path.exists(self.db_path):
            self.db = pd.read_csv(self.db_path)
            # Standardize columns for robust matching
            self.db['license_plate'] = self.db['license_plate'].astype(str).str.replace(r'[\s\-\.]', '', regex=True).str.upper()
            self.db['car_brand'] = self.db['car_brand'].astype(str).str.strip().str.upper()
            self.db['car_color'] = self.db['car_color'].astype(str).str.strip().str.upper()
        else:
            raise FileNotFoundError(f"Database file not found at: {self.db_path}")

    def verify_vehicle(self, detected_plate: str, detected_color: str) -> dict:
        """Verify a detected vehicle against the registered database.

        Matching is plate-first; colour is a secondary verification layer
        to catch a real plate cloned onto a different vehicle.

        Args:
            detected_plate: The recognized plate sequence.
            detected_color: The classified car colour.

        Returns:
            dict with 'status', 'action', and 'message' keys.
        """
        if self.db is None:
            return {'status': 'ERROR', 'action': 'DENY', 'message': 'Database not loaded'}

        clean_plate = str(detected_plate).replace(' ', '').replace('-', '').replace('.', '').upper()
        clean_color = str(detected_color).strip().upper()

        record = self.db[self.db['license_plate'] == clean_plate]

        if record.empty:
            return {
                'status': 'UNREGISTERED',
                'action': 'DENY_ALERT',
                'message': f"Plate {detected_plate} is not registered in the system.",
            }

        registered_color = record.iloc[0]['car_color']
        color_match = clean_color == registered_color

        if color_match:
            return {
                'status': 'AUTHORIZED',
                'action': 'ALLOW',
                'message': f"Vehicle {detected_plate} authorized: Match confirmed.",
            }
        return {
            'status': 'MISMATCH',
            'action': 'DENY_ALERT',
            'message': f"Color Mismatch (Detected: {detected_color}, Registered: {record.iloc[0]['car_color']})",
        }
