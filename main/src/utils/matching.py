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

    def verify_vehicle(self, detected_plate: str, detected_brand: str, detected_color: str) -> dict:
        """
        Verifies if the detected vehicle matches the registered database records.
        
        Args:
            detected_plate (str): The recognized plate sequence.
            detected_brand (str): The classified car brand.
            detected_color (str): The classified car color.
            
        Returns:
            dict: Status dictionary containing 'status' and 'action' keys.
        """
        if self.db is None:
            return {'status': 'ERROR', 'action': 'DENY', 'message': 'Database not loaded'}

        # Sanitize inputs
        clean_plate = str(detected_plate).replace(' ', '').replace('-', '').replace('.', '').upper()
        clean_brand = str(detected_brand).strip().upper()
        clean_color = str(detected_color).strip().upper()

        # Query plate
        record = self.db[self.db['license_plate'] == clean_plate]

        if record.empty:
            return {
                'status': 'UNREGISTERED',
                'action': 'DENY_ALERT',
                'message': f"Plate {detected_plate} is not registered in the system."
            }

        registered_brand = record.iloc[0]['car_brand']
        registered_color = record.iloc[0]['car_color']

        # Simple verification checks (can be expanded to use fuzzy string matching)
        brand_match = clean_brand in registered_brand or registered_brand in clean_brand
        color_match = clean_color == registered_color

        if brand_match and color_match:
            return {
                'status': 'AUTHORIZED',
                'action': 'ALLOW',
                'message': f"Vehicle {detected_plate} authorized: Match confirmed."
            }
        else:
            mismatch_reasons = []
            if not brand_match:
                mismatch_reasons.append(f"Brand Mismatch (Detected: {detected_brand}, Registered: {record.iloc[0]['car_brand']})")
            if not color_match:
                mismatch_reasons.append(f"Color Mismatch (Detected: {detected_color}, Registered: {record.iloc[0]['car_color']})")

            return {
                'status': 'MISMATCH',
                'action': 'DENY_ALERT',
                'message': " | ".join(mismatch_reasons)
            }
