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

        Plate-primary: the plate is the key. A registered plate is AUTHORIZED;
        colour is a *soft* secondary signal — if it differs (possible plate
        cloning, or just a noisy colour prediction) the vehicle is still
        AUTHORIZED but flagged with ``action='ALLOW_WARN'`` and
        ``color_warning=True`` rather than hard-denied. This avoids false
        denials from an imperfect colour classifier while still surfacing the
        discrepancy. An unregistered plate is denied.

        Args:
            detected_plate: The recognized plate sequence.
            detected_color: The classified car colour.

        Returns:
            dict with 'status', 'action', 'message' and 'color_warning' keys.
        """
        if self.db is None:
            return {'status': 'ERROR', 'action': 'DENY', 'message': 'Database not loaded',
                    'color_warning': False}

        clean_plate = str(detected_plate).replace(' ', '').replace('-', '').replace('.', '').upper()
        clean_color = str(detected_color).strip().upper()

        record = self.db[self.db['license_plate'] == clean_plate]

        if record.empty:
            return {
                'status': 'UNREGISTERED',
                'action': 'DENY_ALERT',
                'message': f"Plate {detected_plate} is not registered in the system.",
                'color_warning': False,
            }

        registered_color = record.iloc[0]['car_color']
        if clean_color == registered_color:
            return {
                'status': 'AUTHORIZED',
                'action': 'ALLOW',
                'message': f"Vehicle {detected_plate} authorized: plate and colour match.",
                'color_warning': False,
            }
        return {
            'status': 'AUTHORIZED',
            'action': 'ALLOW_WARN',
            'message': (
                f"Vehicle {detected_plate} authorized by plate; colour differs "
                f"(detected {detected_color}, registered {record.iloc[0]['car_color']})."
            ),
            'color_warning': True,
        }
