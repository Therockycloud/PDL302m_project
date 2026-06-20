import pandas as pd
import os
import yaml

# WS-2: neutral colours are visually/perceptually ambiguous (Report 3 §5.1
# confusion matrix) -- treat them as equivalent to each other so the colour
# classifier's known weak cluster doesn't drive false alarms. Defaults here
# are used when no config override is found; both are read once at import
# time from main/configs/config.yaml's `decision:` block if present, so
# tests that construct a DatabaseMatcher directly (no config involved) still
# get sane defaults.
NEUTRAL = {"BLACK", "GREY", "SILVER", "WHITE"}
COLOR_WARN_CONF = 0.60

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "config.yaml"
)


def _load_decision_overrides():
    """Best-effort read of the `decision:` block from config.yaml. Silently
    keeps the hard-coded defaults above if the file is missing, unreadable,
    or doesn't have the block -- config is optional, not required, so unit
    tests never need a config.yaml on disk."""
    global NEUTRAL, COLOR_WARN_CONF
    try:
        with open(_CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        decision_cfg = cfg.get("decision") or {}
        neutral_override = decision_cfg.get("neutral_colors")
        if neutral_override:
            NEUTRAL = {str(c).strip().upper() for c in neutral_override}
        conf_override = decision_cfg.get("color_warn_conf")
        if conf_override is not None:
            COLOR_WARN_CONF = float(conf_override)
    except (FileNotFoundError, OSError, yaml.YAMLError):
        pass


_load_decision_overrides()


def _colors_equivalent(c1: str, c2: str) -> bool:
    """True if c1/c2 should be treated as a colour match: identical, or both
    members of the neutral cluster (Black/Grey/Silver/White by default --
    these are mutually confusable both to the model and to the human eye in
    parking-lot lighting, so a mismatch between them is not a meaningful
    signal of plate cloning)."""
    return c1 == c2 or (c1 in NEUTRAL and c2 in NEUTRAL)


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

    def verify_vehicle(self, detected_plate: str, detected_color: str, color_conf: float = None) -> dict:
        """Verify a detected vehicle against the registered database.

        Plate-primary: the plate is the key. A registered plate is AUTHORIZED;
        colour is a *soft* secondary signal — if it differs (possible plate
        cloning, or just a noisy colour prediction) the vehicle is still
        AUTHORIZED but flagged with ``action='ALLOW_WARN'`` and
        ``color_warning=True`` rather than hard-denied. This avoids false
        denials from an imperfect colour classifier while still surfacing the
        discrepancy. An unregistered plate is denied.

        WS-2 additions (cuts the measured 14.5% false-alarm rate):
          - Neutral-cluster equivalence: Black/Grey/Silver/White (configurable
            via config.yaml `decision.neutral_colors`) are treated as a
            matching colour against each other, since they're the colour
            classifier's known confusable cluster (Report 3 §5.1) and a
            mismatch within it carries little signal.
          - Confidence gating: a cross-cluster mismatch only raises a warning
            when the detector was reasonably confident in its colour read
            (``color_conf >= COLOR_WARN_CONF``, default 0.60, configurable via
            `decision.color_warn_conf`). A low-confidence mismatch is too
            likely to be classifier noise to be worth surfacing as a warning.

        Args:
            detected_plate: The recognized plate sequence.
            detected_color: The classified car colour.
            color_conf: Confidence of the colour classification (0..1). When
                ``None`` (legacy 2-argument callers), confidence is treated as
                unknown/trusted so a cross-cluster mismatch still warns —
                preserving old behaviour for callers that don't pass it.

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
        if _colors_equivalent(clean_color, registered_color):
            return {
                'status': 'AUTHORIZED',
                'action': 'ALLOW',
                'message': f"Vehicle {detected_plate} authorized: plate and colour match.",
                'color_warning': False,
            }

        if color_conf is not None and color_conf < COLOR_WARN_CONF:
            return {
                'status': 'AUTHORIZED',
                'action': 'ALLOW',
                'message': (
                    f"Vehicle {detected_plate} authorized by plate; colour differs "
                    f"(detected {detected_color}, registered {record.iloc[0]['car_color']}) "
                    f"but colour confidence ({color_conf:.2f}) is below the warning "
                    f"threshold ({COLOR_WARN_CONF:.2f}), so not flagged."
                ),
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
