# Helper and Utility Modules

This directory contains utility functions for database matching, text post-processing, and visual indicators.

---

## 1. Database Matcher (`matching.py`)
* **Purpose:** Verify if the detected vehicle features match the registration record in the CSV file.
* **Key Tasks:**
  * Load `main/data/database.csv` as a dictionary or pandas DataFrame.
  * Define function `verify_vehicle(detected_plate: str, detected_brand: str, detected_color: str) -> dict`:
    * Look up `detected_plate` in database.
    * If plate is not found $\rightarrow$ return `{'status': 'UNREGISTERED', 'action': 'DENY'}`.
    * If plate is found $\rightarrow$ compare `detected_brand` and `detected_color` with registered values (using string matching or Levenshtein distance for brand variations).
    * If features match $\rightarrow$ return `{'status': 'AUTHORIZED', 'action': 'ALLOW'}`.
    * If features do not match $\rightarrow$ return `{'status': 'MISMATCH', 'action': 'DENY_ALERT'}`.

---

## 2. Visualization overlay (`visualization.py`)
* **Purpose:** Annotate image frames with bounding boxes, labels, and security alerts.
* **Key Tasks:**
  * Draw bounding boxes around license plates.
  * Draw text showing detected plate character, brand, and color.
  * Overlay status flags:
    * **GREEN BANNER** for `AUTHORIZED` cars.
    * **FLASHING RED BANNER** for `MISMATCH` or `UNREGISTERED` cars.
  * Trigger sound effects (warning beeps) using system audio libraries when an alarm is triggered.
