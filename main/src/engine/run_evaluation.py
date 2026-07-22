import os
import cv2
import sys
import time
import numpy as np

# Project root (three levels up from main/src/engine/). Added to sys.path and
# used as the base for every data/model path so the script runs from any CWD.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(PROJECT_ROOT)

from main.src.models.detector import PlateDetector
from main.src.models.ocr import PlateOCR
from main.src.models.classifiers import BrandClassifier, ColorClassifier
from main.src.utils.matching import DatabaseMatcher

def main():
    print("Initializing components...")
    detector = PlateDetector()
    ocr = PlateOCR()

    brand_clf = BrandClassifier()
    brand_clf.build_model()
    brand_clf.load_weights(
        os.path.join(PROJECT_ROOT, "main", "data", "models", "brand_classifier.keras")
    )

    color_clf = ColorClassifier()
    color_clf.build_model()
    color_clf.load_weights(
        os.path.join(PROJECT_ROOT, "main", "data", "models", "color_classifier.keras")
    )

    db_path = os.path.join(PROJECT_ROOT, "main", "data", "database.csv")
    matcher = DatabaseMatcher(db_path)

    image_dir = os.path.join(PROJECT_ROOT, "main", "data", "raw", "license_plates")
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(supported_ext)
    )

    print(f"Running batch evaluation on {len(image_files)} images...")
    results = []

    for idx, filename in enumerate(image_files, start=1):
        filepath = os.path.join(image_dir, filename)
        start_time = time.perf_counter()
        
        result = {
            "filename": filename,
            "plate_text": "",
            "brand": "",
            "brand_confidence": 0.0,
            "color": "",
            "color_confidence": 0.0,
            "status": "ERROR",
            "action": "DENY_ALERT",
            "message": "",
            "latency_ms": 0.0,
        }

        try:
            image = cv2.imread(filepath)
            if image is None:
                print(f"[{idx}/{len(image_files)}] {filename} -> Could not read image.")
                continue

            # 1. Plate detection
            plates = detector.detect(image)
            
            # 2. OCR
            plate_text = ""
            if plates:
                plate_crop = plates[0]["cropped_plate"]
                plate_text = ocr.read_plate(plate_crop)
                result["plate_text"] = plate_text

            # 3. Brand classification
            brand, brand_conf = brand_clf.predict(image)
            result["brand"] = brand
            result["brand_confidence"] = float(brand_conf)

            # 4. Color classification
            color, color_conf = color_clf.predict(image)
            result["color"] = color
            result["color_confidence"] = float(color_conf)

            # 5. Database verification
            match_result = matcher.verify_vehicle(
                plate_text, color
            )
            result["status"] = match_result.get("status", "ERROR")
            result["action"] = match_result.get("action", "DENY_ALERT")
            result["message"] = match_result.get("message", "")

        except Exception as exc:
            result["status"] = "ERROR"
            result["message"] = str(exc)

        result["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
        results.append(result)
        
        print(f"[{idx}/{len(image_files)}] {filename} processed in {result['latency_ms']} ms -> {result['status']}")

    # Print final output
    print("\n" + "="*50)
    print("               E2E EVALUATION RESULTS")
    print("="*50)
    for r in results:
        print(f"File: {r['filename']}")
        print(f"  Detected Plate : '{r.get('plate_text')}'")
        print(f"  Detected Brand : {r.get('brand')} (Conf: {r.get('brand_confidence', 0.0):.4f})")
        print(f"  Detected Color : {r.get('color')} (Conf: {r.get('color_confidence', 0.0):.4f})")
        print(f"  Match Status   : {r.get('status')}")
        print(f"  Action         : {r.get('action')}")
        print(f"  Message        : {r.get('message')}")
        print(f"  Latency        : {r.get('latency_ms')} ms")
        print("-"*50)

    # Compute metrics
    total = len(results)
    if total > 0:
        latencies = [r.get("latency_ms", 0.0) for r in results]
        statuses = [r.get("status", "ERROR") for r in results]
        avg_latency = round(sum(latencies) / total, 2)
        metrics = {
            "avg_latency_ms": avg_latency,
            "total_processed": total,
            "authorized_count": statuses.count("AUTHORIZED"),
            "mismatch_count": statuses.count("MISMATCH"),
            "unregistered_count": statuses.count("UNREGISTERED"),
        }
    else:
        metrics = {
            "avg_latency_ms": 0.0,
            "total_processed": 0,
            "authorized_count": 0,
            "mismatch_count": 0,
            "unregistered_count": 0,
        }

    print("\n" + "="*50)
    print("               AGGREGATE METRICS")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k:<20}: {v}")
    print("="*50)

if __name__ == "__main__":
    main()
