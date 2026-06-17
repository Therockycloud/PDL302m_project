import os
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Clean dataset semantically using YOLOv8.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing images to clean.')
    parser.add_argument('--garbage_dir', type=str, default='../data/raw/garbage', help='Path to move garbage images to.')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detection.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, args.data_dir))
    garbage_dir = os.path.abspath(os.path.join(script_dir, args.garbage_dir))
    
    os.makedirs(garbage_dir, exist_ok=True)

    print(f"Loading YOLOv8n model...")
    # Load the COCO pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # COCO classes for vehicles
    # 2: car, 5: bus, 7: truck
    vehicle_classes = [2, 5, 7]

    print(f"Scanning directory: {data_dir}")
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    total_images = 0
    removed_images = 0

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                total_images += 1
                img_path = os.path.join(root, file)
                
                try:
                    # Run inference
                    results = model(img_path, conf=args.conf, verbose=False)
                    has_vehicle = False
                    
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            if cls_id in vehicle_classes:
                                has_vehicle = True
                                break
                        if has_vehicle:
                            break
                    
                    if not has_vehicle:
                        # Move to garbage
                        rel_path = os.path.relpath(img_path, data_dir)
                        # Ensure safe filename in garbage
                        safe_name = rel_path.replace(os.sep, '_')
                        dest_path = os.path.join(garbage_dir, safe_name)
                        
                        shutil.move(img_path, dest_path)
                        removed_images += 1
                        print(f"Moved to garbage: {rel_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print("\n--- Semantic Cleaning Summary ---")
    print(f"Total images scanned: {total_images}")
    print(f"Moved to garbage (no vehicle): {removed_images}")
    print(f"Remaining clean images: {total_images - removed_images}")

if __name__ == "__main__":
    main()
