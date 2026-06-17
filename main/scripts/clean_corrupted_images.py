import os
import cv2
import argparse
from pathlib import Path

def clean_corrupted(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory {data_dir} does not exist.")
        return

    removed_count = 0
    total_count = 0

    print(f"Scanning directory: {data_path} for corrupted images...")
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG']:
        for img_path in data_path.rglob(ext):
            total_count += 1
            try:
                # Try reading the image
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    os.remove(img_path)
                    removed_count += 1
                    print(f"Removed corrupted/unreadable file: {img_path}")
            except Exception as e:
                # If any exception occurs during reading
                try:
                    os.remove(img_path)
                    removed_count += 1
                    print(f"Removed file due to error: {img_path} | Error: {e}")
                except OSError:
                    pass

    print(f"\nCleanup complete. Scanned {total_count} images.")
    print(f"Removed {removed_count} corrupted images.")
    print(f"Remaining valid images: {total_count - removed_count}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, '../data/raw/colors')
    
    parser = argparse.ArgumentParser(description="Remove corrupted images from a dataset directory.")
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Directory containing the images to check.')
    
    args = parser.parse_args()
    
    clean_corrupted(os.path.abspath(args.data_dir))

if __name__ == '__main__':
    main()
