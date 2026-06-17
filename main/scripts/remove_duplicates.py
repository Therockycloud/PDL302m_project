import os
import argparse
from pathlib import Path
from PIL import Image
import imagehash

def remove_duplicates(data_dir, hash_size=8, threshold=5):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory {data_dir} does not exist.")
        return

    total_removed = 0
    total_scanned = 0
    
    for sub_dir in data_path.iterdir():
        if sub_dir.is_dir():
            print(f"\nScanning {sub_dir.name} for duplicates...")
            hashes = {}
            removed = 0
            scanned = 0
            
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in sub_dir.rglob(ext):
                    scanned += 1
                    try:
                        img = Image.open(str(img_path))
                        # Using phash for perceptual hashing
                        h = imagehash.phash(img, hash_size=hash_size)
                        
                        # Check against existing hashes
                        is_duplicate = False
                        for existing_hash, existing_path in hashes.items():
                            if h - existing_hash <= threshold:
                                is_duplicate = True
                                break
                        
                        if is_duplicate:
                            os.remove(img_path)
                            removed += 1
                        else:
                            hashes[h] = img_path
                    except Exception as e:
                        pass # Ignore if we can't open it; clean_corrupted_images.py should have handled it
            
            print(f"  Scanned: {scanned}")
            print(f"  Removed duplicates: {removed}")
            print(f"  Remaining: {scanned - removed}")
            total_removed += removed
            total_scanned += scanned

    print(f"\n--- Deduplication Summary ---")
    print(f"Total scanned: {total_scanned}")
    print(f"Total removed: {total_removed}")
    print(f"Total remaining: {total_scanned - total_removed}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, '../data/raw/colors')
    
    parser = argparse.ArgumentParser(description="Remove duplicate images using perceptual hashing.")
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Directory containing the images to check.')
    parser.add_argument('--hash_size', type=int, default=8, help='Hash size for phash (default: 8).')
    parser.add_argument('--threshold', type=int, default=5, help='Hamming distance threshold for duplicates (default: 5).')
    
    args = parser.parse_args()
    
    remove_duplicates(os.path.abspath(args.data_dir), args.hash_size, args.threshold)

if __name__ == '__main__':
    main()
