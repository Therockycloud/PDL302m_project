import os
import sys
import logging
import io
import urllib.request
import urllib.parse
import time
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base directories
BASE_DATA_DIR = os.path.join("main", "data", "raw")
BRAND_DIR = os.path.join(BASE_DATA_DIR, "car_brands")
COLOR_DIR = os.path.join(BASE_DATA_DIR, "car_colors")
PLATE_DIR = os.path.join(BASE_DATA_DIR, "license_plates")

BRANDS = ["Toyota", "Hyundai", "Kia", "Mazda", "Honda", "VinFast", "Ford", "Mitsubishi"]
COLORS = ["White", "Black", "Grey", "Silver", "Red", "Blue", "Brown", "Yellow"]
USER_AGENT = "DPL302mProjectVehicleClassifier/1.0 (https://github.com/konalyn/PDL302m_project; konalyn@gmail.com)"

# Ensure directories exist
for brand in BRANDS:
    os.makedirs(os.path.join(BRAND_DIR, brand), exist_ok=True)
for color in COLORS:
    os.makedirs(os.path.join(COLOR_DIR, color), exist_ok=True)
os.makedirs(PLATE_DIR, exist_ok=True)


def download_url(url, dest_path):
    """Downloads a file from a URL to a destination path."""
    headers = {"User-Agent": USER_AGENT}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            with open(dest_path, "wb") as f:
                f.write(response.read())
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def clean_and_save_image(img_obj, dest_path):
    """Converts a PIL image to RGB, resizes to 224x224, and saves as clean JPEG."""
    try:
        if isinstance(img_obj, bytes):
            img = Image.open(io.BytesIO(img_obj))
        else:
            img = img_obj
            
        with img:
            img = img.convert("RGB")
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img.save(dest_path, "JPEG")
        return True
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        return False


def search_wikimedia_images(query, limit=100):
    """Searches Wikimedia Commons and returns a list of image/thumbnail URLs in batch."""
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&srnamespace=6&srlimit={limit}&format=json"
    headers = {"User-Agent": USER_AGENT}
    
    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as res:
            data = json.loads(res.read().decode())
            search_results = data.get("query", {}).get("search", [])
            
        titles = [item["title"] for item in search_results if item.get("title")]
        if not titles:
            return []
            
        # Batch info query
        urls = []
        # Wikimedia limit for titles parameter is 50, so chunk the query
        for i in range(0, len(titles), 50):
            chunk = titles[i:i+50]
            titles_str = "|".join(chunk)
            encoded_titles = urllib.parse.quote(titles_str)
            info_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={encoded_titles}&prop=imageinfo&iiprop=url&iiurlwidth=300&format=json"
            
            time.sleep(1.0) # rate limit sleep
            info_req = urllib.request.Request(info_url, headers=headers)
            with urllib.request.urlopen(info_req, timeout=20) as info_res:
                info_data = json.loads(info_res.read().decode())
                pages = info_data.get("query", {}).get("pages", {})
                
            for page_id, page in pages.items():
                if "imageinfo" in page:
                    imageinfo = page["imageinfo"][0]
                    img_url = imageinfo.get("thumburl") or imageinfo.get("url")
                    if img_url:
                        urls.append(img_url)
        return urls
    except Exception as e:
        logger.error(f"Error searching Wikimedia for {query}: {e}")
        return []


def download_brands_from_hf(limit_per_class=200):
    logger.info("Loading full Stanford Cars dataset from Hugging Face...")
    try:
        from datasets import load_dataset
        ds = load_dataset("tanganke/stanford_cars", split="train")
        class_names = ds.features["label"].names
    except Exception as e:
        logger.error(f"Failed to load datasets library or Stanford Cars: {e}")
        return False
        
    logger.info("Processing brand images and mapping classes...")
    brand_counts = {b: 0 for b in BRANDS}
    
    for item in ds:
        label_idx = item["label"]
        class_name = class_names[label_idx]
        
        # Check if the class matches any of our target brands
        matched_brand = None
        for b in BRANDS:
            if b != "VinFast" and class_name.lower().startswith(b.lower()):
                matched_brand = b
                break
                
        if matched_brand and brand_counts[matched_brand] < limit_per_class:
            img_obj = item["image"]
            dest_file = os.path.join(BRAND_DIR, matched_brand, f"img_{brand_counts[matched_brand]}.jpg")
            if clean_and_save_image(img_obj, dest_file):
                brand_counts[matched_brand] += 1
                
        # Check if we have met the limit for all target brands (excluding VinFast)
        all_done = True
        for b in BRANDS:
            if b != "VinFast" and brand_counts[b] < limit_per_class:
                all_done = False
                break
        if all_done:
            break
            
    for b in BRANDS:
        logger.info(f"Brand {b}: loaded {brand_counts[b]} images.")
    return True


def download_vinfast_images(limit=100):
    logger.info("Downloading custom VinFast images from Wikimedia...")
    global json
    import json
    
    query = "VinFast"
    urls = search_wikimedia_images(query, limit=limit)
    
    count = 0
    for url in urls:
        temp_file = "temp_vinfast.dat"
        time.sleep(0.5)
        if download_url(url, temp_file):
            dest_file = os.path.join(BRAND_DIR, "VinFast", f"img_{count}.jpg")
            try:
                with Image.open(temp_file) as img:
                    if clean_and_save_image(img, dest_file):
                        count += 1
            except Exception as e:
                logger.error(f"Failed to open/process downloaded image {url}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if count >= limit:
            break
    logger.info(f"VinFast: loaded {count} images.")


def download_colors_from_hf(limit_per_class=200):
    logger.info("Loading Car Color dataset from Hugging Face...")
    try:
        from datasets import load_dataset
        ds = load_dataset("duyan2803/vqa-cars-balanced-color", split="train")
    except Exception as e:
        logger.error(f"Failed to load Car Color dataset from HF: {e}")
        return False
        
    logger.info("Processing color images and mapping classes...")
    color_counts = {c: 0 for c in COLORS}
    
    for item in ds:
        color_str = item.get("Exterior color", "")
        if not color_str:
            continue
            
        color_str_lower = color_str.lower()
        matched_color = None
        
        # Substring mapping
        if "white" in color_str_lower:
            matched_color = "White"
        elif "black" in color_str_lower:
            matched_color = "Black"
        elif "grey" in color_str_lower or "gray" in color_str_lower or "nardo" in color_str_lower or "gry" in color_str_lower:
            matched_color = "Grey"
        elif "silver" in color_str_lower:
            matched_color = "Silver"
        elif "red" in color_str_lower:
            matched_color = "Red"
        elif "blue" in color_str_lower:
            matched_color = "Blue"
        elif "brown" in color_str_lower or "bronze" in color_str_lower:
            matched_color = "Brown"
        elif "yellow" in color_str_lower or "gold" in color_str_lower:
            matched_color = "Yellow"
            
        if matched_color and color_counts[matched_color] < limit_per_class:
            img_bytes = item.get("image_list")
            if img_bytes:
                dest_file = os.path.join(COLOR_DIR, matched_color, f"img_{color_counts[matched_color]}.jpg")
                if clean_and_save_image(img_bytes, dest_file):
                    color_counts[matched_color] += 1
                    
        all_done = True
        for c in COLORS:
            if color_counts[c] < limit_per_class:
                all_done = False
                break
        if all_done:
            break
            
    for c in COLORS:
        logger.info(f"Color {c}: loaded {color_counts[c]} images.")
    return True


def download_license_plates():
    logger.info("Downloading Vietnamese license plate images and annotations...")
    repo_base_url = "https://raw.githubusercontent.com/mrzaizai2k/License-Plate-Recognition-YOLOv7-and-CNN/master/data/test"
    
    count = 0
    for i in range(5):
        img_name = f"clip3_new_{i}"
        img_url = f"{repo_base_url}/images/{img_name}.jpg"
        txt_url = f"{repo_base_url}/labels/{img_name}.txt"
        
        img_dest = os.path.join(PLATE_DIR, f"{img_name}.jpg")
        txt_dest = os.path.join(PLATE_DIR, f"{img_name}.txt")
        
        logger.info(f"Downloading plate {img_name}...")
        time.sleep(0.5)
        if download_url(img_url, img_dest) and download_url(txt_url, txt_dest):
            try:
                with Image.open(img_dest) as img:
                    img = img.convert("RGB")
                    img.save(img_dest, "JPEG")
                count += 1
            except Exception as e:
                logger.error(f"Failed to process plate image {img_dest}: {e}")
                if os.path.exists(img_dest): os.remove(img_dest)
                if os.path.exists(txt_dest): os.remove(txt_dest)
    logger.info(f"Successfully downloaded {count} license plate images and annotations.")


def main():
    logger.info("Starting dataset setup...")
    
    # Try downloading using datasets library (Hugging Face)
    brand_ok = download_brands_from_hf(limit_per_class=200)
    download_vinfast_images(limit=100)
    color_ok = download_colors_from_hf(limit_per_class=200)
    download_license_plates()
    
    if brand_ok and color_ok:
        logger.info("All datasets successfully loaded, preprocessed and structured!")
    else:
        logger.warning("Some datasets failed to download. Check logs.")


if __name__ == "__main__":
    main()
