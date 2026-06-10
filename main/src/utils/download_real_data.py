import os
import urllib.request
import urllib.parse
import json
import logging
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


def search_wikimedia_images(query, limit=15):
    """Searches Wikimedia Commons and returns a list of image/thumbnail URLs in one batch."""
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&srnamespace=6&srlimit={limit}&format=json"
    headers = {"User-Agent": USER_AGENT}
    
    try:
        # Step 1: Search for files
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as res:
            data = json.loads(res.read().decode())
            search_results = data.get("query", {}).get("search", [])
            
        titles = [item["title"] for item in search_results if item.get("title")]
        if not titles:
            return []
            
        # Step 2: Query imageinfo in batch to save requests and avoid throttling
        titles_str = "|".join(titles)
        encoded_titles = urllib.parse.quote(titles_str)
        info_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={encoded_titles}&prop=imageinfo&iiprop=url&iiurlwidth=300&format=json"
        
        # Rate limiting sleep before query
        time.sleep(1.5)
        
        info_req = urllib.request.Request(info_url, headers=headers)
        with urllib.request.urlopen(info_req, timeout=20) as info_res:
            info_data = json.loads(info_res.read().decode())
            pages = info_data.get("query", {}).get("pages", {})
            
        urls = []
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


def clean_and_save_image(temp_path, dest_path):
    """Loads image, converts to RGB, resizes/saves as clean JPEG."""
    try:
        with Image.open(temp_path) as img:
            img = img.convert("RGB")
            # Downsample to 224x224 to match model inputs
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img.save(dest_path, "JPEG")
        return True
    except Exception as e:
        logger.error(f"Failed to process image {temp_path}: {e}")
        return False


def download_brands():
    logger.info("Downloading car brand images...")
    for brand in BRANDS:
        logger.info(f"Searching and downloading images for brand: {brand}")
        query = f"{brand} car"
        urls = search_wikimedia_images(query, limit=15)
        
        count = 0
        for url in urls:
            temp_file = "temp_img.dat"
            # Sleep slightly between image downloads to respect rate limit
            time.sleep(0.5)
            if download_url(url, temp_file):
                dest_file = os.path.join(BRAND_DIR, brand, f"img_{count}.jpg")
                if clean_and_save_image(temp_file, dest_file):
                    count += 1
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if count >= 15:
                break
        logger.info(f"Successfully downloaded {count} images for brand: {brand}")
        time.sleep(1.0)  # Sleep between classes


def download_colors():
    logger.info("Downloading car color images...")
    for color in COLORS:
        logger.info(f"Searching and downloading images for color: {color}")
        query = f"{color} car front"
        urls = search_wikimedia_images(query, limit=15)
        
        count = 0
        for url in urls:
            temp_file = "temp_img.dat"
            # Sleep slightly between image downloads to respect rate limit
            time.sleep(0.5)
            if download_url(url, temp_file):
                dest_file = os.path.join(COLOR_DIR, color, f"img_{count}.jpg")
                if clean_and_save_image(temp_file, dest_file):
                    count += 1
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if count >= 15:
                break
        logger.info(f"Successfully downloaded {count} images for color: {color}")
        time.sleep(1.0)  # Sleep between classes


def download_license_plates():
    logger.info("Downloading Vietnamese license plate images and annotations...")
    repo_base_url = "https://raw.githubusercontent.com/mrzaizai2k/License-Plate-Recognition-YOLOv7-and-CNN/master/data/test"
    
    count = 0
    # Download 5 images from clip3_new
    for i in range(5):
        img_name = f"clip3_new_{i}"
        img_url = f"{repo_base_url}/images/{img_name}.jpg"
        txt_url = f"{repo_base_url}/labels/{img_name}.txt"
        
        img_dest = os.path.join(PLATE_DIR, f"{img_name}.jpg")
        txt_dest = os.path.join(PLATE_DIR, f"{img_name}.txt")
        
        # Download image and label
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


if __name__ == "__main__":
    download_brands()
    download_colors()
    download_license_plates()
    logger.info("Data download and cleanup completed successfully.")
