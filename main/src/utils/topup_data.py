import os
import sys
import logging
import io
import json
import urllib.request
import urllib.parse
import time
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DATA_DIR = os.path.join("main", "data", "raw")
BRAND_DIR = os.path.join(BASE_DATA_DIR, "car_brands")
USER_AGENT = "DPL302mProjectVehicleClassifier/1.0 (https://github.com/konalyn/PDL302m_project; konalyn.study@gmail.com)"

# Ensure directories exist
for brand in ["Kia", "Mazda", "Mitsubishi", "VinFast"]:
    os.makedirs(os.path.join(BRAND_DIR, brand), exist_ok=True)

def download_url(url, dest_path):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(dest_path, "wb") as f:
                f.write(response.read())
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def clean_and_save_image(img_obj, dest_path):
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
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&srnamespace=6&srlimit={limit}&format=json"
    headers = {"User-Agent": USER_AGENT}
    
    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode())
            search_results = data.get("query", {}).get("search", [])
            
        titles = [item["title"] for item in search_results if item.get("title")]
        if not titles:
            return []
            
        urls = []
        for i in range(0, len(titles), 50):
            chunk = titles[i:i+50]
            titles_str = "|".join(chunk)
            encoded_titles = urllib.parse.quote(titles_str)
            info_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={encoded_titles}&prop=imageinfo&iiprop=url&iiurlwidth=300&format=json"
            
            time.sleep(1.5) # Polite sleep
            info_req = urllib.request.Request(info_url, headers=headers)
            with urllib.request.urlopen(info_req, timeout=15) as info_res:
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

def topup_brand(brand, query, target_count=150):
    dest_dir = os.path.join(BRAND_DIR, brand)
    existing_files = [f for f in os.listdir(dest_dir) if f.endswith(".jpg")]
    current_count = len(existing_files)
    
    logger.info(f"Topping up brand {brand}. Current images: {current_count}. Target: {target_count}")
    if current_count >= target_count:
        logger.info(f"Target count already met for {brand}.")
        return
        
    needed = target_count - current_count
    urls = search_wikimedia_images(query, limit=needed * 2)
    logger.info(f"Found {len(urls)} candidates on Wikimedia for query: {query}")
    
    downloaded = 0
    for url in urls:
        temp_file = f"temp_{brand}.dat"
        time.sleep(1.5) # Polite download interval
        if download_url(url, temp_file):
            dest_file = os.path.join(dest_dir, f"img_wikimedia_{current_count + downloaded}.jpg")
            try:
                with Image.open(temp_file) as img:
                    if clean_and_save_image(img, dest_file):
                        downloaded += 1
                        logger.info(f"[{brand}] Saved image {downloaded}/{needed}: {dest_file}")
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if downloaded >= needed:
            break
            
    logger.info(f"Finished top-up for {brand}. Added {downloaded} images.")

def main():
    logger.info("Starting brand dataset top-up from Wikimedia Commons...")
    topup_brand("Kia", "Kia car", target_count=120)
    topup_brand("Mazda", "Mazda car", target_count=120)
    topup_brand("Mitsubishi", "Mitsubishi car", target_count=120)
    topup_brand("VinFast", "VinFast car", target_count=120)
    logger.info("Top-up process finished!")

if __name__ == "__main__":
    main()
