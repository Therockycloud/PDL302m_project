import os
import argparse
import yaml
from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler

def crawl_images(keyword, max_num, save_dir, crawler_type='google'):
    os.makedirs(save_dir, exist_ok=True)
    if crawler_type == 'google':
        crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
    elif crawler_type == 'bing':
        crawler = BingImageCrawler(storage={'root_dir': save_dir})
    else:
        raise ValueError("Unsupported crawler_type. Use 'google' or 'bing'.")

    crawler.crawl(keyword=keyword, max_num=max_num)

def load_brands_from_config(config_path):
    brands = ['Toyota', 'Hyundai', 'Kia', 'Mazda', 'Honda', 'VinFast', 'Ford', 'Mitsubishi']
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                config_brands = config.get('brand_classifier', {}).get('classes')
                if config_brands:
                    brands = config_brands
    except Exception as e:
        print(f"Could not load config from {config_path}. Using default brands. Error: {e}")
    return brands

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, '../data/raw/brands')
    default_config_path = os.path.join(script_dir, '../configs/config.yaml')
    
    parser = argparse.ArgumentParser(description="Crawl vehicle images for brand classification.")
    parser.add_argument('--max_num', type=int, default=10, help='Max number of images per keyword.')
    parser.add_argument('--crawler', type=str, default='bing', choices=['google', 'bing'], help='Crawler engine to use.')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Base directory to save images.')
    parser.add_argument('--config', type=str, default=default_config_path, help='Path to config.yaml')
    args = parser.parse_args()

    brands = load_brands_from_config(args.config)
    
    base_dir = os.path.abspath(args.data_dir)
    print(f"Saving images to: {base_dir}")
    print(f"Target Brands: {brands}")

    for brand in brands:
        print(f"\n--- Crawling {brand} cars ---")
        save_dir = os.path.join(base_dir, brand)
        
        keywords = [
            f"{brand} car rear view",
            f"{brand} vehicle back view license plate",
            f"{brand} sedan rear view",
            f"{brand} SUV tail view",
            f"{brand} car traffic from behind",
            f"{brand} car driving away rear"
        ]
        
        for keyword in keywords:
            print(f"  -> Crawling keyword: '{keyword}'")
            try:
                crawl_images(keyword, max_num=args.max_num, save_dir=save_dir, crawler_type=args.crawler)
            except Exception as e:
                print(f"Error crawling keyword '{keyword}': {e}")

if __name__ == '__main__':
    main()
