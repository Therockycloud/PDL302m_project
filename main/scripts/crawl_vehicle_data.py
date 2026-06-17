import os
import argparse
from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler

def crawl_images(keyword, max_num, save_dir, crawler_type='google'):
    """
    Crawls images using Google or Bing crawler.
    """
    os.makedirs(save_dir, exist_ok=True)
    if crawler_type == 'google':
        crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
    elif crawler_type == 'bing':
        crawler = BingImageCrawler(storage={'root_dir': save_dir})
    else:
        raise ValueError("Unsupported crawler_type. Use 'google' or 'bing'.")

    # Filters can be added here if needed, but for rear view, keyword is usually enough.
    crawler.crawl(keyword=keyword, max_num=max_num)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, '../data/raw/colors')
    
    parser = argparse.ArgumentParser(description="Crawl vehicle images for color classification.")
    parser.add_argument('--max_num', type=int, default=10, help='Max number of images per color.')
    parser.add_argument('--crawler', type=str, default='google', choices=['google', 'bing'], help='Crawler engine to use.')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Base directory to save images.')
    args = parser.parse_args()

    # Define common car colors to crawl
    colors = ['white', 'black', 'red', 'blue', 'silver', 'grey', 'yellow', 'green', 'brown']
    
    # Ensure base directory exists
    base_dir = os.path.abspath(args.data_dir)
    print(f"Saving images to: {base_dir}")

    for color in colors:
        print(f"\n--- Crawling {color} cars ---")
        save_dir = os.path.join(base_dir, color)
        
        # Sử dụng nhiều từ khóa khác nhau để vượt qua giới hạn ~70 ảnh của công cụ tìm kiếm
        keywords = [
            f"{color} car rear view",
            f"{color} vehicle back view license plate",
            f"{color} sedan rear view",
            f"{color} SUV tail view",
            f"{color} car traffic from behind",
            f"{color} car driving away rear"
        ]
        
        for keyword in keywords:
            print(f"  -> Crawling keyword: '{keyword}'")
            try:
                # Crawl cho mỗi keyword
                crawl_images(keyword, max_num=args.max_num, save_dir=save_dir, crawler_type=args.crawler)
            except Exception as e:
                print(f"Error crawling keyword '{keyword}': {e}")

if __name__ == '__main__':
    main()
