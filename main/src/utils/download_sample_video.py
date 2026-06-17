import os
import urllib.request
from urllib.error import HTTPError
from pathlib import Path

def download_video():
    # Attempt list of possible files
    urls = [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection-courtyard.mp4",
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
    ]
    
    dest_dir = Path(__file__).resolve().parents[3] / "main" / "data" / "test"
    dest_path = dest_dir / "sample_parking.mp4"

    os.makedirs(dest_dir, exist_ok=True)

    if dest_path.exists():
        print(f"File already exists at {dest_path}")
        return

    for url in urls:
        print(f"Trying to download {url}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            )
            with urllib.request.urlopen(req) as response:
                data = response.read()
                with open(dest_path, 'wb') as out_file:
                    out_file.write(data)
            print(f"Successfully downloaded from {url}")
            return
        except HTTPError as e:
            print(f"Failed with HTTP Error {e.code}: {e.reason}")
        except Exception as e:
            print(f"Failed with exception: {e}")
            
    print("All download attempts failed.")

if __name__ == "__main__":
    download_video()
