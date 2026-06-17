import urllib.request
import json

def list_files():
    # Try fetching the contents via GitHub API
    url = "https://api.github.com/repos/intel-iot-devkit/sample-videos/contents"
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            for item in data:
                print(item.get("name"), item.get("download_url"))
    except Exception as e:
        print("API failed:", e)

    # Let's also try fetching the page itself to see if we can parse it
    try:
        url_page = "https://github.com/intel-iot-devkit/sample-videos"
        req = urllib.request.Request(
            url_page,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            html = response.read().decode()
            print("HTML length:", len(html))
            if "car-detection" in html:
                print("Found car-detection in HTML page")
    except Exception as e:
        print("Page failed:", e)

if __name__ == "__main__":
    list_files()
