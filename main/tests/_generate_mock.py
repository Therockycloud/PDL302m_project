"""One-shot script to generate mock car brand images for dataset tests."""
import os
import sys

def main():
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed", file=sys.stderr)
        sys.exit(1)

    brands = [
        "Toyota", "Hyundai", "Kia", "Mazda",
        "Honda", "VinFast", "Ford", "Mitsubishi",
    ]
    colors = [
        (200, 50, 50), (50, 100, 200), (50, 200, 100), (100, 100, 100),
        (150, 50, 150), (220, 220, 220), (30, 30, 30), (255, 128, 0),
    ]
    base = os.path.join("main", "data", "raw", "car_brands")

    for brand, c in zip(brands, colors):
        d = os.path.join(base, brand)
        os.makedirs(d, exist_ok=True)
        for i in range(5):
            path = os.path.join(d, f"sample_{i}.jpg")
            Image.new("RGB", (224, 224), c).save(path)

    count = sum(len(f) for _, _, f in os.walk(base))
    print(f"Generated {count} images in {base}")


if __name__ == "__main__":
    main()
