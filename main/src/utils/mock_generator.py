import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

def generate_solid_image(color_rgb, text, size=(224, 224)):
    """Generates a solid color PIL image with text overlay."""
    img = Image.new("RGB", size, color_rgb)
    draw = ImageDraw.Draw(img)
    # Simple bounding box / text positioning
    draw.text((10, 100), text, fill=(255, 255, 255) if sum(color_rgb) < 380 else (0, 0, 0))
    return img

def create_mock_data():
    raw_dir = "main/data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Generate Car Brands mock data (2 images per brand)
    brands = ["Toyota", "Hyundai", "Kia", "Mazda", "Honda", "VinFast", "Ford", "Mitsubishi"]
    brand_colors = [
        (200, 50, 50),   # Toyota Redish
        (50, 100, 200),  # Hyundai Blueish
        (50, 200, 100),  # Kia Greenish
        (100, 100, 100), # Mazda Grey
        (150, 50, 150),  # Honda Purpleish
        (220, 220, 220), # VinFast Silverish
        (30, 30, 30),    # Ford Dark
        (255, 128, 0)    # Mitsubishi Orange
    ]
    
    print("Generating mock car brand images...")
    for brand, color in zip(brands, brand_colors):
        brand_dir = os.path.join(raw_dir, "car_brands", brand)
        os.makedirs(brand_dir, exist_ok=True)
        for i in range(5):  # 5 samples per class
            img = generate_solid_image(color, f"{brand} Car {i}")
            img.save(os.path.join(brand_dir, f"sample_{i}.jpg"))

    # 2. Generate Car Colors mock data (2 images per color)
    colors = ["White", "Black", "Grey", "Silver", "Red", "Blue", "Brown", "Yellow"]
    color_rgbs = [
        (255, 255, 255), # White
        (0, 0, 0),       # Black
        (128, 128, 128), # Grey
        (192, 192, 192), # Silver
        (255, 0, 0),     # Red
        (0, 0, 255),     # Blue
        (139, 69, 19),   # Brown
        (255, 255, 0)    # Yellow
    ]
    
    print("Generating mock car color images...")
    for color, rgb in zip(colors, color_rgbs):
        color_dir = os.path.join(raw_dir, "car_colors", color)
        os.makedirs(color_dir, exist_ok=True)
        for i in range(5):  # 5 samples per class
            img = generate_solid_image(rgb, f"Car Color: {color} {i}")
            img.save(os.path.join(color_dir, f"sample_{i}.jpg"))

    # 3. Generate License Plates mock data (3 images with YOLO labels)
    print("Generating mock license plate images with YOLO annotations...")
    plates_dir = os.path.join(raw_dir, "license_plates")
    os.makedirs(plates_dir, exist_ok=True)

    sample_plates = [
        ("30F12345", (255, 255, 255)),
        ("51G67890", (255, 255, 255)),
        ("43A11111", (255, 255, 255))
    ]

    for idx, (plate_text, bg_color) in enumerate(sample_plates):
        # Create a "car" image of size 640x640
        car_img = Image.new("RGB", (640, 640), (100, 110, 120))
        draw = ImageDraw.Draw(car_img)
        
        # Draw a white license plate region (bounding box coordinates: [x1, y1, x2, y2])
        # Center = (320, 400), Width = 200, Height = 60
        plate_box = [220, 370, 420, 430]
        draw.rectangle(plate_box, fill=(240, 240, 240), outline=(0, 0, 0), width=3)
        draw.text((250, 390), plate_text, fill=(0, 0, 0))
        
        # Save image
        img_name = f"car_plate_{idx}"
        car_img.save(os.path.join(plates_dir, f"{img_name}.jpg"))
        
        # Save YOLO annotation: class_id x_center y_center width height (normalized)
        # Class 0: license_plate
        x_center = (plate_box[0] + plate_box[2]) / 2.0 / 640.0
        y_center = (plate_box[1] + plate_box[3]) / 2.0 / 640.0
        width = (plate_box[2] - plate_box[0]) / 640.0
        height = (plate_box[3] - plate_box[1]) / 640.0
        
        anno_text = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        with open(os.path.join(plates_dir, f"{img_name}.txt"), "w") as f:
            f.write(anno_text)

    print("Mock datasets created successfully!")

if __name__ == "__main__":
    create_mock_data()
