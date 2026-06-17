"""Normalize every image to clean RGB JPEG so TF's decode_image can read it.

``clean_corrupted_images.py`` uses cv2 (which happily decodes WEBP and some
malformed files), but ``tf.keras.utils.image_dataset_from_directory`` rejects
anything that isn't real JPEG/PNG/GIF/BMP — crawled files often have a .jpg
name but WEBP/HTML content. This pass re-encodes each image via Pillow to
baseline RGB JPEG (converting WEBP/PNG/etc.) and deletes anything Pillow cannot
open. Run BEFORE split_dataset.py.
"""
import argparse
from pathlib import Path
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def normalize(root: Path):
    converted = removed = kept = 0
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in list(cls_dir.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    target = f.with_suffix(".jpg")
                    # load into memory before potentially unlinking source
                    im.load()
                tmp = target.with_name(target.stem + "__norm.jpg")
                im.save(tmp, "JPEG", quality=92)
                if f.exists() and f != target:
                    f.unlink()
                tmp.replace(target)
                if f.suffix.lower() != ".jpg":
                    converted += 1
                kept += 1
            except Exception as e:
                try:
                    f.unlink()
                    removed += 1
                    print(f"  removed unreadable: {f.relative_to(root)} ({e})")
                except OSError:
                    pass
    print(f"[{root.name}] kept={kept} converted_to_jpg={converted} removed={removed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()
    normalize(Path(args.data_dir))


if __name__ == "__main__":
    main()
