"""Consolidate the parallel crawl trees into the single canonical dataset.

Merges ``raw/colors`` -> ``raw/car_colors`` and ``raw/brands`` -> ``raw/car_brands``,
prefixing copied files so they never collide with existing ``img_N`` / ``000001``
names. The ``green`` colour folder has no matching model class (8-class set with
no green) so it is moved aside to ``raw/_quarantine_green`` instead of merged.

Idempotent-ish: copies are skipped if an identically named target already exists.
"""
import argparse
import shutil
from pathlib import Path

_MAIN = Path(__file__).resolve().parents[1]
RAW = _MAIN / "data" / "raw"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# lowercase crawl folder -> canonical capitalised class folder
COLOR_MAP = {
    "black": "Black", "blue": "Blue", "brown": "Brown", "grey": "Grey",
    "red": "Red", "silver": "Silver", "white": "White", "yellow": "Yellow",
}
# 'green' intentionally absent -> quarantined


def _merge(src_root: Path, dst_root: Path, name_map, prefix: str, quarantine: Path):
    copied = skipped = quarantined = 0
    for sub in sorted(src_root.iterdir()):
        if not sub.is_dir():
            continue
        cls = sub.name
        mapped = name_map.get(cls, cls) if name_map else cls
        if name_map and cls not in name_map:
            # unknown class (e.g. green) -> quarantine, do not merge
            qdst = quarantine / cls
            qdst.mkdir(parents=True, exist_ok=True)
            for f in sub.iterdir():
                if f.suffix.lower() in EXTS:
                    shutil.move(str(f), str(qdst / f.name))
                    quarantined += 1
            continue
        dst = dst_root / mapped
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            target = dst / f"{prefix}{f.name}"
            if target.exists():
                skipped += 1
                continue
            shutil.copy2(str(f), str(target))
            copied += 1
    return copied, skipped, quarantined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually perform the merge")
    args = ap.parse_args()
    if not args.apply:
        print("DRY RUN (pass --apply to execute). Planned merges:")
        print(f"  {RAW/'colors'}  -> {RAW/'car_colors'}  (green -> quarantine)")
        print(f"  {RAW/'brands'}  -> {RAW/'car_brands'}")
        return

    quarantine = RAW / "_quarantine_green"
    c1 = _merge(RAW / "colors", RAW / "car_colors", COLOR_MAP, "colors_", quarantine)
    print(f"[colors]  copied={c1[0]} skipped={c1[1]} green_quarantined={c1[2]}")
    c2 = _merge(RAW / "brands", RAW / "car_brands", None, "brands_", quarantine)
    print(f"[brands]  copied={c2[0]} skipped={c2[1]}")


if __name__ == "__main__":
    main()
