"""Cap each class to a target size for a balanced 'few but quality' set.

Overflow images are MOVED (not deleted) to ``raw/_capped_overflow/<tree>/<class>``
so nothing is lost and the cap is reversible. Files whose name starts with any
``--priority`` prefix are kept first (used to retain the model-specific VinFast
crawl over the generic one).
"""
import argparse
import shutil
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def cap(tree: str, target: int, priority: list[str]):
    root = RAW / tree
    overflow_root = RAW / "_capped_overflow" / tree
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in EXTS]
        # priority files first (preserve), then the rest, both name-sorted
        def rank(f):
            for i, p in enumerate(priority):
                if f.name.startswith(p):
                    return (0, i, f.name)
            return (1, 0, f.name)
        files.sort(key=rank)
        keep, drop = files[:target], files[target:]
        if drop:
            odst = overflow_root / cls_dir.name
            odst.mkdir(parents=True, exist_ok=True)
            for f in drop:
                shutil.move(str(f), str(odst / f.name))
        print(f"  {cls_dir.name:12s} kept={len(keep):3d} overflow={len(drop):3d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="e.g. car_colors or car_brands")
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--priority", nargs="*", default=[], help="filename prefixes to keep first")
    args = ap.parse_args()
    print(f"== capping {args.tree} to {args.target} ==")
    cap(args.tree, args.target, args.priority)


if __name__ == "__main__":
    main()
