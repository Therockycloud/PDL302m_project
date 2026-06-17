"""Targeted top-up crawl for under-filled classes.

Crawls into a staging dir (``raw/_topup/<Class>``) so the raw download can be
semantic-cleaned + deduped before being merged into the canonical tree. Unlike
``crawl_vehicle_data.py`` (all colours, fixed keywords) this targets specific
classes and supports model-specific keyword sets — notably VinFast, whose
generic "VinFast car" query returns logos/interiors, so we query by model
(VF8, VF9, Lux, Fadil, President...) and keep the rear-view emphasis used by
the original colour crawl.
"""
import argparse
from pathlib import Path

from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

_MAIN = Path(__file__).resolve().parents[1]
STAGE = _MAIN / "data" / "raw" / "_topup"

COLOR_KW = {
    c: [
        f"{c} car rear view",
        f"{c} vehicle back view license plate",
        f"{c} sedan rear view",
        f"{c} SUV tail view",
        f"{c} car traffic from behind",
        f"{c} car driving away rear",
    ]
    for c in ["brown", "yellow", "red", "white", "black", "blue", "silver", "grey"]
}

# VinFast queried per model line (rear-facing, where plates live)
BRAND_KW = {
    "VinFast": [
        "VinFast VF8 rear view",
        "VinFast VF9 rear view",
        "VinFast VF5 rear",
        "VinFast Lux A2.0 rear",
        "VinFast Lux SA2.0 rear view",
        "VinFast Fadil rear",
        "VinFast VF e34 rear",
        "VinFast President SUV rear",
    ],
    **{
        b: [
            f"{b} car rear view",
            f"{b} sedan back license plate",
            f"{b} SUV tail view",
            f"{b} car driving away rear",
        ]
        for b in ["Ford", "Honda", "Hyundai", "Kia", "Mazda", "Mitsubishi", "Toyota"]
    },
}


def crawl_class(cls, keywords, max_num, engine):
    dest = STAGE / cls
    dest.mkdir(parents=True, exist_ok=True)
    Crawler = BingImageCrawler if engine == "bing" else GoogleImageCrawler
    for kw in keywords:
        print(f"  [{cls}] -> '{kw}'")
        try:
            crawler = Crawler(storage={"root_dir": str(dest)})
            crawler.crawl(keyword=kw, max_num=max_num)
        except Exception as e:
            print(f"    error: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["color", "brand"], required=True)
    ap.add_argument("--classes", nargs="+", required=True, help="class names to crawl")
    ap.add_argument("--max_num", type=int, default=40, help="images per keyword")
    ap.add_argument("--engine", choices=["google", "bing"], default="bing")
    args = ap.parse_args()

    table = COLOR_KW if args.kind == "color" else BRAND_KW
    for cls in args.classes:
        key = cls if cls in table else cls.lower()
        kws = table.get(key) or table.get(cls)
        if not kws:
            print(f"!! no keywords for {cls}, skipping")
            continue
        print(f"== crawling {cls} ({len(kws)} keywords x {args.max_num}) ==")
        crawl_class(cls, kws, args.max_num, args.engine)


if __name__ == "__main__":
    main()
