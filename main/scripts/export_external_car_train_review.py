#!/usr/bin/env python3
"""Export reserved-disjoint review crops from external car plate corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_MAIN = Path(__file__).resolve().parents[1]
if str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))

from scripts.export_real_car_train_review import run_pipeline  # noqa: E402

DEFAULT_SOURCES = (
    _MAIN / "data/raw/license_plates_kaggle/train",
    _MAIN / "data/raw/license_plates_kaggle/mrzaizai2k/train",
)
DEFAULT_OUTPUT = _MAIN / "data/plate_ocr/review/external_kaggle_car_audit"
PROVENANCE = _MAIN / "data/raw/license_plates_kaggle/PROVENANCE.md"


def _resolve_provenance(source_paths: list[Path], explicit: Path | None = None) -> Path | None:
    if explicit is not None and explicit.is_file():
        return explicit.resolve()
    for source in source_paths:
        for parent in [source, *source.resolve().parents]:
            candidate = parent / "PROVENANCE.md"
            if candidate.is_file():
                return candidate.resolve()
            if parent == _MAIN:
                break
    return PROVENANCE.resolve() if PROVENANCE.is_file() else None


def run_external_review(
    *,
    sources: list[Path] | None = None,
    output_dir: Path = DEFAULT_OUTPUT,
    progress_every: int = 200,
    provenance: Path | None = None,
) -> dict[str, object]:
    source_paths = [path for path in (sources or list(DEFAULT_SOURCES)) if path.is_dir()]
    if not source_paths:
        raise FileNotFoundError(
            "no external corpus found; run download_external_car_plates.py first"
        )
    result = run_pipeline(
        sources=source_paths,
        output_dir=output_dir,
        exclude_source_patterns=(),
        progress_every=progress_every,
    )
    stats_path = Path(result["stats_path"])
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    resolved = _resolve_provenance(source_paths, provenance)
    stats["provenance"] = str(resolved) if resolved is not None else None
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["provenance"] = stats["provenance"]
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--provenance", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_external_review(
        sources=args.source,
        output_dir=args.output,
        progress_every=args.progress_every,
        provenance=args.provenance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
