"""Download and verify the reproducible default parking-video artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request


SAMPLE_VIDEO_URL = (
    "https://raw.githubusercontent.com/intel-iot-devkit/"
    "sample-videos/master/car-detection.mp4"
)
SAMPLE_VIDEO_SHA256 = "fac033acb960b0a87e2a0e50b7532025d90c0acfe8af172b3d2b112107a4c1c5"


def default_sample_video_path() -> Path:
    return Path(__file__).resolve().parents[3] / "main" / "data" / "test" / "sample_parking.mp4"


def validate_sample_video(path: Path, expected_sha256: str = SAMPLE_VIDEO_SHA256) -> bool:
    """Return whether *path* exists and matches the expected SHA-256 digest."""
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def download_sample_video(
    destination: Path | None = None,
    *,
    url: str = SAMPLE_VIDEO_URL,
    expected_sha256: str = SAMPLE_VIDEO_SHA256,
) -> Path:
    """Download, verify, and atomically install the default demo video."""
    destination = destination or default_sample_video_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "DPL302m-demo-setup/1.0"})
    temp_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                while chunk := response.read(1024 * 1024):
                    temp_file.write(chunk)
        if not validate_sample_video(temp_path, expected_sha256):
            raise ValueError("Downloaded sample video checksum does not match the expected SHA-256.")
        temp_path.replace(destination)
        return destination
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify the local artifact without downloading")
    args = parser.parse_args()
    destination = default_sample_video_path()
    if args.verify:
        if validate_sample_video(destination):
            print(f"Verified sample video: {destination}")
            return 0
        print(f"Sample video missing or invalid: {destination}")
        return 1
    try:
        print(f"Downloading sample video from {SAMPLE_VIDEO_URL}...")
        print(f"Installed verified sample video: {download_sample_video(destination)}")
        return 0
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"Could not install sample video: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
