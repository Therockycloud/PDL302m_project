import hashlib

from src.utils import download_sample_video as sample_video


def test_validate_sample_video_rejects_missing_and_invalid_files(tmp_path):
    expected = hashlib.sha256(b"expected-video").hexdigest()
    missing = tmp_path / "missing.mp4"
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"wrong-video")

    assert sample_video.validate_sample_video(missing, expected) is False
    assert sample_video.validate_sample_video(invalid, expected) is False


def test_download_sample_video_rejects_wrong_checksum(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __init__(self):
            self.sent = False

        def read(self, _size=-1):
            if self.sent:
                return b""
            self.sent = True
            return b"wrong-video"

    monkeypatch.setattr(sample_video.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    destination = tmp_path / "sample_parking.mp4"

    try:
        sample_video.download_sample_video(destination, expected_sha256="0" * 64)
    except ValueError as exc:
        assert "checksum" in str(exc).lower()
    else:
        raise AssertionError("download with a bad checksum must fail")

    assert not destination.exists()
