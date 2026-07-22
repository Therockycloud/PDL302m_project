import pytest

try:
    import easyocr
except ImportError:
    easyocr = None

@pytest.fixture(scope="session", autouse=True)
def initialize_libraries():
    """Pre-initialize EasyOCR to prevent threading/OpenMP conflicts with TensorFlow.

    PyTorch/EasyOCR must be initialized before TensorFlow loads its datasets to avoid
    segmentation faults (exit code 139) inside Docker containers.

    EasyOCR is a train/eval-only dependency in the runtime image; if it isn't
    installed, skip this legacy guard cleanly.
    """
    if easyocr is None:
        return
    try:
        # Instantiating the reader once caches it and ensures PyTorch's runtime is fully initialized first.
        _ = easyocr.Reader(['en'], gpu=False)
    except Exception:
        pass
