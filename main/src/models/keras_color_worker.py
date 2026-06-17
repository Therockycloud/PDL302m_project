"""Out-of-process TF/Keras colour-inference worker.

Runs in the isolated ``dpl-train`` env (the only one where TensorFlow imports
and runs). TF cannot live in the dashboard process — it crashes together with
PaddleOCR (``mutex lock failed``) and its protobuf pin conflicts with paddle in
the base env. So the dashboard talks to this worker over stdin/stdout instead of
importing TF.

Protocol (one JSON object per line):
    parent -> worker:  {"path": "/tmp/crop.png"}
    worker -> parent:  {"label": "White", "conf": 0.83}
On startup the worker prints a single line ``{"ready": true}`` once the model
is loaded so the parent can synchronise.
"""
import json
import os
import sys

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Sorted folder order used by image_dataset_from_directory at train time.
CLASSES = ["Black", "Blue", "Brown", "Grey", "Red", "Silver", "White", "Yellow"]
IMG = 224


def main() -> None:
    model_path = sys.argv[1]
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    model = tf.keras.models.load_model(model_path)

    # The model graph starts with Rescaling(255.0) (expects [0,1] input), so we
    # feed pixel/255 — mirroring the training dataset normalisation exactly.
    def predict(path: str):
        with Image.open(path) as im:
            # BILINEAR to match tf.keras image_dataset_from_directory's default
            # resize (avoids a train/serve preprocessing skew).
            im = im.convert("RGB").resize((IMG, IMG), Image.BILINEAR)
            arr = np.asarray(im, dtype="float32") / 255.0
        probs = model.predict(arr[None], verbose=0)[0]
        idx = int(probs.argmax())
        return CLASSES[idx], float(probs[idx])

    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            label, conf = predict(req["path"])
            resp = {"label": label, "conf": conf}
        except Exception as exc:  # noqa: BLE001
            resp = {"label": "UNKNOWN", "conf": 0.0, "error": str(exc)}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
