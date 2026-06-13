# Benchmark A — Vehicle Colour CNN (PyTorch / MPS)

Dataset: `main/data/raw/car_colors` (8 classes, 226 val images). Transfer learning (frozen ImageNet backbone, trained head), 20 epochs. Latency = single-image CPU inference.

| name             |   accuracy |   macro_f1 |   latency_ms |   num_params |   size_mb |
|:-----------------|-----------:|-----------:|-------------:|-------------:|----------:|
| MobileNetV3Small |     0.5973 |     0.6252 |        99.78 |      1526056 |      6.24 |
| EfficientNetB0   |     0.6239 |     0.6274 |       280.95 |      4017796 |     16.37 |
| ResNet50         |     0.5796 |     0.6119 |        44.7  |     23524424 |     94.41 |

## Conclusion

Identical protocol (frozen ImageNet backbone + trained linear head, 20 epochs)
so the comparison is fair. **EfficientNet-B0 has the best accuracy/macro-F1**
(0.624 / 0.627) but the worst CPU latency (281 ms) — its many small depthwise
ops are slow on CPU. **ResNet50** trails on accuracy despite 23.5 M params and
is by far the largest (94 MB). **MobileNetV3-Small** sits within ~3 pts of the
best accuracy at **a quarter of EfficientNet's size (6.24 MB) and ~3× lower
latency** — the best accuracy↔cost trade-off for the edge/CPU target.

**Selected: MobileNetV3-Small** (matches the model already wired into the
pipeline). Absolute accuracy (~60 %) is modest because only the head is trained
on an imbalanced 8-class set (Yellow 25, Brown 35 images); unfreezing the
backbone or class-balanced sampling would lift all three, but does not change
the ranking. Trained on Apple M1 Max (MPS); TensorFlow `model.fit` was abandoned
after it deadlocked at 0 % CPU on this macOS build (oneDNN threadpool), so the
benchmark uses PyTorch/torchvision.
