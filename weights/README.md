# 权重

本目录只有 ft3 分割权重，没有训练图片。

- `rock_yolov8s_seg_boulder_ft3.pt` — 默认推理权重。YOLOv8s-seg，单类 `rock`。
- `rock_yolov8s_seg_boulder_ft3.onnx` — 同一模型的 ONNX（imgsz 640，opset 17，FP32，dynamic false）。

`configs/infer.yaml` 的 `segment_weights` 指向 `.pt`，`conf` 为 0.4。把该字段改成 `weights/rock_yolov8s_seg_boulder_ft3.onnx` 即可用 ONNX。

The default checkpoint is the ft3 `.pt`. The `.onnx` file is the same model.
