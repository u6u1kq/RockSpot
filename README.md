# RockSpot

对一张照片做岩石实例分割，并给出每块的掩膜面积。模型是 **YOLOv8s-seg**，单类 `rock`（框 + 掩膜）。

YOLOv8s-seg finds rock-like instances (boxes and masks) and writes mask-area fields. It does not identify mineral species.

分类支路仍是 torchvision **MobileNetV3-Small** 的 ImageNet 权重，除非你自己提供分类器和类名文件。ImageNet 的 top-1 名称**不是**岩种。特征向量是分类头之前的池化结果，维度 576。

MobileNet features stay on ImageNet unless you add your own classifier.

本仓库只有推理代码和 ft3 权重。**没有训练图片**，请自备一张测试照片。

No training images are in this repo. Bring your own photo.

在项目自有验证集上，ft3 的 mask mAP50 约 **0.724**，mask mAP50-95 约 **0.382**。这是项目内部验证，不是公开排行榜成绩。

## 它不做什么

不做矿物鉴定，也不做岩石类型识别。密堆石块时把置信度保持在 **0.4**（`configs/infer.yaml` 已是这个值）。没有检出时，脚本会把整张图走一次分类支路，并在结果里标记 `fallback_full_image`。

## 安装

建议 Python 3.10 及以上。只需要 CPU 时，先装 CPU 版 PyTorch：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

要用 NVIDIA GPU，先按 [pytorch.org](https://pytorch.org) 安装对应 CUDA 的 torch / torchvision，再执行 `pip install -r requirements.txt`。

`ultralytics` 是 **AGPL-3.0** 依赖，本仓库的 MIT 许可证不覆盖它。数据出处见 [NOTICE.md](NOTICE.md) 与 [DATA_LICENSE.md](DATA_LICENSE.md)。

## 推理

在仓库根目录执行。把 `path/to/your.jpg` 换成你的图片：

```powershell
py -3.12 scripts/infer.py --image path/to/your.jpg --config configs/infer.yaml --out outputs/demo
```

- 默认权重：`weights/rock_yolov8s_seg_boulder_ft3.pt`
- 配置：`configs/infer.yaml`（`conf: 0.4`，`imgsz: 640`）
- ONNX：`weights/rock_yolov8s_seg_boulder_ft3.onnx`（640，opset 17，FP32）。要用它，把 yaml 里的 `segment_weights` 改成该相对路径。

结果写到 `outputs/demo/`：`results.json`、裁剪图、掩膜，以及 `.npy` 特征。字段包括 `mask_area_px`、`mask_area_ratio`、`bbox_xyxy`、`centroid_xy`、`equivalent_diameter_px`。`mask_area_ratio = mask_area_px / (高 × 宽)`。`equivalent_diameter_px = 2 * sqrt(mask_area_px / pi)`。

## 目录

| 路径 | 作用 |
| --- | --- |
| `configs/infer.yaml` | 设备、阈值、权重路径 |
| `rockspot/` | 分割、掩膜裁剪、ImageNet 特征 |
| `scripts/infer.py` | 命令行入口 |
| `weights/` | ft3 `.pt` 与 `.onnx` |
| `data/images/` | 空目录，自备图片放这里 |
| `outputs/` | 推理输出 |

相对路径相对于仓库根目录。许可：[LICENSE](LICENSE)（代码，MIT）、[NOTICE.md](NOTICE.md)、[DATA_LICENSE.md](DATA_LICENSE.md)。
