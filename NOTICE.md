# NOTICE

RockSpot 项目代码以 [MIT](LICENSE) 发布。本文件说明第三方软件和训练数据出处。

This repository redistributes inference code and model weights. It does not redistribute training images.

## 软件依赖 / Dependencies

`ultralytics` 包（YOLOv8）采用 **AGPL-3.0**。它是运行时依赖，**不**因本仓库的 MIT 许可证而被再许可。把本项目代码与 ultralytics 组合使用时，AGPL-3.0 可能约束该组合作品。请自行确认你的分发方式。

The `ultralytics` package is AGPL-3.0. This MIT file does not relicense it.

PyTorch 与 torchvision 使用各自的许可证。分类支路默认是 torchvision 的 MobileNetV3-Small ImageNet 权重，不是岩石类别模型。

## 训练数据 / Training data

`weights/rock_yolov8s_seg_boulder_ft3.pt` 与同名 ONNX 的训练谱系使用过下列数据。图片留在原出处，**本仓库不附带这些图像文件**。上游许可证约束的是这些数据本身；引用权重时仍应保留下面的署名。

- Boulder / BOULDERING（Boulder2024），Prieur et al.，Zenodo [10.5281/zenodo.14250874](https://doi.org/10.5281/zenodo.14250874)，**CC BY 4.0**。用于热身阶段。
- ImageGrains 2.0，Mair et al.，Zenodo [10.5281/zenodo.17866827](https://doi.org/10.5281/zenodo.17866827)，**CC BY 4.0**。
- Segmenteverygrain 训练掩膜，Sylvester, Z.，Zenodo [10.5281/zenodo.15786086](https://doi.org/10.5281/zenodo.15786086)，**CC BY 4.0**。
- 作者自摄金标：照片权利属于作者本人，不在本仓库中分发。

Author self-capture photos are not included. Fetch upstream images from the Zenodo records above if you need to retrain.
