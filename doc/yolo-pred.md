# yolo-pred

`yolo-pred` 是 `pycvt` 里的 YOLO 数据集预测工具。它使用 `cvmd` 加载模型，使用 Ray actor 并行推理，并把预测标签按 YOLO 风格保存回数据集目录。

## 命令

```bash
yolo-pred --config configs/predict.yaml
```

兼容入口：

```bash
pycvt-yolo-predict --config configs/predict.yaml
```

## 依赖

这个工具的可选依赖包括：

- `cvmd`
- `ray`
- `torch`
- `torchvision`

只安装这个工具需要的可选依赖：

```bash
uv sync --extra yolo-predict
```

## 预测配置

当前配置格式：

```yaml
dataset: data/coco128.yaml

prediction_store:
  root: predictions
  run: yolov8m-ts-640

model:
  name: yolov8det
  weights: models/yolov8m.torchscript
  conf: 0.25
  iou: 0.45
  classes: null
  imgsz: 640
  half: true
  nc: 80

ray:
  num_actors: null
  num_cpus: 2.0
  gpus_per_actor: 0.25
```

字段说明：

- `dataset`: 数据集 YAML 路径
- `prediction_store.root`: 预测标签根目录名
- `prediction_store.run`: 当前模型/实验名
- `model.name`: `cvmd` 使用的模型类型
- `model.weights`: 模型权重路径
- `model.conf`: 置信度阈值
- `model.iou`: NMS IoU 阈值
- `model.classes`: 指定类别过滤；`null` 表示不过滤
- `model.imgsz`: 推理尺寸
- `model.half`: 是否使用 FP16
- `model.nc`: 类别数
- `ray.num_actors`: actor 数量，`null` 时交给 `cvmd.utils.ray_infer` 的默认行为处理
- `ray.num_cpus`: 每个 actor 使用的 CPU 数
- `ray.gpus_per_actor`: 每个 actor 使用的 GPU 数

## 数据集 YAML

当前数据集 YAML 需要长这样：

```yaml
path: data/coco128
train: images/train2017

names:
  0: person
  1: bicycle
  2: car
```

工具会自动检查 `train` / `val` / `test` 哪些存在，并把这些 split 里的图片都拿来推理。

## 输出路径规则

输出路径遵循 YOLO 的 `image -> label` 路径映射，只是在中间插入了一层 `run`：

```text
images/val/a.jpg
-> predictions/<run>/val/a.txt
```

例如：

```text
data/coco128/images/train2017/0001.jpg
-> data/coco128/predictions/yolov8m-ts-640/train2017/0001.txt
```

保存的每一行格式为：

```text
<class_id> <x_center> <y_center> <width> <height> <conf>
```

## GPU 行为

这个工具不需要在配置里单独指定 `device`。

如果 Ray 分配到了 GPU，就会使用 GPU 推理；如果当前运行环境没有可用 GPU，则会回退到 CPU。

## 进度与错误处理

正常运行时只打印简洁进度，不逐张刷输出。

示例：

```text
Predicting 128 images with Ray...
Progress: 64/128 | ok: 64 | failed: 0
```

如果某张图片推理失败：

- 不会导致整个任务崩溃
- 会继续处理后续图片
- 最后统一打印失败图片和错误摘要

## 多模型并存

`prediction_store.run` 的作用是隔离不同模型或不同实验的结果。

例如同一张图：

```text
predictions/yolov8m-a/train2017/0001.txt
predictions/yolov8m-b/train2017/0001.txt
```

这样不会互相覆盖。

## 相关代码

- CLI 入口: `pycvt/tools/yolo_predict.py`
- 配置解析: `pycvt/tools/predict_config.py`
- 数据集预测: `pycvt/tools/yolo_dataset.py`
