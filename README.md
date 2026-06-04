# pycvt



## Install

```
pip install pycvt --upgrade
```

## dev

``` 
uv sync
```

## Usage
``` python
from pycvt import (
    load_yolo_annotations,
    save_yolo_annotations,
    load_yolo_names,
    draw_bounding_boxes,
    xyxy2xywh,
    xywh2xyxy,
    xyxy2xywhn,
    xywhn2xyxy,
    box_iou,
    generate_sliding_windows,
    crop_with_bbox,
    sliding_crop,
    scale_boxes,
    get_color,
    example_file,
)

"load_yolo_annotations", # to load yolo format annotations from a file
"save_yolo_annotations", # to save yolo format annotations to a file
"load_yolo_names",      # to load class names from a yolo names file
"draw_bounding_boxes",  # to draw bounding boxes on an image
"xyxy2xywh",            # convert bounding box from (x1, y1, x2, y2) to (x_center, y_center, width, height)
"xywh2xyxy",            # convert bounding box from (x_center, y_center, width, height) to (x1, y1, x2, y2)
"xyxy2xywhn",           # convert bounding box from (x1, y1, x2, y2) to normalized (x_center, y_center, width, height)
"xywhn2xyxy",           # convert bounding box from normalized (x_center, y_center, width, height) to (x1, y1, x2, y2)
"box_iou",              # calculate Intersection over Union (IoU) between two sets of boxes
"generate_sliding_windows",  # generate sliding window coordinates for an image
"crop_with_bbox",         # crop image regions with bounding boxes and adjust boxes accordingly
"sliding_crop",         # crop an image using sliding windows
"scale_boxes",          # scale bounding boxes by a factor
"get_color",             # get a color for a given class id
"example_file",            # one example file for testing purposes  


```

## YOLO Predict CLI

集成后的数据集预测工具可以直接通过 `pycvt` 使用：

```bash
pip install -e .[yolo-predict]
pycvt-yolo-predict --config demo.yaml
```

也可以在代码里调用：

```python
from pycvt.tools.predict_config import load_predict_config
from pycvt.tools.yolo_dataset import predict_dataset

config = load_predict_config("demo.yaml")
predict_dataset(config, "demo.yaml")
```

配置格式示例：

```yaml
dataset: /path/to/dataset.yaml

prediction_store:
  root: predictions
  run: xxx-model
  plot: true

model:
  name: yolov8det
  weights: /models/xxx-model.torchscript
  conf: 0.25
  iou: 0.45
  classes: null
  imgsz: 640
  half: true
  nc: null

ray:
  num_actors: null
  num_cpus: 2.0
  gpus_per_actor: 0.25
```
