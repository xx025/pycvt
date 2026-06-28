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
    convert_yolo_dataset_to_coco,
    prepare_yolo_dataset_for_coco,
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
"convert_yolo_dataset_to_coco", # convert a YOLO data.yaml dataset into RF-DETR/COCO layout
"prepare_yolo_dataset_for_coco", # prepare a shared converted dataset directory with file locking
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

`pycvt` 提供了一个基于 Ray 的 YOLO 数据集预测工具，详细用法见 [doc/yolo-pred.md](doc/yolo-pred.md)。

## YOLO To COCO

```python
from pycvt import convert_yolo_dataset_to_coco

summary = convert_yolo_dataset_to_coco(
    yaml_path="data.yaml",
    output_dir="output/rfdetr-dataset",
    copy_images=False,
    overwrite=False,
    include_test=True,
    num_workers=8,
)
print(summary)
```
