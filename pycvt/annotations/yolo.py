import os
from pathlib import Path
import shutil
import numpy as np
import imageio.v3 as iio
from typing import Any
from pycvt.vision.bbox import xyxy2xywhn
from pycvt.vision.plot_boxes import draw_bounding_boxes


def parase_yolo_line(line):
    line = str(line)
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    segs = [float(p) for p in parts[5:]]
    return class_id, [x_center, y_center, width, height], segs


def load_yolo_annotations(file_path) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Load YOLO annotations from a text file.
    Each line in the file should be in the format:

    <class_id> <x_center> <y_center> <width> <height> [<segmentation_points>...]
    where coordinates are normalized between 0 and 1.
    Args:
        file_path (str): Path to the YOLO annotation text file.
    Returns:
        tuple: A tuple containing:
            - cls (np.ndarray): Array of class IDs.
            - bboxes (np.ndarray): Array of bounding boxes in the format [x_center, y_center, width, height].
            - segs (list): List of segmentation points for each object.
    """
    cls = []
    bboxes = []
    segs = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            class_id, bbox, seg = parase_yolo_line(line)
            cls.append(class_id)
            bboxes.append(bbox)
            segs.append(seg)
    return np.array(cls, dtype=int), np.array(bboxes, dtype=float), segs


def save_yolo_annotations(file_path, cls, bboxes, segs=None):

    cls = np.asarray(cls, dtype=int)
    bboxes = np.asarray(bboxes, dtype=float)

    with open(file_path, "w") as file:
        for i in range(len(cls)):
            line = (
                f"{cls[i]} {bboxes[i][0]} {bboxes[i][1]} {bboxes[i][2]} {bboxes[i][3]}"
            )
            if segs is not None and len(segs) > i:
                seg_str = " ".join(map(str, segs[i]))
                line += f" {seg_str}"
            file.write(line + "\n")


def load_yolo_names(names_path):
    names_map = {}
    with open(names_path, "r") as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            names_map[idx] = line.strip()
    return names_map



def save_inference_results(
    save_stem: str,
    image: Any,
    image_path: Path,
    pred: Any,
    save_txt: bool,
    save_plot: bool,
    save_source: bool,
    save_dir: Path,
    save_txt_dir: Path = None,
    save_plot_dir: Path = None,
    save_source_dir: Path = None,
):
    save_txt_dir = save_txt_dir or save_dir / "labels"
    save_plot_dir = save_plot_dir or save_dir / "plot"
    save_source_dir = save_source_dir or save_dir / "source"

    cls = pred[:, 5]
    boxes = pred[:, :4]
    scores = pred[:, 4]

    if save_txt:
        save_txt_dir.mkdir(parents=True, exist_ok=True)
        xywhn_boxes = xyxy2xywhn(boxes, image.shape[1], image.shape[0])
        save_yolo_annotations(
            file_path=save_txt_dir / f"{save_stem}.txt",
            cls=cls,
            bboxes=xywhn_boxes,
        )
    if save_plot:
        save_plot_dir.mkdir(parents=True, exist_ok=True)
        plot_img = draw_bounding_boxes(
            image=image,
            boxes=boxes.astype(int),
            labels=[f"{int(c)} {s:.2f}" for c, s in zip(cls, scores)],
            line_width=2,
        )
        iio.imwrite(
            (save_plot_dir / f"{save_stem}.jpg").as_posix(), plot_img, quality=95
        )
    if save_source:
        save_source_dir.mkdir(parents=True, exist_ok=True)
        source_path = save_source_dir / f"{save_stem}.jpg"
        if not source_path.exists():
            try:
                os.link(image_path.as_posix(), source_path.as_posix())
            except Exception:
                shutil.copyfile(image_path.as_posix(), source_path.as_posix())




if __name__ == "__main__":
    path = "/mnt/d/workspace/pycvt/scripts/coco8/labels/train/000000000009.txt"
    cls, bbox = load_yolo_annotations(path)[:2]
    print(cls, bbox)

    save_path = "/mnt/d/workspace/pycvt/scripts/coco8/000000000009_out.txt"
    import torch

    cls = torch.from_numpy(cls)
    bbox = torch.from_numpy(bbox)
    save_yolo_annotations(save_path, cls, bbox)
    print(f"Saved annotations to {save_path}")
