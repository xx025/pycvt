from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from filelock import FileLock
from PIL import Image

from .yolo import load_yolo_annotations
from ..utils.yaml_utils import load_yaml
from ..vision.bbox import xywhn2xyxy


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageRecord:
    source_path: Path
    label_path: Optional[Path]
    relative_name: str


@dataclass
class SplitSummary:
    split_name: str
    image_count: int
    annotation_count: int
    missing_label_count: int
    output_dir: str
    worker_count: int


@dataclass
class ConversionSummary:
    yaml_path: str
    dataset_root: str
    output_dir: str
    class_count: int
    mode: str
    splits: list[SplitSummary]


def resolve_dataset_root(yaml_path: str | Path, yaml_data: dict) -> Path:
    yaml_path = Path(yaml_path).resolve()
    path_value = yaml_data.get("path")
    if not path_value:
        return yaml_path.parent

    dataset_root = Path(path_value)
    if dataset_root.is_absolute():
        return dataset_root.resolve()
    return (yaml_path.parent / dataset_root).resolve()


def normalize_names(names: object) -> list[str]:
    if isinstance(names, list):
        return [str(item) for item in names]
    if isinstance(names, dict):
        pairs = sorted((int(key), str(value)) for key, value in names.items())
        return [name for _, name in pairs]
    raise ValueError("YAML must contain 'names' as a list or dict")


def resolve_entry(path_like: str, dataset_root: Path, yaml_dir: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()

    dataset_candidate = (dataset_root / path).resolve()
    if dataset_candidate.exists():
        return dataset_candidate
    return (yaml_dir / path).resolve()


def flatten_split_value(split_value: object) -> list[str]:
    if split_value is None:
        return []
    if isinstance(split_value, str):
        return [split_value]
    if isinstance(split_value, list) and all(isinstance(item, str) for item in split_value):
        return split_value
    raise ValueError(f"Unsupported split value: {split_value!r}")


def list_images_in_dir(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise ValueError(f"Expected image directory, got: {image_dir}")

    images = [path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    images.sort()
    return images


def read_images_from_txt(txt_path: Path, dataset_root: Path) -> list[Path]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Image list file not found: {txt_path}")

    image_paths: list[Path] = []
    with txt_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            path = Path(line)
            if not path.is_absolute():
                path = dataset_root / path
            image_paths.append(path.resolve())
    return image_paths


def collect_split_images(split_value: object, dataset_root: Path, yaml_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    seen: set[Path] = set()

    for entry in flatten_split_value(split_value):
        resolved = resolve_entry(entry, dataset_root=dataset_root, yaml_dir=yaml_dir)
        if resolved.is_dir():
            candidates = list_images_in_dir(resolved)
        elif resolved.is_file() and resolved.suffix.lower() == ".txt":
            candidates = read_images_from_txt(resolved, dataset_root=dataset_root)
        elif resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES:
            candidates = [resolved]
        else:
            raise ValueError(f"Unsupported split entry: {resolved}")

        for path in candidates:
            resolved_path = path.resolve()
            if resolved_path not in seen:
                seen.add(resolved_path)
                image_paths.append(resolved_path)

    return image_paths


def make_sequential_name(index: int, image_path: Path) -> str:
    suffix = image_path.suffix.lower() or ".jpg"
    return f"{index:08d}{suffix}"


def replace_images_with_labels(path: Path) -> Optional[Path]:
    parts = list(path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return None


def find_label_path(image_path: str | Path) -> Optional[Path]:
    image_path = Path(image_path).resolve()
    candidates: list[Path] = []

    replaced = replace_images_with_labels(image_path)
    if replaced is not None:
        candidates.append(replaced)

    candidates.append(image_path.with_suffix(".txt"))
    candidates.append(image_path.parent / "labels" / f"{image_path.stem}.txt")

    if image_path.parent.parent != image_path.parent:
        candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
        candidates.append(image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def build_image_records(image_paths: Sequence[Path]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for index, image_path in enumerate(image_paths, start=1):
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        records.append(
            ImageRecord(
                source_path=image_path,
                label_path=find_label_path(image_path),
                relative_name=make_sequential_name(index=index, image_path=image_path),
            )
        )
    return records


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path, do_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if do_copy:
        shutil.copy2(src, dst)
        return

    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def safe_link_or_copy_json(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    try:
        relative_src = os.path.relpath(src, start=dst.parent)
        os.symlink(relative_src, dst)
    except OSError:
        shutil.copy2(src, dst)


def categories_from_names(names: Sequence[str]) -> list[dict]:
    return [{"id": index + 1, "name": name, "supercategory": "object"} for index, name in enumerate(names)]


def yolo_bbox_to_coco_bbox(values: Sequence[float], width: int, height: int) -> list[float]:
    xyxy = xywhn2xyxy([values], w=width, h=height, safe=True)[0]
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def load_coco_annotations_from_yolo(
    label_path: str | Path | None,
    *,
    image_id: int,
    width: int,
    height: int,
    num_categories: int,
    annotation_id_start: int = 1,
) -> tuple[list[dict], int]:
    if label_path is None:
        return [], annotation_id_start

    label_path = Path(label_path)
    if not label_path.exists():
        return [], annotation_id_start

    classes, bboxes, _ = load_yolo_annotations(label_path)
    annotations: list[dict] = []
    next_annotation_id = annotation_id_start

    for class_id, bbox in zip(classes.tolist(), bboxes.tolist()):
        class_id = int(class_id)
        if class_id < 0 or class_id >= num_categories:
            raise ValueError(
                f"Class id {class_id} out of range in {label_path}; names has {num_categories} classes"
            )

        coco_bbox = yolo_bbox_to_coco_bbox(bbox, width=width, height=height)
        area = coco_bbox[2] * coco_bbox[3]
        if area <= 0:
            continue

        annotations.append(
            {
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
            }
        )
        next_annotation_id += 1

    return annotations, next_annotation_id


def process_image_record(
    record: ImageRecord,
    image_id: int,
    split_out_dir: Path,
    do_copy: bool,
    num_categories: int,
) -> dict:
    target_path = split_out_dir / record.relative_name
    safe_link_or_copy(record.source_path, target_path, do_copy=do_copy)

    with Image.open(record.source_path) as image:
        width, height = image.size

    annotations, _ = load_coco_annotations_from_yolo(
        record.label_path,
        image_id=image_id,
        width=width,
        height=height,
        num_categories=num_categories,
    )

    return {
        "image": {
            "id": image_id,
            "file_name": record.relative_name,
            "width": width,
            "height": height,
        },
        "annotations": annotations,
        "missing_label": record.label_path is None,
    }


def write_split(
    split_name: str,
    split_dir_name: str,
    records: Sequence[ImageRecord],
    output_dir: Path,
    names: Sequence[str],
    do_copy: bool,
    overwrite: bool,
    num_workers: int,
) -> SplitSummary:
    if not records:
        return SplitSummary(
            split_name=split_name,
            image_count=0,
            annotation_count=0,
            missing_label_count=0,
            output_dir=str(output_dir / split_dir_name),
            worker_count=max(1, num_workers),
        )

    split_out_dir = output_dir / split_dir_name
    ensure_clean_dir(split_out_dir, overwrite=overwrite)
    annotations_out_dir = output_dir / "annotations"
    annotations_out_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": f"Converted from YOLO YAML split: {split_name}"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories_from_names(names),
    }

    worker_count = max(1, num_workers)
    results_by_image_id: dict[int, dict] = {}
    missing_label_count = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                process_image_record,
                record,
                image_id,
                split_out_dir,
                do_copy,
                len(names),
            ): image_id
            for image_id, record in enumerate(records, start=1)
        }

        for future in as_completed(futures):
            image_id = futures[future]
            results_by_image_id[image_id] = future.result()

    next_annotation_id = 1
    for image_id in range(1, len(records) + 1):
        result = results_by_image_id[image_id]
        coco["images"].append(result["image"])

        for annotation in result["annotations"]:
            annotation["id"] = next_annotation_id
            coco["annotations"].append(annotation)
            next_annotation_id += 1

        if result["missing_label"]:
            missing_label_count += 1

    annotation_store_path = annotations_out_dir / f"{split_dir_name}.json"
    with annotation_store_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, ensure_ascii=False)

    safe_link_or_copy_json(annotation_store_path, split_out_dir / "_annotations.coco.json")
    return SplitSummary(
        split_name=split_name,
        image_count=len(coco["images"]),
        annotation_count=len(coco["annotations"]),
        missing_label_count=missing_label_count,
        output_dir=str(split_out_dir),
        worker_count=worker_count,
    )


def convert_yolo_dataset_to_coco(
    *,
    yaml_path: str | Path,
    output_dir: str | Path,
    copy_images: bool = False,
    overwrite: bool = False,
    include_test: bool = True,
    num_workers: int = 8,
) -> ConversionSummary:
    yaml_path = Path(yaml_path).resolve()
    yaml_dir = yaml_path.parent
    output_dir = Path(output_dir).resolve()

    yaml_data = load_yaml(yaml_path)
    dataset_root = resolve_dataset_root(yaml_path, yaml_data)
    names = normalize_names(yaml_data.get("names"))

    splits: list[tuple[str, str, object]] = [
        ("train", "train", yaml_data.get("train")),
        ("valid", "valid", yaml_data.get("val")),
    ]
    if include_test:
        splits.append(("test", "test", yaml_data.get("test")))

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_clean_dir(output_dir / "annotations", overwrite=overwrite)
    split_summaries: list[SplitSummary] = []

    for split_name, split_dir_name, split_value in splits:
        if split_value is None:
            continue

        image_paths = collect_split_images(split_value, dataset_root=dataset_root, yaml_dir=yaml_dir)
        records = build_image_records(image_paths)
        split_summaries.append(
            write_split(
                split_name=split_name,
                split_dir_name=split_dir_name,
                records=records,
                output_dir=output_dir,
                names=names,
                do_copy=copy_images,
                overwrite=overwrite,
                num_workers=num_workers,
            )
        )

    return ConversionSummary(
        yaml_path=str(yaml_path),
        dataset_root=str(dataset_root),
        output_dir=str(output_dir),
        class_count=len(names),
        mode="copy" if copy_images else "symlink-or-copy",
        splits=split_summaries,
    )


def conversion_summary_to_dict(summary: ConversionSummary) -> dict:
    return {
        "yaml_path": summary.yaml_path,
        "dataset_root": summary.dataset_root,
        "output_dir": summary.output_dir,
        "class_count": summary.class_count,
        "mode": summary.mode,
        "splits": [
            {
                "split_name": split.split_name,
                "image_count": split.image_count,
                "annotation_count": split.annotation_count,
                "missing_label_count": split.missing_label_count,
                "output_dir": split.output_dir,
                "worker_count": split.worker_count,
            }
            for split in summary.splits
        ],
    }


def prepare_yolo_dataset_for_coco(
    *,
    yaml_path: str | Path,
    save_dir: str | Path,
    wait_time: float = 3600,
    num_workers: int = 8,
) -> str:
    dataset_dir = Path(save_dir).resolve()
    ready_file = dataset_dir / ".ready"
    lock_file = f"{dataset_dir}.lock"

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"DATA_PREPARATION_WAIT_TIME={wait_time}s")

    with FileLock(lock_file, timeout=wait_time):
        if not ready_file.exists():
            convert_yolo_dataset_to_coco(
                yaml_path=yaml_path,
                output_dir=dataset_dir,
                copy_images=True,
                overwrite=True,
                include_test=True,
                num_workers=num_workers,
            )
            ready_file.write_text("ok\n", encoding="utf-8")

    return str(dataset_dir)


