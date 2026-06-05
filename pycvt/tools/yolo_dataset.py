from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from pycvt.annotations.yolo import save_yolo_annotations
from pycvt.tools.predict_config import PredictConfig, build_model, default_run_name
from pycvt.utils.yaml_utils import load_yaml
from pycvt.vision.bbox import xyxy2xywhn

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_dataset_yaml(dataset_yaml_path: str | Path) -> tuple[Path, dict[str, Any]]:
    yaml_path = Path(dataset_yaml_path).expanduser().resolve()
    data = load_yaml(yaml_path)
    dataset_root = Path(data.get("path", yaml_path.parent)).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()
    return dataset_root, data


def resolve_source_path(dataset_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (dataset_root / path).resolve()


def resolve_split_sources(dataset_root: Path, dataset_yaml: dict[str, Any]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for split in ("train", "val", "test"):
        value = dataset_yaml.get(split)
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                sources.append((split, resolve_source_path(dataset_root, item)))
            continue
        sources.append((split, resolve_source_path(dataset_root, value)))

    if not sources:
        raise ValueError("None of train/val/test splits exist in dataset yaml")
    return sources


def collect_images(source: str | Path) -> list[Path]:
    source_path = Path(source).expanduser().resolve()
    if source_path.is_file():
        if source_path.suffix.lower() == ".txt":
            images: list[Path] = []
            with source_path.open("r", encoding="utf-8") as f:
                for line in f:
                    item = line.strip()
                    if not item:
                        continue
                    item_path = Path(item).expanduser()
                    image_path = (
                        item_path.resolve()
                        if item_path.is_absolute()
                        else (source_path.parent / item_path).resolve()
                    )
                    if image_path.suffix.lower() in IMAGE_SUFFIXES:
                        images.append(image_path)
            return images
        if source_path.suffix.lower() in IMAGE_SUFFIXES:
            return [source_path]
        return []

    if source_path.is_dir():
        return sorted(
            path.resolve()
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    raise FileNotFoundError(f"dataset source not found: {source_path}")


def image_to_prediction_label_path(
    image_path: str | Path,
    run: str,
    prediction_root: str = "predictions",
) -> Path:
    """Map image path to prediction label path.

    Same idea as YOLO dataset loading:
        images -> labels

    Here:
        images -> prediction_root/run
    """
    image_file = Path(image_path).resolve()
    parts = image_file.parts

    if "images" not in parts:
        raise ValueError(f"image path does not contain an 'images' directory: {image_file}")

    # YOLO 数据集里通常只有一个 images；
    # 用最后一个更安全，避免上层目录名也叫 images。
    idx = len(parts) - 1 - parts[::-1].index("images")

    return Path(
        *parts[:idx],
        prediction_root,
        run,
        *parts[idx + 1:],
    ).with_suffix(".txt")

def ensure_detection_array(detections: Any) -> np.ndarray:
    output = np.asarray(detections, dtype=np.float32)
    if output.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    output = np.atleast_2d(output)
    if output.shape[1] < 6:
        raise ValueError(
            f"Expected detection output with at least 6 columns, got shape {output.shape}"
        )
    return output[:, :6]


def save_prediction_txt(
    txt_path: str | Path,
    boxes: np.ndarray,
    classes: np.ndarray,
    confs: np.ndarray,
) -> None:
    path = Path(txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(boxes) == 0:
        path.write_text("", encoding="utf-8")
        return

    save_yolo_annotations(
        file_path=path,
        cls=classes,
        bboxes=boxes,
        confs=confs,
    )


class YoloPredictActor:
    def __init__(self, model_kwargs: dict[str, Any], runs_config: dict[str, Any]):
        build_kwargs = dict(model_kwargs["build_kwargs"])

        try:
            import ray

            gpu_ids = ray.get_gpu_ids()
        except Exception:
            gpu_ids = []

        build_kwargs["device"] = "cuda" if gpu_ids else "cpu"
        self.model = build_model(model_kwargs["model_name"], **build_kwargs)
        self.model.load_model()
        self.runs_config = runs_config

    def infer(self, task: str) -> dict[str, str]:
        try:
            txt_path = run_prediction_task(
                image_path=task,
                model=self.model,
                run_name=self.runs_config["run_name"],
                prediction_root=self.runs_config["prediction_root"],
            )
            return {"status": "ok", "image_path": task, "txt_path": str(txt_path)}
        except Exception as exc:
            return {"status": "error", "image_path": task, "error": f"{type(exc).__name__}: {exc}"}


def collect_dataset_images(sources: list[tuple[str, Path]]) -> list[Path]:
    image_paths: list[Path] = []
    seen: set[Path] = set()
    for _, source in sources:
        for image_path in collect_images(source):
            if image_path not in seen:
                seen.add(image_path)
                image_paths.append(image_path)
    return image_paths


def run_prediction_task(
    image_path: str | Path,
    model: Any,
    run_name: str,
    prediction_root: str,
) -> Path:
    image_file = Path(image_path).resolve()
    image = iio.imread(image_file, mode="RGB")
    detections = ensure_detection_array(model(image))
    yolo_boxes = xyxy2xywhn(detections[:, :4], w=image.shape[1], h=image.shape[0])
    scores = detections[:, 4]
    classes = detections[:, 5].astype(int)
    txt_path = image_to_prediction_label_path(
        image_path=image_file,
        run=run_name,
        prediction_root=prediction_root,
    )
    save_prediction_txt(
        txt_path=txt_path,
        boxes=yolo_boxes,
        classes=classes,
        confs=scores,
    )

    return txt_path


def predict_dataset_ray(
    config: PredictConfig,
    config_path: str | Path,
    dataset_root: Path,
    image_paths: list[Path],
) -> None:
    try:
        import ray
        from cvmd.utils.ray_infer import ray_infer_iter
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"Failed to import 'cvmd.utils.ray_infer': {exc}. "
            "Make sure 'cvmd', 'ray', and runtime dependencies such as 'torch' are installed."
        ) from exc

    if not ray.is_initialized():
        os.environ.setdefault("RAY_DEDUP_LOGS", "1")
        ray.init(logging_level="ERROR", log_to_driver=False)

    run_name = config.prediction_store.run or default_run_name(
        config.model.weights, config_path
    )
    model_kwargs = {
        "model_name": config.model.name,
        "build_kwargs": {
            "weights": config.model.weights,
            "conf": config.model.conf,
            "iou": config.model.iou,
            "classes": config.model.classes,
            "imgsz": config.model.imgsz,
            "half": config.model.half,
            "nc": config.model.nc,
        },
    }
    runs_config = {
        "run_name": run_name,
        "prediction_root": config.prediction_store.root,
    }
    actor_kwargs = {
        "runs_config": runs_config,
        "model_kwargs": model_kwargs,
    }

    total = len(image_paths)
    done = 0
    failed: list[dict[str, str]] = []

    print(f"Predicting {total} images with Ray...")

    for result in ray_infer_iter(
        YoloPredictActor,
        tasks=[str(path) for path in image_paths],
        num_actors=config.ray.num_actors,
        num_cpus=config.ray.num_cpus,
        gpus_per_actor=config.ray.gpus_per_actor,
        actor_kwargs=actor_kwargs,
    ):
        done += 1
        if result.get("status") != "ok":
            failed.append(result)

        if done == total or done % max(1, min(50, total // 20 if total > 20 else 10)) == 0:
            print(
                f"\rProgress: {done}/{total} | "
                f"ok: {done - len(failed)} | failed: {len(failed)}",
                end="" if done < total else "\n",
                flush=True,
            )

    if failed:
        print(f"\nCompleted with {len(failed)} failed image(s):")
        for item in failed[:20]:
            print(f"- {item['image_path']}: {item['error']}")
        if len(failed) > 20:
            print(f"- ... and {len(failed) - 20} more")


def predict_dataset(config: PredictConfig, config_path: str | Path) -> None:
    dataset_yaml_path = Path(config.dataset).expanduser().resolve()
    dataset_root, dataset_yaml = resolve_dataset_yaml(dataset_yaml_path)
    sources = resolve_split_sources(dataset_root, dataset_yaml)
    image_paths = collect_dataset_images(sources)
    if not image_paths:
        raise ValueError("No images found in requested dataset splits")
    predict_dataset_ray(config, config_path, dataset_root, image_paths)
