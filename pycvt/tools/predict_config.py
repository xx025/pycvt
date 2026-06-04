from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pycvt.utils.yaml_utils import load_yaml


@dataclass
class ModelConfig:
    weights: str
    name: str = "yolov8det"
    conf: float = 0.25
    iou: float = 0.45
    classes: list[int] | None = None
    imgsz: int = 640
    half: bool = False
    nc: int | None = None


@dataclass
class PredictionStoreConfig:
    root: str = "predictions"
    run: str | None = None
    plot: bool = False


@dataclass
class RayConfig:
    num_actors: int | None = None
    num_cpus: float = 2.0
    gpus_per_actor: float = 0.25


@dataclass
class PredictConfig:
    dataset: str
    model: ModelConfig
    prediction_store: PredictionStoreConfig
    ray: RayConfig | None = None


def build_model(*args: Any, **kwargs: Any) -> Any:
    try:
        from cvmd import build
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"Failed to import 'cvmd': {exc}. "
            "Make sure required runtime dependencies such as 'torch' are installed."
        ) from exc
    return build(*args, **kwargs)


def load_predict_config(config_path: str | Path) -> PredictConfig:
    raw = load_yaml(config_path)
    model_raw = raw.get("model") or {}
    pred_raw = raw.get("prediction_store") or {}
    ray_raw = raw.get("ray") or {}

    if "dataset" not in raw:
        raise ValueError("config missing required field: dataset")
    if "weights" not in model_raw:
        raise ValueError("config missing required field: model.weights")

    return PredictConfig(
        dataset=str(raw["dataset"]),
        model=ModelConfig(
            name=str(model_raw.get("name", "yolov8det")),
            weights=str(model_raw["weights"]),
            conf=float(model_raw.get("conf", 0.25)),
            iou=float(model_raw.get("iou", 0.45)),
            classes=(
                None
                if model_raw.get("classes") in (None, "None", "null", "")
                else np.atleast_1d(model_raw.get("classes")).astype(int).tolist()
            ),
            imgsz=int(model_raw.get("imgsz", 640)),
            half=bool(model_raw.get("half", False)),
            nc=int(model_raw["nc"]) if model_raw.get("nc") is not None else None,
        ),
        prediction_store=PredictionStoreConfig(
            root=str(pred_raw.get("root", "predictions")),
            run=str(pred_raw["run"]) if pred_raw.get("run") else None,
            plot=bool(pred_raw.get("plot", False)),
        ),
        ray=RayConfig(
            num_actors=int(ray_raw["num_actors"]) if ray_raw.get("num_actors") is not None else None,
            num_cpus=float(ray_raw.get("num_cpus", 2.0)),
            gpus_per_actor=float(ray_raw.get("gpus_per_actor", 0.25)),
        ),
    )


def default_run_name(model_weights: str, config_path: str | Path) -> str:
    path = Path(config_path)
    stem = Path(model_weights).stem or "model"
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:6]
    return f"run_{stem}_{digest}"
