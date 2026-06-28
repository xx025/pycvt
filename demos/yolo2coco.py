import os
from pathlib import Path

from clearml import Task
from pycvt import prepare_yolo_dataset_for_coco
from rfdetr import RFDETRMedium
from torch.distributed.utils import is_main_process

import torch


def main():
    SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
    if is_main_process():
        Task.init(
            project_name="train-demo",
            task_name=f"experiment-{SLURM_JOB_ID}",
        )

    # 1. Prepare the dataset， yolo dataset yaml file path
    yaml_path = Path("/mnt/dataset.yaml")

    dataset_dir = prepare_yolo_dataset_for_coco(
        yaml_path=yaml_path,
        save_dir=Path(f"/tmp/dataset/{yaml_path.stem}_{SLURM_JOB_ID}"),
    )

    output_dir = Path(f"/mnt/runs/output2-{SLURM_JOB_ID}")
    
    # 2. Train the model
    model = RFDETRMedium()
    model.train(
        dataset_dir=dataset_dir,
        epochs=200,
        batch_size=8,
        grad_accum_steps=2,
        lr=1e-4,
        output_dir=output_dir,
        devices="auto",
    )



if __name__ == "__main__":

    main()
