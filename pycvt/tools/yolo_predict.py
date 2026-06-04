from pathlib import Path

from pycvt.tools.predict_config import load_predict_config
from pycvt.tools.yolo_dataset import predict_dataset


def main() -> None:
    try:
        import click
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The 'yolo-predict' CLI requires the optional dependency group "
            "'pycvt[yolo-predict]'."
        ) from exc

    @click.command(help="Run cvmd predictions on a YOLO dataset and save txt outputs.")
    @click.option(
        "-c",
        "--config",
        "config_path",
        type=click.Path(path_type=Path, dir_okay=False, exists=True),
        required=True,
        help="Path to the prediction config yaml file.",
    )
    def cli(config_path: Path) -> None:
        resolved_config_path = config_path.expanduser().resolve()
        config = load_predict_config(resolved_config_path)
        predict_dataset(config, resolved_config_path)

    cli()
