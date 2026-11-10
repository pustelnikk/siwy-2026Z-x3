"""Command-line interface: ``audiosanity --help``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def download(
    dataset: Annotated[str, typer.Argument(help="Dataset to fetch (esc50).")] = "esc50",
    raw_dir: Path = Path("data/raw"),
) -> None:
    """Download a dataset into data/raw (never modified afterwards)."""
    from audiosanity.data import download_esc50

    if dataset != "esc50":
        raise typer.BadParameter(f"unknown dataset {dataset!r}")
    typer.echo(download_esc50(raw_dir))


@app.command()
def run(
    config: Annotated[Path, typer.Argument(exists=True, help="YAML config, e.g. configs/pilot.yaml")],
    overrides: Annotated[list[str] | None, typer.Argument(help="Overrides like data.n_clips=10")] = None,
) -> None:
    """Run the battery described by CONFIG and write a results directory."""
    from audiosanity.config import load_config
    from audiosanity.pipeline import run_experiment

    run_dir = run_experiment(load_config(config, overrides))
    typer.echo(f"Results written to {run_dir}")


@app.command()
def plot(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    overrides: Annotated[list[str] | None, typer.Argument(help="e.g. plot.measure=ssim")] = None,
) -> None:
    """Re-draw the figures of an existing run from its summary.csv and config.yaml."""
    from audiosanity.config import load_config
    from audiosanity.plotting import plot_all

    cfg = load_config(run_dir / "config.yaml", overrides)
    summary = pd.read_csv(run_dir / "summary.csv")
    for path in plot_all(summary, run_dir / "figures", cfg.plot.normalization, cfg.plot.measure):
        typer.echo(path)


@app.command()
def methods() -> None:
    """List available models, attribution methods, checks and measures."""
    from audiosanity.attribution import METHODS
    from audiosanity.checks import CHECKS
    from audiosanity.measures import MEASURES
    from audiosanity.models import REGISTRY

    typer.echo(f"models:   {', '.join(REGISTRY)}")
    typer.echo(f"methods:  {', '.join(METHODS)}")
    typer.echo(f"checks:   {', '.join(CHECKS)}")
    typer.echo(f"measures: {', '.join(MEASURES)}")


if __name__ == "__main__":
    app()
