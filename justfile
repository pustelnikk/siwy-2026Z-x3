# Task runner - `just` lists the recipes. Every recipe runs inside the uv environment.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

default:
    @just --list

# Create / update the virtual environment
setup:
    uv sync

# Download ESC-50 into data/raw
data:
    uv run audiosanity download esc50

# Run the battery: `just run configs/pilot.yaml data.n_clips=10`
run config="configs/pilot.yaml" *overrides:
    uv run audiosanity run {{config}} {{overrides}}

# Fast end-to-end run on 4 clips
smoke:
    uv run audiosanity run configs/smoke.yaml

# Re-draw figures of a run: `just plot results/<run> plot.measure=ssim`
plot run_dir *overrides:
    uv run audiosanity plot {{run_dir}} {{overrides}}

# Unit tests (tiny random models, no downloads)
test *args:
    uv run pytest {{args}}

# Lint + format check
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-format and fix lint issues
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Open TensorBoard over all runs
tb:
    uv run tensorboard --logdir results
