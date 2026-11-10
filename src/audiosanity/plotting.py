"""Figures from ``summary.csv``.

Each check has its own panel type, because its conditions mean different things:

    randomization  line over the (ordered) cascade of randomized layer groups
    shift          line over the shift in frames (numeric axis)
    invariance     dots with CI per transformation (categories, no lines between them)
    loudness       dots with CI per reference signal, next to its random control
    agreement      method x method matrix: upper triangle = same clip, lower = control

For every check two files are written: ``<check>.png`` with the primary normalisation/measure
(``plot`` section of the config) and ``<check>_all.png`` with every combination.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CONTROL = "control"
CONTROL_STYLE = dict(ls="--", lw=0.9)


def _is_control(conditions: pd.Series) -> pd.Series:
    return conditions.str.startswith(CONTROL)


def _colors(methods: list[str]) -> dict[str, str]:
    return {m: f"C{i}" for i, m in enumerate(methods)}


def _ci(group: pd.DataFrame) -> np.ndarray:
    """Asymmetric error bars ``(2, n)`` from the CI columns."""
    return np.vstack([group["mean"] - group["ci_low"], group["ci_high"] - group["mean"]]).clip(min=0)


# ---------------------------------------------------------------------------------------------- panels


def _line_panel(ax, panel: pd.DataFrame, colors: dict, x_of: Callable[[pd.Series], list], xlabels: bool):
    for method, group in panel.groupby("method", sort=False):
        main, control = group[~_is_control(group["condition"])], group[_is_control(group["condition"])]
        x = x_of(main["condition"])
        color = colors[method]
        ax.plot(x, main["mean"], marker="o", ms=3, color=color, label=method)
        ax.fill_between(x, main["ci_low"], main["ci_high"], color=color, alpha=0.2)
        for value in control["mean"]:
            ax.axhline(value, color=color, **CONTROL_STYLE)
    return x


def randomization_panel(ax, panel, colors, xlabels):
    conditions = list(dict.fromkeys(panel.loc[~_is_control(panel["condition"]), "condition"]))
    _line_panel(ax, panel, colors, lambda c: [conditions.index(v) for v in c], xlabels)
    ax.set_xticks(range(len(conditions)), conditions if xlabels else [], rotation=60, ha="right", fontsize=7)


def _shift_value(condition: str) -> int:
    return int(re.fullmatch(r"shift_([+-]?\d+)", condition).group(1))


def shift_panel(ax, panel, colors, xlabels):
    """Points, not lines: equivariance need not be smooth in the shift (e.g. AST's 10-frame patch grid)."""
    methods = list(colors)
    shifts = sorted({_shift_value(c) for c in panel["condition"] if not c.startswith(CONTROL)})
    step = min(np.diff(shifts).min(), 10) if len(shifts) > 1 else 1
    width = 0.6 * step / max(1, len(methods))
    for i, method in enumerate(methods):
        group = panel[panel["method"] == method]
        main, control = group[~_is_control(group["condition"])], group[_is_control(group["condition"])]
        x = np.array([_shift_value(c) for c in main["condition"]]) + (i - (len(methods) - 1) / 2) * width
        color = colors[method]
        ax.errorbar(x, main["mean"], yerr=_ci(main), fmt="o", ms=4, color=color, capsize=2, label=method)
        for value in control["mean"]:
            ax.axhline(value, color=color, **CONTROL_STYLE)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xticks(shifts, [f"{s:+d}" for s in shifts] if xlabels else [], fontsize=7)
    if xlabels:
        ax.set_xlabel("shift [frames]", fontsize=8)


def _category_order(conditions: pd.Series) -> list[str]:
    """Conditions in order of appearance; ``control_<x>`` right after ``<x>``, ``control`` last."""
    seen = list(dict.fromkeys(conditions))
    order = []
    for c in seen:
        if not c.startswith(CONTROL):
            order.append(c)
            if f"{CONTROL}_{c}" in seen:
                order.append(f"{CONTROL}_{c}")
    order += [c for c in seen if c not in order]
    return order


def dot_panel(ax, panel, colors, xlabels):
    order = _category_order(panel["condition"])
    methods = list(colors)
    width = 0.7 / max(1, len(methods))
    for k, c in enumerate(order):
        if c.startswith(CONTROL):
            ax.axvspan(k - 0.45, k + 0.45, color="0.92", zorder=0)
    for i, method in enumerate(methods):
        group = panel[panel["method"] == method].set_index("condition").reindex(order).dropna(subset=["mean"])
        if group.empty:
            continue
        x = np.array([order.index(c) for c in group.index]) + (i - (len(methods) - 1) / 2) * width
        ctrl = group.index.str.startswith(CONTROL)
        ax.errorbar(
            x[~ctrl], group["mean"][~ctrl], yerr=_ci(group[~ctrl]),
            fmt="o", ms=4, color=colors[method], capsize=2, label=method,
        )  # fmt: skip
        ax.errorbar(
            x[ctrl], group["mean"][ctrl], yerr=_ci(group[ctrl]),
            fmt="o", ms=4, mfc="white", color=colors[method], capsize=2,
        )  # fmt: skip
    labels = [c.replace(f"{CONTROL}_", f"{CONTROL}\n") for c in order]
    ax.set_xticks(range(len(order)), labels if xlabels else [], rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.6, len(order) - 0.4)


def agreement_panel(ax, panel, colors, xlabels):
    methods = list(colors)
    n = len(methods)
    matrix = np.full((n, n), np.nan)
    for row in panel.itertuples():
        a, b = row.method.split("|")
        i, j = sorted((methods.index(a), methods.index(b)))
        if row.condition.startswith(CONTROL):
            matrix[j, i] = row.mean
        else:
            matrix[i, j] = row.mean
    finite = matrix[np.isfinite(matrix)]
    image = ax.imshow(matrix, cmap="viridis", vmin=min(0.0, finite.min()), vmax=max(1e-6, finite.max()))
    for i in range(n):
        for j in range(n):
            if np.isfinite(matrix[i, j]):
                color = "white" if matrix[i, j] < finite.mean() else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
    short = [m.replace("_", "\n") for m in methods]
    ax.set_xticks(range(n), short if xlabels else [], fontsize=7)
    ax.set_yticks(range(n), short, fontsize=7)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------------------------- figures

PANELS = {
    "randomization": (randomization_panel, "similarity to the trained model's map (dashed: random control)"),
    "shift": (shift_panel, "similarity to the shifted-back original map (dashed: random control)"),
    "invariance": (dot_panel, "similarity to the map of the unmodified clip (hollow: random control)"),
    "loudness": (dot_panel, "correlation of the map with the input (hollow: random control)"),
    "agreement": (agreement_panel, "upper triangle: same clip, lower triangle: random control"),
}


def _methods(data: pd.DataFrame, check: str) -> list[str]:
    if check == "agreement":
        return list(dict.fromkeys(m for pair in data["method"] for m in pair.split("|")))
    return list(dict.fromkeys(data["method"]))


def plot_check(summary: pd.DataFrame, check: str, path: Path, norms=None, measures=None) -> None:
    data = summary[summary["check"] == check]
    norms = norms or list(dict.fromkeys(data["normalization"]))
    measures = measures or list(dict.fromkeys(data["measure"]))
    draw, subtitle = PANELS.get(check, (randomization_panel, ""))
    colors = _colors(_methods(data, check))

    size = (4.6, 3.8) if check == "agreement" else (4.2, 3.0)
    fig, axes = plt.subplots(
        len(norms), len(measures), figsize=(size[0] * len(measures), size[1] * len(norms)), squeeze=False
    )
    for r, norm in enumerate(norms):
        for c, measure in enumerate(measures):
            ax = axes[r][c]
            panel = data[(data["normalization"] == norm) & (data["measure"] == measure)]
            if panel["mean"].isna().all():  # e.g. SSIM on time-only maps of waveform models
                ax.text(0.5, 0.5, "n/a for this model", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{norm} / {measure}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            draw(ax, panel, colors, xlabels=(r == len(norms) - 1) or check == "agreement")
            ax.set_title(f"{norm} / {measure}", fontsize=9)
            if check != "agreement":
                ax.grid(alpha=0.3)
    fig.tight_layout()
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
            ncol=min(4, len(labels)), fontsize=8, title_fontsize=9,
            title=f"{check}: {subtitle}\n(bars/bands: 95% bootstrap CI)",
        )  # fmt: skip
    else:  # matrix plots: mean values, no intervals
        fig.text(0.5, 1.0, f"{check}: {subtitle}", ha="center", va="bottom", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_all(
    summary: pd.DataFrame, out_dir: Path, normalization: str = "abs", measure: str = "spearman"
) -> list[Path]:
    """Write ``<check>.png`` (primary normalisation/measure) and ``<check>_all.png`` for every check."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summary.dropna(subset=["check"])
    paths = []
    for check in summary["check"].unique():
        data = summary[summary["check"] == check]
        norm = normalization if normalization in set(data["normalization"]) else data["normalization"].iloc[0]
        meas = measure if measure in set(data["measure"]) else data["measure"].iloc[0]
        main, full = out_dir / f"{check}.png", out_dir / f"{check}_all.png"
        plot_check(summary, check, main, [norm], [meas])
        plot_check(summary, check, full)
        paths += [main, full]
    return paths
