"""Render a slide-ready PhD workplan Gantt chart."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/deliverables/presentation/assets/phd_gantt_timeline"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

PALETTE = {
    "navy": "#0B132B",
    "teal": "#00A6A6",
    "gold": "#F6C85F",
    "coral": "#FF6B6B",
    "blue": "#3B82F6",
    "grey": "#5C677D",
    "light_grey": "#E7ECF3",
}


@dataclass(frozen=True)
class Task:
    chapter: str
    label: str
    start: date
    end: date
    color: str


TASKS = [
    Task(
        "Chapter 1",
        "Finish analysis",
        date(2026, 7, 9),
        date(2026, 8, 31),
        PALETTE["teal"],
    ),
    Task(
        "Chapter 1",
        "Write paper",
        date(2026, 9, 1),
        date(2026, 9, 30),
        PALETTE["gold"],
    ),
    Task(
        "Chapter 1",
        "Submit paper",
        date(2026, 10, 1),
        date(2026, 10, 31),
        PALETTE["coral"],
    ),
    Task(
        "Chapter 2",
        "Multiangular sugar beet",
        date(2026, 11, 1),
        date(2027, 1, 15),
        PALETTE["blue"],
    ),
    Task(
        "Chapter 2",
        "Landscape Cercospora prognosis in Europe",
        date(2026, 12, 1),
        date(2027, 3, 15),
        PALETTE["teal"],
    ),
    Task(
        "Chapter 2",
        "DL 3D field reconstruction",
        date(2027, 1, 15),
        date(2027, 4, 30),
        PALETTE["gold"],
    ),
]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"plot_phd_gantt_timeline_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logging.info("Log file: %s", log_path)
    return log_path


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - started)


def add_month_bands(ax, start: date, end: date) -> None:
    month = date(start.year, start.month, 1)
    index = 0
    while month <= end:
        next_month = date(month.year + (month.month == 12), 1 if month.month == 12 else month.month + 1, 1)
        if index % 2 == 0:
            ax.axvspan(month, next_month, color=PALETTE["light_grey"], alpha=0.45, zorder=0)
        month = next_month
        index += 1


def draw_gantt() -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    x_start = date(2026, 7, 1)
    x_end = date(2027, 4, 30)
    add_month_bands(ax, x_start, x_end)

    y_positions = list(range(len(TASKS)))[::-1]
    for y, task in zip(y_positions, TASKS, strict=True):
        start_num = mdates.date2num(task.start)
        duration = mdates.date2num(task.end) - start_num + 1
        ax.barh(
            y,
            duration,
            left=start_num,
            height=0.62,
            color=task.color,
            edgecolor="none",
            zorder=3,
        )
        ax.text(
            start_num + duration / 2,
            y,
            task.label,
            ha="center",
            va="center",
            fontsize=12.8,
            fontweight="bold",
            color="white" if task.color in {PALETTE["teal"], PALETTE["blue"], PALETTE["coral"]} else PALETTE["navy"],
            zorder=4,
        )

    ax.axvline(date(2026, 7, 9), color=PALETTE["navy"], linewidth=1.6, linestyle="--", alpha=0.85, zorder=2)
    ax.text(
        date(2026, 7, 9),
        len(TASKS) - 0.35,
        "Now",
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    chapter_y = {
        "Chapter 1": (y_positions[0] + y_positions[2]) / 2,
        "Chapter 2": (y_positions[3] + y_positions[5]) / 2,
    }
    for chapter, y in chapter_y.items():
        ax.text(
            mdates.date2num(x_start) - 10,
            y,
            chapter,
            ha="right",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=PALETTE["navy"],
        )

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(-0.8, len(TASKS) - 0.2)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", labelsize=11, colors=PALETTE["grey"], length=0, pad=10)
    ax.grid(axis="x", color="#C8D2DF", linewidth=0.8, alpha=0.65, zorder=1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "PhD Workplan Timeline",
        fontsize=24,
        fontweight="bold",
        color=PALETTE["navy"],
        pad=18,
    )
    ax.text(
        0.5,
        1.005,
        "Chapter 1 paper through October submission; Chapter 2 exploratory work over roughly six months",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        color=PALETTE["grey"],
    )

    legend_items = [
        Patch(facecolor=PALETTE["teal"], label="Analysis / modelling"),
        Patch(facecolor=PALETTE["gold"], label="Writing / synthesis"),
        Patch(facecolor=PALETTE["coral"], label="Submission"),
        Patch(facecolor=PALETTE["blue"], label="New chapter work"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        labelcolor=PALETTE["grey"],
    )

    fig.subplots_adjust(left=0.18, right=0.98, top=0.84, bottom=0.20)

    paths = [
        FIGURES_DIR / "phd_gantt_timeline.png",
        FIGURES_DIR / "phd_gantt_timeline.pdf",
        FIGURES_DIR / "phd_gantt_timeline.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, transparent=path.suffix == ".png", facecolor="none" if path.suffix == ".png" else "white")
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("draw Gantt chart", started)
    return paths


def write_report(paths: list[Path], log_path: Path) -> Path:
    started = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"| {task.chapter} | {task.label} | {task.start.isoformat()} | {task.end.isoformat()} |"
        for task in TASKS
    )
    report = f"""## PhD Gantt Timeline

| Chapter | Task | Start | End |
| --- | --- | --- | --- |
{rows}

**Interpretation**: Chapter 1 is planned around finishing analysis by the end of August 2026, writing in September 2026, and submission in October 2026. Chapter 2 then allocates roughly six months to multiangular sugar beet work, landscape-scale Cercospora prognosis in Europe, and deep-learning 3D field reconstruction.

**Outputs**:
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in paths)}

**Reproducibility**:

- Source script: `scripts/plotting/plot_phd_gantt_timeline.py`
- Current-date marker: `2026-07-09`
- Timeline range: `2026-07-01` to `2027-04-30`
- Log: `{log_path.relative_to(ROOT)}`
"""
    path = REPORTS_DIR / "phd_gantt_timeline_summary.md"
    path.write_text(report, encoding="utf-8")
    logging.info("Wrote report: %s", path)
    log_phase("write report", started)
    return path


def main() -> None:
    log_path = setup_logging()
    paths = draw_gantt()
    write_report(paths, log_path)


if __name__ == "__main__":
    main()
