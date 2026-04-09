#!/usr/bin/env python3
"""Render Flow emission.csv trajectories into a clear 2D top-down animation."""

import argparse
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.lines import Line2D


REQUIRED_COLUMNS = ("time", "id", "x", "y")
OPTIONAL_COLUMNS = ("lane_number",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render emission.csv trajectories to MP4 or GIF."
    )
    parser.add_argument("csv_path", type=str, help="Path to emission.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="simulation_replay.mp4",
        help="Output path (.mp4 or .gif). Default: simulation_replay.mp4",
    )
    parser.add_argument("--fps", type=int, default=20, help="Frames per second.")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    parser.add_argument(
        "--marker-size",
        type=float,
        default=40.0,
        help="Vehicle marker size in points^2.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every N-th timestamp frame (>=1) to speed up rendering.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(12.0, 6.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches, e.g. --figsize 14 7",
    )
    parser.add_argument(
        "--x-window",
        type=float,
        default=220.0,
        help=(
            "Dynamic x-axis window size in meters. "
            "Set <=0 to show full trajectory range."
        ),
    )
    parser.add_argument(
        "--aspect",
        choices=("auto", "equal"),
        default="auto",
        help="Axis aspect ratio. 'auto' is better for highway lane visibility.",
    )
    parser.add_argument(
        "--y-margin",
        type=float,
        default=1.0,
        help="Extra y margin in meters around lane boundaries.",
    )
    parser.add_argument(
        "--title-fontsize",
        type=float,
        default=9.0,
        help="Title font size.",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=9.0,
        help="Axis label/tick font size.",
    )
    parser.add_argument(
        "--hide-lane-lines",
        action="store_true",
        help="Disable horizontal lane boundary lines.",
    )
    return parser.parse_args()


def id_to_color(vehicle_id: str) -> str:
    vehicle_id = str(vehicle_id).lower()
    if "ego" in vehicle_id:
        return "green"
    if "llm" in vehicle_id:
        return "red"
    return "gray"


def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing_columns:
        raise ValueError(
            "CSV missing required columns: "
            + ", ".join(missing_columns)
            + f". Expected at least: {', '.join(REQUIRED_COLUMNS)}"
        )

    keep_columns = list(REQUIRED_COLUMNS) + [
        col for col in OPTIONAL_COLUMNS if col in raw_df.columns
    ]
    df = raw_df[keep_columns].copy()
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(str)

    for col in ("time", "x", "y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "lane_number" in df.columns:
        df["lane_number"] = pd.to_numeric(df["lane_number"], errors="coerce")

    df = df.dropna(subset=["time", "x", "y", "id"])
    if df.empty:
        raise ValueError("No valid trajectory rows after cleaning.")

    # Keep only one position per (time, id) to avoid rendering ambiguity.
    df = df.sort_values(["time", "id"]).drop_duplicates(["time", "id"], keep="last")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def build_frame_data(df: pd.DataFrame, frame_step: int):
    times = np.sort(df["time"].unique())
    times = times[::frame_step]

    grouped = dict(tuple(df.groupby("time", sort=True)))
    frames = []
    for t in times:
        frame_df = grouped[t]
        xs = frame_df["x"].to_numpy(dtype=float)
        ys = frame_df["y"].to_numpy(dtype=float)
        ids = frame_df["id"].tolist()
        colors = [id_to_color(v_id) for v_id in ids]
        frames.append((float(t), xs, ys, ids, colors))
    return frames


def infer_lane_layout(df: pd.DataFrame):
    if "lane_number" not in df.columns:
        return None

    lane_df = df.dropna(subset=["lane_number", "y"]).copy()
    if lane_df.empty:
        return None

    lane_df["lane_number"] = lane_df["lane_number"].astype(int)
    centers = lane_df.groupby("lane_number")["y"].median().sort_index()
    if centers.empty:
        return None

    lane_ids = centers.index.to_list()
    lane_centers = centers.to_numpy(dtype=float)
    if len(lane_centers) >= 2:
        lane_width = float(np.median(np.diff(np.sort(lane_centers))))
        if lane_width <= 0:
            lane_width = 3.5
    else:
        lane_width = 3.5

    boundaries = [lane_centers[0] - lane_width / 2.0]
    if len(lane_centers) >= 2:
        boundaries.extend(((lane_centers[:-1] + lane_centers[1:]) / 2.0).tolist())
    boundaries.append(lane_centers[-1] + lane_width / 2.0)

    return {
        "lane_ids": lane_ids,
        "centers": lane_centers,
        "boundaries": np.array(boundaries, dtype=float),
    }


def select_focus_x(ids, xs):
    for keyword in ("ego", "llm"):
        for i, vehicle_id in enumerate(ids):
            if keyword in vehicle_id.lower():
                return float(xs[i])
    if xs.size:
        return float(np.median(xs))
    return 0.0


def create_animation(frames, df: pd.DataFrame, args: argparse.Namespace, lane_layout):
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    x_min_data = float(df["x"].min())
    x_max_data = float(df["x"].max())
    x_span = max(x_max_data - x_min_data, 1.0)
    x_min = x_min_data - 0.03 * x_span
    x_max = x_max_data + 0.03 * x_span

    if lane_layout is not None:
        boundaries = lane_layout["boundaries"]
        y_min = float(boundaries.min() - args.y_margin)
        y_max = float(boundaries.max() + args.y_margin)
    else:
        y_min_data = float(df["y"].min())
        y_max_data = float(df["y"].max())
        y_span = max(y_max_data - y_min_data, 1.0)
        y_min = y_min_data - 0.2 * y_span
        y_max = y_max_data + 0.2 * y_span

    ax.set_ylim(y_min, y_max)
    ax.set_aspect(args.aspect, adjustable="box")
    ax.set_facecolor("#f7f7f7")
    ax.set_xlabel("x (m)", fontsize=args.label_fontsize)
    ax.set_ylabel("y (m)", fontsize=args.label_fontsize)
    ax.tick_params(axis="both", labelsize=max(args.label_fontsize - 1.0, 6.0))
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    lane_lines = []
    if lane_layout is not None and not args.hide_lane_lines:
        boundaries = lane_layout["boundaries"]
        for idx, y in enumerate(boundaries):
            is_outer = idx in (0, len(boundaries) - 1)
            line = ax.plot(
                [x_min, x_max],
                [float(y), float(y)],
                color="#a0a0a0" if is_outer else "#c6c6c6",
                linewidth=1.4 if is_outer else 1.0,
                linestyle="-" if is_outer else "--",
                zorder=1,
            )[0]
            lane_lines.append(line)

    scatter = ax.scatter(
        [],
        [],
        s=args.marker_size,
        marker="s",
        edgecolors="black",
        linewidths=0.25,
        alpha=0.95,
        zorder=3,
    )
    title = ax.set_title("Simulation Replay", fontsize=args.title_fontsize)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="ego",
            markerfacecolor="green",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="llm",
            markerfacecolor="red",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="background",
            markerfacecolor="gray",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=7,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        framealpha=0.9,
        facecolor="white",
        fontsize=max(args.label_fontsize - 1.0, 7.0),
        title="Vehicle Type",
        title_fontsize=max(args.label_fontsize - 0.5, 7.0),
    )

    def apply_xlim(left: float, right: float):
        ax.set_xlim(left, right)
        if lane_lines:
            for line in lane_lines:
                line.set_xdata([left, right])

    def init():
        if args.x_window > 0:
            _, init_xs, _, init_ids, _ = frames[0]
            center = select_focus_x(init_ids, init_xs)
            half = args.x_window / 2.0
            left, right = center - half, center + half
            if x_max - x_min > args.x_window:
                left = max(left, x_min)
                right = min(right, x_max)
                if right - left < args.x_window:
                    if left <= x_min:
                        right = min(x_min + args.x_window, x_max)
                    else:
                        left = max(x_max - args.x_window, x_min)
            apply_xlim(left, right)
        else:
            apply_xlim(x_min, x_max)

        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_color([])
        title.set_text("Simulation Replay")
        artists = [scatter, title]
        artists.extend(lane_lines)
        return tuple(artists)

    def update(frame):
        t, xs, ys, ids, colors = frame
        if args.x_window > 0 and xs.size:
            center = select_focus_x(ids, xs)
            half = args.x_window / 2.0
            left, right = center - half, center + half
            if x_max - x_min > args.x_window:
                left = max(left, x_min)
                right = min(right, x_max)
                if right - left < args.x_window:
                    if left <= x_min:
                        right = min(x_min + args.x_window, x_max)
                    else:
                        left = max(x_max - args.x_window, x_min)
            apply_xlim(left, right)
        elif args.x_window <= 0:
            apply_xlim(x_min, x_max)

        offsets = np.column_stack((xs, ys)) if xs.size else np.empty((0, 2))
        scatter.set_offsets(offsets)
        scatter.set_color(colors)
        title.set_text(f"Replay | t={t:.2f}s | vehicles={len(xs)}")
        artists = [scatter, title]
        artists.extend(lane_lines)
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1000.0 / args.fps,
        blit=False,
        repeat=False,
    )
    return fig, anim


def pick_mp4_codec() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return "libx264"

    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    encoders = f"{result.stdout}\n{result.stderr}"
    if "libx264" in encoders:
        return "libx264"
    if "mpeg4" in encoders:
        return "mpeg4"
    return "libx264"


def save_animation(anim, output_path: str, fps: int, dpi: int):
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".gif":
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
        return "gif"

    if ext not in (".mp4", ".m4v", ".mov"):
        raise ValueError(f"Unsupported output extension: {ext}. Use .mp4 or .gif.")
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is required for MP4 output but was not found. "
            "Install ffmpeg or use --output simulation_replay.gif"
        )

    codec = pick_mp4_codec()
    writer = animation.FFMpegWriter(
        fps=fps,
        codec=codec,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(output_path, writer=writer, dpi=dpi)
    return codec


def main() -> int:
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1")
    if args.marker_size <= 0:
        raise ValueError("--marker-size must be > 0")

    df = load_csv(args.csv_path)
    frames = build_frame_data(df, args.frame_step)
    if not frames:
        raise ValueError("No frames generated from CSV.")

    lane_layout = infer_lane_layout(df)
    fig, anim = create_animation(frames=frames, df=df, args=args, lane_layout=lane_layout)

    try:
        codec = save_animation(anim, args.output, fps=args.fps, dpi=args.dpi)
    finally:
        plt.close(fig)

    if lane_layout is not None:
        print(f"Detected lanes: {len(lane_layout['lane_ids'])}")
    print(f"Saved animation to: {os.path.abspath(args.output)}")
    if codec != "gif":
        print(f"MP4 codec: {codec}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[render_video] Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
