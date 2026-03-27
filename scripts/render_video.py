#!/usr/bin/env python3
"""Render Flow emission.csv trajectories into a 2D top-down animation."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation


REQUIRED_COLUMNS = ("time", "id", "x", "y")


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
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI.")
    parser.add_argument(
        "--marker-size",
        type=float,
        default=24.0,
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
        default=(8.0, 6.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches, e.g. --figsize 10 7",
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

    df = pd.read_csv(csv_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "CSV missing required columns: "
            + ", ".join(missing_columns)
            + f". Expected at least: {', '.join(REQUIRED_COLUMNS)}"
        )

    df = df[list(REQUIRED_COLUMNS)].copy()
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(str)
    for col in ("time", "x", "y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
        colors = [id_to_color(v_id) for v_id in frame_df["id"].tolist()]
        frames.append((float(t), xs, ys, colors))
    return frames


def create_animation(
    frames, x_min: float, x_max: float, y_min: float, y_max: float, fps: int, marker_size: float, figsize
):
    fig, ax = plt.subplots(figsize=tuple(figsize))

    x_pad = max((x_max - x_min) * 0.05, 1.0)
    y_pad = max((y_max - y_min) * 0.05, 1.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)

    scatter = ax.scatter([], [], s=marker_size)
    title = ax.set_title("Simulation Replay")

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_color([])
        title.set_text("Simulation Replay")
        return scatter, title

    def update(frame):
        t, xs, ys, colors = frame
        offsets = np.column_stack((xs, ys)) if xs.size else np.empty((0, 2))
        scatter.set_offsets(offsets)
        scatter.set_color(colors)
        title.set_text(f"Simulation Replay | t={t:.2f}")
        return scatter, title

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1000.0 / fps,
        blit=True,
        repeat=False,
    )
    return fig, anim


def save_animation(anim, output_path: str, fps: int, dpi: int):
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        if ext not in (".mp4", ".m4v", ".mov"):
            raise ValueError(
                f"Unsupported output extension: {ext}. Use .mp4 or .gif."
            )
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "FFmpeg is required for MP4 output but was not found. "
                "Install ffmpeg or use --output simulation_replay.gif"
            )
        writer = animation.FFMpegWriter(fps=fps, codec="libx264")

    anim.save(output_path, writer=writer, dpi=dpi)


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

    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    fig, anim = create_animation(
        frames=frames,
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        fps=args.fps,
        marker_size=args.marker_size,
        figsize=args.figsize,
    )

    try:
        save_animation(anim, args.output, fps=args.fps, dpi=args.dpi)
    finally:
        plt.close(fig)

    print(f"Saved animation to: {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[render_video] Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
