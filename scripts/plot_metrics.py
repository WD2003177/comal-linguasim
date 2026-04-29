#!/usr/bin/env python3
"""Plot speed, acceleration, and TTC curves from a Flow emission.csv file."""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("time", "id", "speed")
OPTIONAL_COLUMNS = ("realized_accel", "headway", "leader_rel_speed", "leader_id")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ego/LLM speed, acceleration, and TTC from emission.csv."
    )
    parser.add_argument("csv_path", help="Path to Flow emission.csv")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for output PNG files. Default: CSV directory.",
    )
    parser.add_argument(
        "--prefix",
        default="metrics",
        help="Output filename prefix. Default: metrics",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI. Default: 180",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(10.0, 4.8),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches. Default: 10 4.8",
    )
    parser.add_argument(
        "--ttc-max",
        type=float,
        default=10.0,
        help="Clip TTC y-axis to this max in seconds. Default: 10",
    )
    parser.add_argument(
        "--hard-brake",
        type=float,
        default=-2.5,
        help="Hard-brake reference line for acceleration plot. Default: -2.5",
    )
    return parser.parse_args()


def vehicle_color(vehicle_id):
    lowered = str(vehicle_id).lower()
    if "ego" in lowered:
        return "green"
    if "llm_0" in lowered:
        return "red"
    if "llm_1" in lowered:
        return "darkorange"
    if "llm" in lowered:
        return "crimson"
    return "gray"


def vehicle_line_style(vehicle_id):
    lowered = str(vehicle_id).lower()
    if "ego" in lowered:
        return "-"
    if "llm_0" in lowered:
        return "-"
    if "llm_1" in lowered:
        return "--"
    return "-."


def load_metrics(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("CSV file not found: {}".format(csv_path))

    raw = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(
            "CSV missing required columns: {}. Expected at least: {}".format(
                ", ".join(missing), ", ".join(REQUIRED_COLUMNS)
            )
        )

    keep = list(REQUIRED_COLUMNS) + [col for col in OPTIONAL_COLUMNS if col in raw.columns]
    df = raw[keep].copy()
    df = df.dropna(subset=["time", "id"])
    df["id"] = df["id"].astype(str)
    for col in keep:
        if col != "id" and col != "leader_id":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "leader_id" in df.columns:
        df["leader_id"] = df["leader_id"].fillna("").astype(str)

    df = df.dropna(subset=["time", "speed"])
    df = df.sort_values(["id", "time"]).drop_duplicates(["id", "time"], keep="last")
    df = df.sort_values(["time", "id"]).reset_index(drop=True)

    focus_ids = sorted(
        veh_id for veh_id in df["id"].unique()
        if "ego" in veh_id.lower() or "llm" in veh_id.lower()
    )
    if not focus_ids:
        raise ValueError("No ego/llm vehicles found in CSV id column.")

    df = df[df["id"].isin(focus_ids)].copy()
    df["time"] = df["time"] - float(df["time"].min())
    return df, focus_ids


def add_acceleration(df):
    if "realized_accel" in df.columns and df["realized_accel"].notna().any():
        df["plot_accel"] = pd.to_numeric(df["realized_accel"], errors="coerce")
        return df, "realized_accel"

    pieces = []
    for _, group in df.groupby("id", sort=False):
        group = group.sort_values("time").copy()
        dt = group["time"].diff()
        dv = group["speed"].diff()
        group["plot_accel"] = dv / dt.replace(0, np.nan)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True), "diff(speed)"


def add_ttc(df):
    if not {"headway", "leader_rel_speed"}.issubset(df.columns):
        df["ttc"] = np.nan
        return df, "missing headway/leader_rel_speed"

    headway = pd.to_numeric(df["headway"], errors="coerce")
    leader_rel_speed = pd.to_numeric(df["leader_rel_speed"], errors="coerce")

    # Flow/SUMO leader_rel_speed is usually leader_speed - self_speed.
    # TTC is meaningful only when self is closing in on the leader.
    closing_speed = -leader_rel_speed
    valid = (headway > 0) & (closing_speed > 1e-3)
    df["ttc"] = np.where(valid, headway / closing_speed, np.nan)
    return df, "headway / max(self_speed - leader_speed, 0)"


def setup_axis(ax, ylabel):
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.35)


def save_plot(fig, output_path, dpi):
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_series(df, focus_ids, column, ylabel, title, output_path, args):
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    for veh_id in focus_ids:
        group = df[df["id"] == veh_id].sort_values("time")
        if group.empty or column not in group:
            continue
        ax.plot(
            group["time"],
            group[column],
            label=veh_id,
            color=vehicle_color(veh_id),
            linestyle=vehicle_line_style(veh_id),
            linewidth=1.8 if "ego" in veh_id.lower() else 1.5,
        )
    setup_axis(ax, ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    save_plot(fig, output_path, args.dpi)


def plot_acceleration(df, focus_ids, output_path, args, accel_source):
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    for veh_id in focus_ids:
        group = df[df["id"] == veh_id].sort_values("time")
        if group.empty:
            continue
        ax.plot(
            group["time"],
            group["plot_accel"],
            label=veh_id,
            color=vehicle_color(veh_id),
            linestyle=vehicle_line_style(veh_id),
            linewidth=1.8 if "ego" in veh_id.lower() else 1.5,
        )
    ax.axhline(args.hard_brake, color="black", linestyle=":", linewidth=1.0, label="hard brake")
    setup_axis(ax, "acceleration (m/s^2)")
    ax.set_title("Acceleration ({})".format(accel_source))
    ax.legend(loc="best", fontsize=8)
    save_plot(fig, output_path, args.dpi)


def plot_ttc(df, focus_ids, output_path, args, ttc_source):
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    plotted = False
    for veh_id in focus_ids:
        group = df[df["id"] == veh_id].sort_values("time")
        if group.empty:
            continue
        series = group["ttc"].clip(upper=args.ttc_max)
        if series.notna().any():
            plotted = True
        ax.plot(
            group["time"],
            series,
            label=veh_id,
            color=vehicle_color(veh_id),
            linestyle=vehicle_line_style(veh_id),
            linewidth=1.8 if "ego" in veh_id.lower() else 1.5,
        )
    ax.axhline(3.0, color="black", linestyle=":", linewidth=1.0, label="TTC=3s")
    ax.set_ylim(0.0, args.ttc_max)
    setup_axis(ax, "TTC (s, clipped)")
    ax.set_title("Time-to-Collision ({})".format(ttc_source))
    ax.legend(loc="best", fontsize=8)
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No closing-leader TTC samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    save_plot(fig, output_path, args.dpi)


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv_path)) or "."
    os.makedirs(output_dir, exist_ok=True)

    df, focus_ids = load_metrics(args.csv_path)
    df, accel_source = add_acceleration(df)
    df, ttc_source = add_ttc(df)

    outputs = {
        "speed": os.path.join(output_dir, "{}_speed.png".format(args.prefix)),
        "accel": os.path.join(output_dir, "{}_acceleration.png".format(args.prefix)),
        "ttc": os.path.join(output_dir, "{}_ttc.png".format(args.prefix)),
    }

    plot_series(
        df,
        focus_ids,
        column="speed",
        ylabel="speed (m/s)",
        title="Speed",
        output_path=outputs["speed"],
        args=args,
    )
    plot_acceleration(df, focus_ids, outputs["accel"], args, accel_source)
    plot_ttc(df, focus_ids, outputs["ttc"], args, ttc_source)

    print("Vehicles plotted: {}".format(", ".join(focus_ids)))
    print("Saved speed plot: {}".format(os.path.abspath(outputs["speed"])))
    print("Saved acceleration plot: {}".format(os.path.abspath(outputs["accel"])))
    print("Saved TTC plot: {}".format(os.path.abspath(outputs["ttc"])))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[plot_metrics] Error: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
