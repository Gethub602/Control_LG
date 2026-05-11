import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.append(str(ROOT_DIR))

from config import FIGURE_DIR, RESULTS_DIR


KAFKA_DIR = RESULTS_DIR / "kafka_control"
SUMMARY_DIR = RESULTS_DIR / "summary"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot DB/RF Kafka schedule chunk gain changes."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario label in filenames, e.g. seq_70_85_100.",
    )
    parser.add_argument(
        "--run-prefix",
        default="",
        help="Optional prefix before method in filenames, e.g. seg5.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of latest DB/RF repetitions to include.",
    )
    return parser.parse_args()


def find_logs(scenario: str, method: str, reps: int, run_prefix: str = ""):
    method_label = f"{run_prefix}_{method}" if run_prefix else method
    pattern = (
        f"local_kafka_controller_log_esp32_delay_aware_"
        f"{method_label}_{scenario}_rep*.csv"
    )
    paths = sorted(KAFKA_DIR.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No logs found for pattern: {pattern}")
    return paths[-reps:]


def load_logs(paths, method: str):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["method"] = method.upper()
        df["source_file"] = path.name
        match = pd.Series(path.name).str.extract(r"rep(\d+)").iloc[0, 0]
        df["rep"] = int(match) if pd.notna(match) else len(frames) + 1
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_target_changes(ax, df: pd.DataFrame):
    target_changes = df.loc[df["target"].ne(df["target"].shift()), "time"].tolist()
    for change_time in target_changes[1:]:
        ax.axvline(change_time, color="0.25", linestyle=":", linewidth=1.0, alpha=0.7)


def plot_repeated_lines(ax, df, y_col, label, color, linestyle="-", draw_target=False):
    reps = sorted(df["rep"].unique())
    for idx, rep in enumerate(reps):
        rep_df = df[df["rep"] == rep]
        line_label = label if idx == 0 else None
        ax.plot(
            rep_df["time"],
            rep_df[y_col],
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            alpha=0.55,
            label=line_label,
        )

    if draw_target:
        first = df[df["rep"] == reps[0]]
        ax.step(
            first["time"],
            first["target"],
            where="post",
            color="black",
            linestyle="--",
            linewidth=1.6,
            label="target",
        )


def build_gain_distribution(df: pd.DataFrame):
    sched = df[df.get("schedule_source", "") == "schedule_chunk"].copy()
    if sched.empty:
        return pd.DataFrame()

    grouped = (
        sched.groupby(["method", "rep", "base_kp", "base_ki", "base_kd"])
        .size()
        .reset_index(name="count")
        .sort_values(["method", "rep", "count"], ascending=[True, True, False])
    )
    grouped["gain"] = grouped.apply(
        lambda row: f"({row.base_kp:.3g}, {row.base_ki:.3g}, {row.base_kd:.3g})",
        axis=1,
    )
    return grouped


def plot_gain_distribution_bar(ax, gain_dist: pd.DataFrame, method: str, top_n: int = 8):
    method_df = gain_dist[gain_dist["method"] == method.upper()]
    if method_df.empty:
        ax.text(0.5, 0.5, f"No {method.upper()} schedule chunks", ha="center")
        return

    total = (
        method_df.groupby("gain")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()
    )
    ax.barh(total.index, total.values, color="#4C78A8" if method == "db" else "#F58518")
    ax.set_title(f"{method.upper()} top scheduled gains")
    ax.set_xlabel("applied steps across reps")
    ax.grid(True, axis="x", alpha=0.25)


def make_figure(db_df: pd.DataFrame, rf_df: pd.DataFrame, scenario: str, timestamp: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"Kafka Schedule Chunk Gain Changes: {scenario}", fontsize=15)

    plot_repeated_lines(
        axes[0], db_df, "current", "DB rpm", "#4C78A8", draw_target=True
    )
    plot_repeated_lines(axes[0], rf_df, "current", "RF rpm", "#F58518")
    axes[0].set_ylabel("RPM")
    axes[0].set_title("Motor response")

    plot_repeated_lines(axes[1], db_df, "pwm", "DB pwm", "#4C78A8")
    plot_repeated_lines(axes[1], rf_df, "pwm", "RF pwm", "#F58518")
    axes[1].set_ylabel("PWM")
    axes[1].set_title("PWM command")

    for ax, gain_col, title in zip(
        axes[2:],
        ["base_kp", "base_ki", "base_kd"],
        ["Scheduled Kp", "Scheduled Ki", "Scheduled Kd"],
    ):
        plot_repeated_lines(ax, db_df, gain_col, f"DB {gain_col}", "#4C78A8")
        plot_repeated_lines(ax, rf_df, gain_col, f"RF {gain_col}", "#F58518")
        ax.set_ylabel(gain_col)
        ax.set_title(title)

    for ax in axes:
        plot_target_changes(ax, db_df[db_df["rep"] == db_df["rep"].min()])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time [s]")
    save_path = FIGURE_DIR / f"kafka_gain_chunk_timeseries_{scenario}_{timestamp}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def make_distribution_figure(gain_dist: pd.DataFrame, scenario: str, timestamp: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Scheduled Gain Combination Frequency: {scenario}", fontsize=14)

    plot_gain_distribution_bar(axes[0], gain_dist, "db")
    plot_gain_distribution_bar(axes[1], gain_dist, "rf")

    save_path = FIGURE_DIR / f"kafka_gain_chunk_distribution_{scenario}_{timestamp}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    db_paths = find_logs(args.scenario, "db", args.reps, args.run_prefix)
    rf_paths = find_logs(args.scenario, "rf", args.reps, args.run_prefix)

    db_df = load_logs(db_paths, "db")
    rf_df = load_logs(rf_paths, "rf")
    all_df = pd.concat([db_df, rf_df], ignore_index=True)

    time_plot = make_figure(db_df, rf_df, args.scenario, timestamp)

    gain_dist = build_gain_distribution(all_df)
    dist_csv = SUMMARY_DIR / f"kafka_gain_chunk_distribution_{args.scenario}_{timestamp}.csv"
    gain_dist.to_csv(dist_csv, index=False, encoding="utf-8-sig")
    dist_plot = make_distribution_figure(gain_dist, args.scenario, timestamp)

    print(f"DB logs: {[path.name for path in db_paths]}")
    print(f"RF logs: {[path.name for path in rf_paths]}")
    print(f"Saved: {time_plot}")
    print(f"Saved: {dist_plot}")
    print(f"Saved: {dist_csv}")


if __name__ == "__main__":
    main()
