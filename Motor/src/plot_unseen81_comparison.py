import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
KAFKA_DIR = ROOT_DIR / "results" / "kafka_control"
FIGURE_DIR = ROOT_DIR / "results" / "figures"
SUMMARY_DIR = ROOT_DIR / "results" / "summary"


SCENARIO = "seq_70_81_100"
METHODS = ["db", "rf", "mlp", "mt_mlp"]
COLORS = {
    "db": "#4C78A8",
    "rf": "#F58518",
    "mlp": "#54A24B",
    "mt_mlp": "#B279A2",
}


def load_latest_logs():
    frames = []
    for method in METHODS:
        paths = sorted(
            KAFKA_DIR.glob(
                "local_kafka_controller_log_esp32_delay_aware_"
                f"unseen81_{method}_{SCENARIO}_rep*.csv"
            ),
            key=lambda path: path.stat().st_mtime,
        )[-3:]
        if len(paths) < 3:
            raise FileNotFoundError(f"Expected 3 logs for {method}, found {len(paths)}")
        for path in paths:
            df = pd.read_csv(path)
            match = re.search(r"rep(\d+)", path.name)
            df["method"] = method.upper()
            df["rep"] = int(match.group(1)) if match else 0
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_lines(ax, df, y_col, title, ylabel):
    for method in METHODS:
        method_df = df[df["method"] == method.upper()]
        for idx, rep in enumerate(sorted(method_df["rep"].unique())):
            rep_df = method_df[method_df["rep"] == rep]
            ax.plot(
                rep_df["time"],
                rep_df[y_col],
                color=COLORS[method],
                alpha=0.45,
                linewidth=1.3,
                label=method.upper() if idx == 0 else None,
            )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def make_timeseries(df, timestamp):
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    plot_lines(axes[0], df, "current", "Motor response", "RPM")
    first = df[(df["method"] == "DB") & (df["rep"] == sorted(df[df["method"] == "DB"]["rep"].unique())[0])]
    axes[0].step(
        first["time"],
        first["target"],
        where="post",
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="target",
    )
    axes[0].legend(loc="best")

    plot_lines(axes[1], df, "pwm", "PWM command", "PWM")
    plot_lines(axes[2], df, "base_kp", "Scheduled Kp", "Kp")
    plot_lines(axes[3], df, "base_ki", "Scheduled Ki", "Ki")
    plot_lines(axes[4], df, "base_kd", "Scheduled Kd", "Kd")

    for ax in axes:
        ax.axvline(5, color="0.25", linestyle=":", linewidth=1)
        ax.axvline(10, color="0.25", linestyle=":", linewidth=1)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Unseen target scenario: 70 -> 81 -> 100 RPM", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = FIGURE_DIR / f"unseen81_db_rf_mlp_timeseries_{timestamp}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def make_gain_distribution(df, timestamp):
    sched = df[df.get("schedule_source", "") == "schedule_chunk"].copy()
    gain_dist = (
        sched.groupby(["method", "base_kp", "base_ki", "base_kd"])
        .size()
        .reset_index(name="count")
    )
    gain_dist["gain"] = gain_dist.apply(
        lambda row: f"({row.base_kp:.3g}, {row.base_ki:.3g}, {row.base_kd:.3g})",
        axis=1,
    )

    csv_path = SUMMARY_DIR / f"unseen81_gain_distribution_{timestamp}.csv"
    gain_dist.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, method in zip(axes, METHODS):
        method_df = (
            gain_dist[gain_dist["method"] == method.upper()]
            .groupby("gain")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
            .sort_values()
        )
        ax.barh(method_df.index, method_df.values, color=COLORS[method])
        ax.set_title(f"{method.upper()} top gains")
        ax.set_xlabel("applied steps")
        ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Unseen 81 RPM scheduled gain distribution", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig_path = FIGURE_DIR / f"unseen81_gain_distribution_{timestamp}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path, csv_path


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = load_latest_logs()
    time_path = make_timeseries(df, timestamp)
    dist_path, csv_path = make_gain_distribution(df, timestamp)
    print(f"Saved: {time_path}")
    print(f"Saved: {dist_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
