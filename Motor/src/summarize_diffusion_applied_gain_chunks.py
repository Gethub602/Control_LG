import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import RESULTS_DIR  # noqa: E402


KAFKA_CONTROL_DIR = RESULTS_DIR / "kafka_control"
SUMMARY_DIR = RESULTS_DIR / "summary"
FIGURE_DIR = RESULTS_DIR / "figures" / "diffusion_gain_chunks"


SCENARIO_RE = re.compile(
    r"local_kafka_controller_log_esp32_delay_aware_"
    r"(?P<label>diffusion_unet_ddim20_actual_(?P<scenario>.+?)_rep(?P<rep>\d+))_"
    r"\d{8}_\d{6}\.csv$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize actually applied diffusion gain chunk values."
    )
    parser.add_argument(
        "--pattern",
        default="local_kafka_controller_log_esp32_delay_aware_diffusion_unet_ddim20_actual_*_rep1_*.csv",
    )
    parser.add_argument("--run-label", default="diffusion_unet_ddim20_actual")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def parse_log_name(path: Path):
    match = SCENARIO_RE.match(path.name)
    if not match:
        return {
            "run_label": path.stem,
            "scenario_id": path.stem,
            "repeat": np.nan,
        }
    return {
        "run_label": match.group("label"),
        "scenario_id": match.group("scenario"),
        "repeat": int(match.group("rep")),
    }


def load_logs(pattern: str):
    rows = []
    for path in sorted(KAFKA_CONTROL_DIR.glob(pattern), key=lambda p: p.stat().st_mtime):
        meta = parse_log_name(path)
        df = pd.read_csv(path)
        for key, value in meta.items():
            df[key] = value
        df["log_file"] = str(path)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No logs found: {KAFKA_CONTROL_DIR / pattern}")
    return pd.concat(rows, ignore_index=True)


def make_step_table(df: pd.DataFrame):
    sched = df[df["schedule_source"].astype(str) == "schedule_chunk"].copy()
    if sched.empty:
        return sched
    sched["schedule_chunk_short"] = sched["schedule_chunk_id"].astype(str).str[:8]
    keep = [
        "scenario_id",
        "repeat",
        "run_id",
        "time",
        "step",
        "target",
        "current",
        "error",
        "pwm",
        "raw_pwm",
        "schedule_chunk_id",
        "schedule_chunk_short",
        "schedule_chunk_index",
        "schedule_chunk_age_sec",
        "schedule_source_to_accept_sec",
        "schedule_source_to_apply_sec",
        "schedule_generator_duration_sec",
        "schedule_timing_slack_sec",
        "base_kp",
        "base_ki",
        "base_kd",
        "kp",
        "ki",
        "kd",
        "gain_update_reason",
        "log_file",
    ]
    return sched[[col for col in keep if col in sched.columns]].sort_values(
        ["scenario_id", "repeat", "time", "schedule_chunk_index"]
    )


def seq_json(values):
    return json.dumps([round(float(v), 6) for v in values], ensure_ascii=True)


def make_chunk_summary(step_df: pd.DataFrame):
    if step_df.empty:
        return step_df
    rows = []
    group_cols = ["scenario_id", "repeat", "schedule_chunk_id"]
    for keys, group in step_df.groupby(group_cols, dropna=False):
        scenario_id, repeat, chunk_id = keys
        group = group.sort_values(["time", "schedule_chunk_index"])
        targets = group["target"].dropna().unique().tolist()
        row = {
            "scenario_id": scenario_id,
            "repeat": repeat,
            "schedule_chunk_id": chunk_id,
            "schedule_chunk_short": str(chunk_id)[:8],
            "target_values": seq_json(targets),
            "applied_steps": int(len(group)),
            "time_start": float(group["time"].iloc[0]),
            "time_end": float(group["time"].iloc[-1]),
            "chunk_index_min": int(group["schedule_chunk_index"].min()),
            "chunk_index_max": int(group["schedule_chunk_index"].max()),
            "chunk_indices": seq_json(group["schedule_chunk_index"].to_numpy()),
            "kp_first": float(group["kp"].iloc[0]),
            "kp_last": float(group["kp"].iloc[-1]),
            "kp_mean": float(group["kp"].mean()),
            "kp_min": float(group["kp"].min()),
            "kp_max": float(group["kp"].max()),
            "ki_first": float(group["ki"].iloc[0]),
            "ki_last": float(group["ki"].iloc[-1]),
            "ki_mean": float(group["ki"].mean()),
            "ki_min": float(group["ki"].min()),
            "ki_max": float(group["ki"].max()),
            "kd_first": float(group["kd"].iloc[0]),
            "kd_last": float(group["kd"].iloc[-1]),
            "kd_mean": float(group["kd"].mean()),
            "kd_min": float(group["kd"].min()),
            "kd_max": float(group["kd"].max()),
            "kp_seq_applied": seq_json(group["kp"].to_numpy()),
            "ki_seq_applied": seq_json(group["ki"].to_numpy()),
            "kd_seq_applied": seq_json(group["kd"].to_numpy()),
            "pwm_mean_during_chunk": float(group["pwm"].mean()),
            "pwm_max_during_chunk": float(group["pwm"].max()),
            "error_abs_mean_during_chunk": float(group["error"].abs().mean()),
            "source_to_apply_p90": float(group["schedule_source_to_apply_sec"].quantile(0.9)),
            "generator_duration_mean": float(group["schedule_generator_duration_sec"].mean()),
            "timing_slack_min": float(group["schedule_timing_slack_sec"].min()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario_id", "repeat", "time_start"])


def make_scenario_summary(step_df: pd.DataFrame, all_df: pd.DataFrame):
    rows = []
    for scenario_id, group in all_df.groupby("scenario_id"):
        sched = step_df[step_df["scenario_id"] == scenario_id]
        row = {
            "scenario_id": scenario_id,
            "target_profile": "->".join(
                f"{value:g}" for value in group["target"].drop_duplicates().tolist()
            ),
            "total_steps": int(len(group)),
            "schedule_steps": int(len(sched)),
            "schedule_step_ratio": float(len(sched) / max(len(group), 1)),
            "unique_chunks_applied": int(sched["schedule_chunk_id"].nunique()) if not sched.empty else 0,
        }
        for col in ["kp", "ki", "kd"]:
            if sched.empty:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_min"] = np.nan
                row[f"{col}_max"] = np.nan
                row[f"{col}_std"] = np.nan
            else:
                row[f"{col}_mean"] = float(sched[col].mean())
                row[f"{col}_min"] = float(sched[col].min())
                row[f"{col}_max"] = float(sched[col].max())
                row[f"{col}_std"] = float(sched[col].std(ddof=0))
        if sched.empty:
            row["pwm_mean_on_schedule_steps"] = np.nan
            row["pwm_max_on_schedule_steps"] = np.nan
            row["generator_duration_p90"] = np.nan
        else:
            row["pwm_mean_on_schedule_steps"] = float(sched["pwm"].mean())
            row["pwm_max_on_schedule_steps"] = float(sched["pwm"].max())
            row["generator_duration_p90"] = float(
                sched["schedule_generator_duration_sec"].quantile(0.9)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scenario_id")


def plot_scenarios(all_df: pd.DataFrame, step_df: pd.DataFrame, timestamp: str):
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for scenario_id, group in all_df.groupby("scenario_id"):
        sched = step_df[step_df["scenario_id"] == scenario_id]
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(group["time"], group["target"], "k--", label="target")
        axes[0].plot(group["time"], group["current"], label="rpm")
        axes[0].set_ylabel("RPM")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(group["time"], group["pwm"], color="tab:blue")
        axes[1].set_ylabel("PWM")
        axes[1].grid(True, alpha=0.25)

        for ax, col, color in zip(
            axes[2:],
            ["kp", "ki"],
            ["tab:orange", "tab:green"],
        ):
            ax.plot(group["time"], group[col], color="0.75", linewidth=1, label=f"{col} all")
            if not sched.empty:
                ax.scatter(sched["time"], sched[col], s=14, color=color, label=f"{col} schedule")
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

        ax = axes[3].twinx()
        ax.plot(group["time"], group["kd"], color="0.75", linewidth=1)
        if not sched.empty:
            ax.scatter(sched["time"], sched["kd"], s=14, color="tab:red", label="kd schedule")
        ax.set_ylabel("kd")
        ax.legend(loc="upper right")
        axes[3].set_xlabel("time [s]")
        fig.suptitle(f"Diffusion DDIM20 Applied Gain Chunks: {scenario_id}")
        fig.tight_layout()
        path = FIGURE_DIR / f"diffusion_ddim20_applied_gain_{scenario_id}_{timestamp}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_df = load_logs(args.pattern)
    step_df = make_step_table(all_df)
    chunk_summary = make_chunk_summary(step_df)
    scenario_summary = make_scenario_summary(step_df, all_df)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    step_path = SUMMARY_DIR / f"{args.run_label}_applied_gain_steps_{timestamp}.csv"
    chunk_path = SUMMARY_DIR / f"{args.run_label}_applied_chunk_summary_{timestamp}.csv"
    scenario_path = SUMMARY_DIR / f"{args.run_label}_applied_gain_scenario_summary_{timestamp}.csv"
    step_df.to_csv(step_path, index=False, encoding="utf-8-sig")
    chunk_summary.to_csv(chunk_path, index=False, encoding="utf-8-sig")
    scenario_summary.to_csv(scenario_path, index=False, encoding="utf-8-sig")

    figure_paths = []
    if args.plot:
        figure_paths = plot_scenarios(all_df, step_df, timestamp)

    print(f"Saved step table: {step_path}")
    print(f"Saved chunk summary: {chunk_path}")
    print(f"Saved scenario summary: {scenario_path}")
    if figure_paths:
        print("Saved figures:")
        for path in figure_paths:
            print(f"  {path}")
    print(scenario_summary.to_string(index=False))


if __name__ == "__main__":
    main()
