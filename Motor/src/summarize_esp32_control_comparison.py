import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import RESULTS_DIR, FIGURE_DIR


# ============================================================
# Paths
# ============================================================

COMPARISON_DIR = RESULTS_DIR / "esp32_control_comparison"
KAFKA_DIR = RESULTS_DIR / "kafka_control"
SUMMARY_DIR = RESULTS_DIR / "summary"


# ============================================================
# Utility
# ============================================================

def get_latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No file found: {folder / pattern}")

    return files[-1]


def load_metrics():
    fixed_metric_path = get_latest_file(
        COMPARISON_DIR,
        "esp32_fixed_pid_metrics_*.csv",
    )

    local_db_metric_path = get_latest_file(
        COMPARISON_DIR,
        "esp32_local_gain_db_metrics_*.csv",
    )

    kafka_metric_path = get_latest_file(
        KAFKA_DIR,
        "local_kafka_controller_metrics_esp32_*.csv",
    )

    fixed_df = pd.read_csv(fixed_metric_path)
    local_db_df = pd.read_csv(local_db_metric_path)
    kafka_df = pd.read_csv(kafka_metric_path)

    fixed_df["case_name"] = "A_fixed_pid"
    local_db_df["case_name"] = "B_local_gain_db"
    kafka_df["case_name"] = "C_kafka_server_assisted"

    metrics_df = pd.concat(
        [fixed_df, local_db_df, kafka_df],
        ignore_index=True,
        sort=False,
    )

    return metrics_df, {
        "fixed_pid": fixed_metric_path,
        "local_gain_db": local_db_metric_path,
        "kafka_server_assisted": kafka_metric_path,
    }


def load_logs():
    fixed_log_path = get_latest_file(
        COMPARISON_DIR,
        "esp32_fixed_pid_log_*.csv",
    )

    local_db_log_path = get_latest_file(
        COMPARISON_DIR,
        "esp32_local_gain_db_log_*.csv",
    )

    kafka_log_path = get_latest_file(
        KAFKA_DIR,
        "local_kafka_controller_log_esp32_*.csv",
    )

    fixed_log = pd.read_csv(fixed_log_path)
    local_db_log = pd.read_csv(local_db_log_path)
    kafka_log = pd.read_csv(kafka_log_path)

    fixed_log["case_name"] = "A_fixed_pid"
    local_db_log["case_name"] = "B_local_gain_db"
    kafka_log["case_name"] = "C_kafka_server_assisted"

    return {
        "A_fixed_pid": fixed_log,
        "B_local_gain_db": local_db_log,
        "C_kafka_server_assisted": kafka_log,
    }, {
        "fixed_pid": fixed_log_path,
        "local_gain_db": local_db_log_path,
        "kafka_server_assisted": kafka_log_path,
    }


# ============================================================
# Plot
# ============================================================

def plot_metric_bar(metrics_df: pd.DataFrame, metric: str, ylabel: str, timestamp: str):
    if metric not in metrics_df.columns:
        print(f"Skip {metric}: column not found")
        return

    plt.figure(figsize=(8, 5))

    plot_df = metrics_df[["case_name", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

    plt.bar(plot_df["case_name"], plot_df[metric])

    plt.xlabel("Control mode")
    plt.ylabel(ylabel)
    plt.title(f"ESP32 Control Comparison: {metric}")
    plt.xticks(rotation=15)
    plt.grid(axis="y")

    save_path = FIGURE_DIR / f"esp32_comparison_{metric}_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_response_overlay(log_dict: dict, timestamp: str):
    plt.figure(figsize=(10, 5))

    for case_name, df in log_dict.items():
        plt.plot(df["time"], df["current"], label=case_name)

    # target은 아무 case에서나 가져와도 됨
    ref_df = next(iter(log_dict.values()))
    plt.plot(
        ref_df["time"],
        ref_df["target"],
        linestyle="--",
        label="target",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("RPM")
    plt.title("ESP32 Control Mode Comparison: RPM Response")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_comparison_rpm_overlay_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_pwm_overlay(log_dict: dict, timestamp: str):
    plt.figure(figsize=(10, 5))

    for case_name, df in log_dict.items():
        plt.plot(df["time"], df["pwm"], label=case_name)

    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("ESP32 Control Mode Comparison: PWM Command")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_comparison_pwm_overlay_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_error_overlay(log_dict: dict, timestamp: str):
    plt.figure(figsize=(10, 5))

    for case_name, df in log_dict.items():
        if "measured_error" in df.columns:
            error = df["measured_error"]
        elif "error" in df.columns:
            error = df["error"]
        else:
            error = df["target"] - df["current"]

        plt.plot(df["time"], error, label=case_name)

    plt.axhline(0.0, linewidth=1.0)

    plt.xlabel("Time [s]")
    plt.ylabel("Error [RPM]")
    plt.title("ESP32 Control Mode Comparison: Error Response")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_comparison_error_overlay_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# Summary
# ============================================================

def save_summary(metrics_df, metric_paths, log_paths, timestamp):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    selected_columns = [
        "case_name",
        "mode",
        "backend",
        "control_dt",
        "target_before",
        "target_after",
        "IAE",
        "after_change_IAE",
        "mean_abs_error",
        "final_error",
        "settling_time_after_change",
        "overshoot_percent",
        "mean_pwm",
        "max_pwm",
        "saturation_ratio_percent",
        "server_gain_applied_count",
        "duplicate_gain_discard_count",
        "unsafe_gain_discard_count",
        "local_gain_db_update_count",
        "local_gain_reduction_count",
        "local_gain_recovery_count",
    ]

    existing_columns = [col for col in selected_columns if col in metrics_df.columns]
    summary_df = metrics_df[existing_columns].copy()

    summary_path = SUMMARY_DIR / f"esp32_control_comparison_summary_{timestamp}.csv"
    markdown_path = SUMMARY_DIR / f"esp32_control_comparison_summary_{timestamp}.md"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# ESP32 Control Mode Comparison Summary")
    lines.append("")
    lines.append("## 1. Compared Cases")
    lines.append("")
    lines.append("- Case A: Fixed PID")
    lines.append("- Case B: Local gain DB-based gain-scheduled PID")
    lines.append("- Case C: Kafka server-assisted gain-scheduled PID")
    lines.append("")

    lines.append("## 2. Loaded Metric Files")
    lines.append("")
    for name, path in metric_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")

    lines.append("## 3. Loaded Log Files")
    lines.append("")
    for name, path in log_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")

    lines.append("## 4. Metric Summary")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(
        "The three ESP32 control modes were compared under the same target step condition "
        "of 30 to 50 RPM. The fixed PID baseline verifies stable closed-loop control using "
        "a single gain set. The local gain DB mode verifies target-dependent gain scheduling "
        "without network communication. The Kafka server-assisted mode verifies the complete "
        "communication loop, including motor-state publication, server-side gain recommendation, "
        "gain-command reception, duplicate-command filtering, unsafe-gain guarding, and local PWM "
        "rate limiting."
    )
    lines.append("")
    lines.append(
        "If the metrics of Case C remain close to Case A and Case B while successfully applying "
        "server-side gain commands, this supports the feasibility of the proposed server-assisted "
        "adaptive PID control structure on the real ESP32 motor setup."
    )
    lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved summary Markdown: {markdown_path}")

    return summary_df


# ============================================================
# Main
# ============================================================

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_df, metric_paths = load_metrics()
    log_dict, log_paths = load_logs()

    print("=" * 80)
    print("ESP32 Control Mode Comparison")
    print("=" * 80)

    print("\nMetrics:")
    print(metrics_df)

    summary_df = save_summary(
        metrics_df=metrics_df,
        metric_paths=metric_paths,
        log_paths=log_paths,
        timestamp=timestamp,
    )

    print("\nSelected summary:")
    print(summary_df)

    plot_response_overlay(log_dict, timestamp)
    plot_pwm_overlay(log_dict, timestamp)
    plot_error_overlay(log_dict, timestamp)

    plot_metric_bar(metrics_df, "IAE", "IAE", timestamp)
    plot_metric_bar(metrics_df, "after_change_IAE", "After-change IAE", timestamp)
    plot_metric_bar(metrics_df, "final_error", "Final error [RPM]", timestamp)
    plot_metric_bar(metrics_df, "max_pwm", "Max PWM", timestamp)


if __name__ == "__main__":
    main()