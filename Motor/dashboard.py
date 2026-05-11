import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config import LOG_DIR, RESULTS_DIR


st.set_page_config(
    page_title="Adaptive PID Dashboard",
    layout="wide",
)

st.title("Adaptive PID Dashboard")

SUMMARY_DIR = RESULTS_DIR / "summary"
KAFKA_CONTROL_DIR = RESULTS_DIR / "kafka_control"


def get_latest_file(directory: Path, pattern: str):
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not files:
        return None
    return files[-1]


def read_csv_if_exists(path: Path | None):
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Failed to read {path.name}: {exc}")
        return None


def metric_value(df: pd.DataFrame, column: str):
    if df is None or df.empty or column not in df.columns:
        return None
    value = df[column].iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def show_metric_cards(df: pd.DataFrame, specs):
    cols = st.columns(len(specs))
    for col, (label, column, suffix) in zip(cols, specs):
        value = metric_value(df, column)
        text = "-" if value is None else f"{value:.4f}{suffix}"
        col.metric(label, text)


def select_columns(df: pd.DataFrame, columns):
    if df is None:
        return None
    return df[[col for col in columns if col in df.columns]]


def pretty_dataframe(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns
    return df.style.format({col: "{:.4f}" for col in numeric_cols})


def metrics_to_log_path(metrics_file: str):
    log_name = metrics_file.replace(
        "local_kafka_controller_metrics_", "local_kafka_controller_log_", 1
    )
    return KAFKA_CONTROL_DIR / log_name


def representative_metrics_row(raw_df: pd.DataFrame, scenario: str, method: str):
    subset = raw_df[
        raw_df["scenario"].eq(scenario) & raw_df["method"].eq(method)
    ].copy()
    if subset.empty or "IAE" not in subset.columns:
        return None
    mean_iae = subset["IAE"].mean()
    subset["_distance_to_mean_IAE"] = (subset["IAE"] - mean_iae).abs()
    return subset.sort_values("_distance_to_mean_IAE").iloc[0]


def load_representative_log(raw_df: pd.DataFrame, scenario: str, method: str):
    row = representative_metrics_row(raw_df, scenario, method)
    if row is None or "metrics_file" not in row:
        return None, None
    log_path = metrics_to_log_path(str(row["metrics_file"]))
    if not log_path.exists():
        return row, None
    return row, read_csv_if_exists(log_path)


def comparison_series(left_df: pd.DataFrame, right_df: pd.DataFrame, columns):
    frames = []
    for label, df in [("Direct", left_df), ("MT-DL", right_df)]:
        if df is None or "time" not in df.columns:
            continue
        available = [col for col in columns if col in df.columns]
        if not available:
            continue
        sub = df[["time", *available]].copy()
        sub = sub.set_index("time")
        sub = sub.rename(columns={col: f"{label} {col}" for col in available})
        frames.append(sub)
    if not frames:
        return None
    return pd.concat(frames, axis=1).sort_index()


def comparison_series_named(series_specs, columns):
    frames = []
    for label, df in series_specs:
        if df is None or "time" not in df.columns:
            continue
        available = [col for col in columns if col in df.columns]
        if not available:
            continue
        sub = df[["time", *available]].copy()
        sub = sub.set_index("time")
        sub = sub.rename(columns={col: f"{label} {col}" for col in available})
        frames.append(sub)
    if not frames:
        return None
    return pd.concat(frames, axis=1).sort_index()


def representative_row_by_iae(df: pd.DataFrame):
    if df is None or df.empty or "IAE" not in df.columns:
        return None
    data = df.copy()
    mean_iae = pd.to_numeric(data["IAE"], errors="coerce").mean()
    data["_distance_to_mean_IAE"] = (
        pd.to_numeric(data["IAE"], errors="coerce") - mean_iae
    ).abs()
    return data.sort_values("_distance_to_mean_IAE").iloc[0]


def load_log_from_metrics_name(metrics_file):
    if metrics_file is None or pd.isna(metrics_file):
        return None
    path = metrics_to_log_path(str(metrics_file))
    if not path.exists():
        return None
    return read_csv_if_exists(path)


def clean_final_row(final_df: pd.DataFrame, scenario: str, method: str):
    if final_df is None:
        return None
    subset = final_df[
        final_df["scenario"].eq(scenario) & final_df["method"].eq(method)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def make_ablation_summary():
    rows = [
        {
            "stage": "Clean Direct MLP",
            "intent": "State-to-gain supervised baseline",
            "outcome": "Strong baseline, but not trajectory-objective trained",
            "decision": "Reference baseline",
        },
        {
            "stage": "IAE/PWM label weighting",
            "intent": "Prefer lower IAE and weaker PWM use during label selection",
            "outcome": "Did not beat clean Direct in real control",
            "decision": "Rejected",
        },
        {
            "stage": "DB-reference / residual DB",
            "intent": "Inject discrete gain DB prior into the policy",
            "outcome": "Discrete prior alone did not improve closed-loop IAE",
            "decision": "Rejected",
        },
        {
            "stage": "Rank-weighted labels",
            "intent": "Use multiple top candidates per context instead of best-1 labels",
            "outcome": "Offline gain MAE improved, real IAE did not",
            "decision": "Rejected",
        },
        {
            "stage": "Surrogate-objective V1",
            "intent": "Train Direct policy through frozen MT-DL horizon IAE objective",
            "outcome": "Won 4/4 baseline scenarios in 5-repeat validation",
            "decision": "Selected",
        },
        {
            "stage": "Surrogate-objective V2 PWM penalty",
            "intent": "Reduce peak PWM while keeping V1 IAE benefit",
            "outcome": "Average IAE and max PWM did not improve over V1",
            "decision": "Rejected",
        },
    ]
    return pd.DataFrame(rows)


def get_latest_log_file():
    log_files = sorted(LOG_DIR.glob("*.csv"))

    if not log_files:
        return None

    return log_files[-1]


refresh_interval = st.sidebar.slider(
    "Refresh interval [s]",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5,
)

auto_refresh = st.sidebar.checkbox("Auto refresh realtime tab", value=True)

realtime_tab, results_tab, logs_tab = st.tabs(
    ["Realtime Monitor", "Experiment Results", "Kafka Logs"]
)

with realtime_tab:
    log_file = get_latest_log_file()

    if log_file is None:
        st.warning("No realtime log file found.")
    else:
        st.caption(f"Current log file: {log_file}")

        try:
            df = pd.read_csv(log_file)
        except Exception as e:
            st.error(f"Failed to read log file: {e}")
            df = pd.DataFrame()

        if len(df) == 0:
            st.warning("Log file is empty.")
        else:
            latest = df.iloc[-1]

            mode = latest["mode"] if "mode" in df.columns else "unknown"
            env_type = latest["env_type"] if "env_type" in df.columns else "unknown"

            st.caption(f"Mode: {mode} | Environment: {env_type}")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Target", f"{latest['target']:.2f}")
            col2.metric("Current", f"{latest['current']:.2f}")
            col3.metric("Error", f"{latest['error']:.2f}")
            col4.metric("PWM", f"{latest['pwm']:.2f}")

            col5, col6, col7 = st.columns(3)

            col5.metric("Kp", f"{latest['kp']:.3f}")
            col6.metric("Ki", f"{latest['ki']:.3f}")
            col7.metric("Kd", f"{latest['kd']:.3f}")

            col8, col9, col10 = st.columns(3)

            if "pwm_saturated" in df.columns:
                col8.metric("PWM Saturated", str(latest["pwm_saturated"]))

            if "high_saturation" in df.columns:
                col9.metric("High Saturation", str(latest["high_saturation"]))

            if "gain_update_flag" in df.columns:
                col10.metric("Gain Updated", str(latest["gain_update_flag"]))

            col11, col12 = st.columns(2)

            if "integral" in df.columns:
                col11.metric("Integral", f"{latest['integral']:.3f}")

            if "prev_pwm" in df.columns:
                col12.metric("Previous PWM", f"{latest['prev_pwm']:.2f}")

            if "time" in df.columns:
                time_df = df.set_index("time")

                if "integral" in df.columns:
                    st.subheader("Integral Term")
                    st.line_chart(time_df[["integral"]])

                if "gain_update_flag" in df.columns:
                    st.subheader("Gain Update Flag")
                    st.line_chart(time_df[["gain_update_flag"]])

                sat_cols = [
                    col
                    for col in ["pwm_saturated", "high_saturation", "low_saturation"]
                    if col in df.columns
                ]
                if sat_cols:
                    st.subheader("PWM Saturation")
                    st.line_chart(time_df[sat_cols])

                st.subheader("Target vs Current")
                st.line_chart(time_df[["target", "current"]])

                st.subheader("PWM Output")
                st.line_chart(time_df[["pwm"]])

                st.subheader("PID Gains")
                st.line_chart(time_df[["kp", "ki", "kd"]])

            st.subheader("Latest Data")
            st.dataframe(df.tail(20), use_container_width=True)

with results_tab:
    st.subheader("Current Recommended Setting")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Policy", "Direct MLP")
    col2.metric("Inference Delay", "0.5 s")
    col3.metric("Chunk Horizon", "20 steps")
    col4.metric("Chunk Duration", "2.0 s")

    st.caption(
        "Dynamic scenario used for timing ablation: 70 -> 95 -> 90 -> 73 rpm."
    )

    st.markdown("**Research Result Snapshot**")
    st.caption(
        "Current best policy: Surrogate-Objective Direct MLP v1 "
        "(model id 20260502_153840). The policy is trained through a frozen "
        "MT-DL horizon model, but online inference uses only the Direct MLP."
    )

    final5_model_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_v1_final5_validation_model_summary_*.csv"
    )
    final5_model_df = read_csv_if_exists(final5_model_path)
    if final5_model_df is not None and not final5_model_df.empty:
        show_metric_cards(
            final5_model_df,
            [
                ("Final Runs", "runs", ""),
                ("Scenario Mean IAE", "IAE_mean", ""),
                ("After-change IAE", "after_change_IAE_mean", ""),
                ("Mean Max PWM", "max_pwm_mean", ""),
            ],
        )

    st.markdown("**Model Structure**")
    st.code(
        "\n".join(
            [
                "Online:",
                "  controller state (28 features)",
                "    -> Direct Policy MLP",
                "    -> normalized [kp, ki, kd]",
                "    -> gain bounds",
                "    -> 20-step gain chunk",
                "    -> Kafka / delay-aware controller",
                "",
                "Training:",
                "  state -> Direct Policy MLP -> predicted gain",
                "    -> frozen MT-DL horizon surrogate",
                "    -> predicted horizon metrics",
                "    -> loss: IAE + PWM/risk + gain-anchor + smoothness",
            ]
        ),
        language="text",
    )

    final_path = get_latest_file(
        SUMMARY_DIR, "final_clean_comparison_summary_*.csv"
    )
    final_df = read_csv_if_exists(final_path)

    final5_summary_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_v1_final5_validation_summary_*.csv"
    )
    final5_summary_df = read_csv_if_exists(final5_summary_path)
    final5_raw_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_v1_final5_validation_raw_*.csv"
    )
    final5_raw_df = read_csv_if_exists(final5_raw_path)
    final5_manifest_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_v1_final5_scenario_manifest_*.csv"
    )
    final5_manifest_df = read_csv_if_exists(final5_manifest_path)

    if final5_summary_df is not None:
        st.markdown("**Final 5-Repeat Validation Summary**")
        st.caption(f"Source: {final5_summary_path.name}")
        final5_cols = [
            "scenario",
            "category",
            "n",
            "IAE",
            "IAE_std",
            "after_change_IAE",
            "after_change_IAE_std",
            "mean_pwm",
            "max_pwm",
            "max_pwm_std",
            "latency_generator_duration_sec_p90",
            "delta_IAE_vs_clean",
            "delta_after_change_IAE_vs_clean",
            "delta_max_pwm_vs_clean",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(final5_summary_df, final5_cols)),
            use_container_width=True,
        )

        chart_cols = [
            col
            for col in ["IAE", "after_change_IAE", "max_pwm"]
            if col in final5_summary_df.columns
        ]
        if chart_cols:
            chart_df = final5_summary_df[["scenario", *chart_cols]].copy()
            st.bar_chart(chart_df.set_index("scenario")[chart_cols])

    reliability_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_v1_reliability_compact_*.csv"
    )
    reliability_df = read_csv_if_exists(reliability_path)
    if reliability_df is not None:
        st.markdown("**Direct Policy v1 Real-Motor Reliability**")
        st.caption(f"Source: {reliability_path.name}")
        reliability_cols = [
            "scenario_id",
            "target_profile",
            "runs",
            "IAE_mean",
            "IAE_std",
            "IAE_cv_percent",
            "after_change_IAE_mean",
            "after_change_IAE_std",
            "max_pwm_mean",
            "max_pwm_max",
            "saturation_ratio_percent_max",
            "unsafe_gain_discard_count_max",
            "latency_generator_duration_sec_p90_mean",
            "latency_source_to_accept_sec_p90_mean",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(reliability_df, reliability_cols)),
            use_container_width=True,
        )

        reliability_chart_cols = [
            col
            for col in ["IAE_mean", "after_change_IAE_mean", "max_pwm_mean"]
            if col in reliability_df.columns
        ]
        if reliability_chart_cols and "target_profile" in reliability_df.columns:
            st.bar_chart(
                reliability_df.set_index("target_profile")[reliability_chart_cols]
            )

    baseline_compare_path = get_latest_file(
        SUMMARY_DIR, "baseline_comparison_same_scenarios_summary_*.csv"
    )
    baseline_compare_df = read_csv_if_exists(baseline_compare_path)
    baseline_overall_path = get_latest_file(
        SUMMARY_DIR, "baseline_comparison_same_scenarios_overall_*.csv"
    )
    baseline_overall_df = read_csv_if_exists(baseline_overall_path)
    if baseline_compare_df is not None:
        st.markdown("**Same-Scenario Baseline Comparison**")
        st.caption(f"Source: {baseline_compare_path.name}")

        if baseline_overall_df is not None:
            st.caption(f"Overall source: {baseline_overall_path.name}")
            overall_cols = [
                "model_label",
                "scenarios",
                "IAE_mean",
                "after_change_IAE_mean",
                "max_pwm_mean",
                "max_pwm_observed",
                "saturation_max",
                "generator_p90_mean",
            ]
            st.dataframe(
                pretty_dataframe(select_columns(baseline_overall_df, overall_cols)),
                use_container_width=True,
            )

        compare_cols = [
            "target_profile",
            "model_label",
            "runs",
            "IAE_mean",
            "IAE_std",
            "after_change_IAE_mean",
            "max_pwm_mean",
            "max_pwm_max",
            "delta_IAE_vs_direct_v1",
            "delta_after_change_IAE_vs_direct_v1",
            "is_best_IAE",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(baseline_compare_df, compare_cols)),
            use_container_width=True,
        )

        if {"target_profile", "model_label", "IAE_mean"}.issubset(
            baseline_compare_df.columns
        ):
            chart_df = baseline_compare_df.pivot_table(
                index="target_profile",
                columns="model_label",
                values="IAE_mean",
                aggfunc="first",
            )
            st.bar_chart(chart_df)

    if final5_manifest_df is not None:
        st.markdown("**Final Scenario Definition**")
        st.caption(f"Source: {final5_manifest_path.name}")
        manifest_cols = [
            "scenario",
            "category",
            "target_sequence",
            "target_change_times",
            "sim_time",
            "repeats",
            "inference_delay_sec",
            "chunk_horizon_steps",
            "description",
        ]
        st.dataframe(
            select_columns(final5_manifest_df, manifest_cols),
            use_container_width=True,
        )

    st.markdown("**Ablation Summary**")
    st.dataframe(make_ablation_summary(), use_container_width=True)

    if final_df is not None:
        st.markdown("**Clean Method Comparison**")
        st.caption(f"Source: {final_path.name}")

        best_final = final_df.sort_values(["scenario_tag", "IAE_mean"]).groupby(
            "scenario", as_index=False
        ).head(1)
        best_cols = [
            "scenario",
            "method",
            "IAE_mean",
            "after_change_IAE_mean",
            "max_pwm_mean",
            "latency_generator_duration_sec_p90_mean",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(best_final, best_cols)),
            use_container_width=True,
        )

        final_cols = [
            "scenario",
            "method",
            "n",
            "IAE_mean",
            "IAE_std",
            "after_change_IAE_mean",
            "after_change_IAE_std",
            "mean_pwm_mean",
            "max_pwm_mean",
            "schedule_chunk_accepted_count_mean",
            "schedule_fallback_count_mean",
            "latency_generator_duration_sec_p90_mean",
            "IAE_rank",
        ]
        final_view = select_columns(final_df, final_cols)
        st.dataframe(pretty_dataframe(final_view), use_container_width=True)

        chart_df = select_columns(
            final_df,
            ["scenario", "method", "IAE_mean", "after_change_IAE_mean"],
        )
        if chart_df is not None:
            chart_df = chart_df.copy()
            chart_df["case"] = chart_df["scenario"] + " | " + chart_df["method"]
            st.bar_chart(chart_df.set_index("case")[["IAE_mean"]])
    else:
        st.info("No clean method comparison summary found.")

    raw_final_path = get_latest_file(
        SUMMARY_DIR, "final_clean_comparison_raw_metrics_*.csv"
    )
    raw_final_df = read_csv_if_exists(raw_final_path)
    interpretation_path = get_latest_file(
        SUMMARY_DIR, "final_clean_direct_vs_mtdl_interpretation_*.csv"
    )
    interpretation_df = read_csv_if_exists(interpretation_path)

    if raw_final_df is not None and final5_raw_df is not None:
        st.markdown("**Representative Trajectory: Clean Direct vs V1**")
        comparable = [
            scenario
            for scenario in final5_raw_df["scenario"].dropna().unique()
            if scenario in set(raw_final_df["scenario"].dropna().unique())
        ]
        if comparable:
            selected_v1_scenario = st.selectbox(
                "Scenario for Clean vs V1 trajectory",
                comparable,
                index=0,
            )
            clean_row, clean_log = load_representative_log(
                raw_final_df,
                selected_v1_scenario,
                "Direct policy MLP",
            )
            v1_subset = final5_raw_df[
                final5_raw_df["scenario"].eq(selected_v1_scenario)
            ]
            v1_row = representative_row_by_iae(v1_subset)
            v1_log = (
                load_log_from_metrics_name(v1_row["metrics_file"])
                if v1_row is not None and "metrics_file" in v1_row
                else None
            )

            if clean_log is None or v1_log is None:
                st.warning("Representative Clean/V1 log pair not found.")
            else:
                compare_rows = []
                if clean_row is not None:
                    row = dict(clean_row)
                    row["model"] = "Clean Direct"
                    compare_rows.append(row)
                if v1_row is not None:
                    row = dict(v1_row)
                    row["model"] = "V1 Surrogate-Objective"
                    compare_rows.append(row)
                metric_cols = [
                    "model",
                    "IAE",
                    "after_change_IAE",
                    "mean_pwm",
                    "max_pwm",
                    "latency_generator_duration_sec_p90",
                    "metrics_file",
                ]
                st.dataframe(
                    pretty_dataframe(
                        select_columns(pd.DataFrame(compare_rows), metric_cols)
                    ),
                    use_container_width=True,
                )

                rpm_chart = comparison_series_named(
                    [("Clean", clean_log), ("V1", v1_log)],
                    ["target", "current"],
                )
                if rpm_chart is not None:
                    st.caption("Target and RPM")
                    st.line_chart(rpm_chart)

                error_chart = comparison_series_named(
                    [("Clean", clean_log), ("V1", v1_log)],
                    ["error"],
                )
                if error_chart is not None:
                    st.caption("Tracking Error")
                    st.line_chart(error_chart)

                pwm_chart = comparison_series_named(
                    [("Clean", clean_log), ("V1", v1_log)],
                    ["pwm"],
                )
                if pwm_chart is not None:
                    st.caption("PWM")
                    st.line_chart(pwm_chart)

                gain_chart = comparison_series_named(
                    [("Clean", clean_log), ("V1", v1_log)],
                    ["kp", "ki", "kd"],
                )
                if gain_chart is not None:
                    st.caption("Applied Gains")
                    st.line_chart(gain_chart)
        else:
            st.info("No overlapping Clean/V1 scenarios found for trajectory plots.")

    if interpretation_df is not None:
        st.markdown("**Direct Policy vs MT-DL Interpretation**")
        st.caption(f"Source: {interpretation_path.name}")
        interpretation_cols = [
            "scenario",
            "winner_by_IAE",
            "direct_minus_mtdl_IAE",
            "direct_minus_mtdl_after_change_IAE",
            "direct_minus_mtdl_max_pwm",
            "direct_latency_advantage_sec",
            "improvement_focus",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(interpretation_df, interpretation_cols)),
            use_container_width=True,
        )

    tiebreak_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_tiebreak_eval_comparison_*.csv"
    )
    tiebreak_df = read_csv_if_exists(tiebreak_path)

    if tiebreak_df is not None:
        st.markdown("**Direct Policy Improvement Trial**")
        st.caption(f"Source: {tiebreak_path.name}")
        tiebreak_cols = [
            "label",
            "n",
            "IAE",
            "after_change_IAE",
            "mean_pwm",
            "max_pwm",
            "latency_generator_duration_sec_p90",
            "target_81_IAE",
            "target_100_IAE",
            "target_95_IAE",
            "target_90_IAE",
            "target_73_IAE",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(tiebreak_df, tiebreak_cols)),
            use_container_width=True,
        )

    dbfeature_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_dbfeature_eval_comparison_*.csv"
    )
    dbfeature_df = read_csv_if_exists(dbfeature_path)

    if dbfeature_df is not None:
        st.markdown("**Direct Policy DB-Reference Feature Trial**")
        st.caption(f"Source: {dbfeature_path.name}")
        dbfeature_cols = [
            "label",
            "n",
            "IAE",
            "after_change_IAE",
            "mean_pwm",
            "max_pwm",
            "latency_generator_duration_sec_p90",
            "target_81_IAE",
            "target_100_IAE",
            "target_95_IAE",
            "target_90_IAE",
            "target_73_IAE",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(dbfeature_df, dbfeature_cols)),
            use_container_width=True,
        )

    transition_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_transition_eval_comparison_*.csv"
    )
    transition_df = read_csv_if_exists(transition_path)

    if transition_df is not None:
        st.markdown("**Direct Policy Transition-Context Trial**")
        st.caption(f"Source: {transition_path.name}")
        transition_cols = [
            "label",
            "n",
            "IAE",
            "after_change_IAE",
            "mean_pwm",
            "max_pwm",
            "latency_generator_duration_sec_p90",
            "target_81_IAE",
            "target_100_IAE",
            "target_95_IAE",
            "target_90_IAE",
            "target_73_IAE",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(transition_df, transition_cols)),
            use_container_width=True,
        )

    pwmaware_transition_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_pwmaware_transition_eval_comparison_*.csv"
    )
    pwmaware_transition_df = read_csv_if_exists(pwmaware_transition_path)

    if pwmaware_transition_df is not None:
        st.markdown("**Direct Policy PWM-Aware Transition Trial**")
        st.caption(f"Source: {pwmaware_transition_path.name}")
        st.caption(
            "This trial reduced peak PWM in the dynamic case, but increased IAE, "
            "so the stable latest model remains the clean Direct policy."
        )
        pwmaware_transition_cols = [
            "label",
            "n",
            "IAE",
            "after_change_IAE",
            "mean_pwm",
            "max_pwm",
            "near_high_saturation_ratio_percent",
            "latency_generator_duration_sec_p90",
            "target_81_IAE",
            "target_100_IAE",
            "target_95_IAE",
            "target_90_IAE",
            "target_73_IAE",
        ]
        st.dataframe(
            pretty_dataframe(
                select_columns(pwmaware_transition_df, pwmaware_transition_cols)
            ),
            use_container_width=True,
        )

    iae_first_screening_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_iae_first_screening_*.csv"
    )
    iae_first_screening_df = read_csv_if_exists(iae_first_screening_path)

    if iae_first_screening_df is not None:
        st.markdown("**Direct Policy IAE-First Candidate Screening**")
        st.caption(f"Source: {iae_first_screening_path.name}")
        st.caption(
            "One-run screening for weak PWM tie-break candidates. The best "
            "candidate did not beat the clean Direct policy baseline."
        )
        iae_first_cols = [
            "candidate",
            "scenario",
            "n",
            "IAE",
            "after_change_IAE",
            "mean_pwm",
            "max_pwm",
            "latency_generator_duration_sec_p90",
            "target_81_IAE",
            "target_100_IAE",
            "target_95_IAE",
            "target_90_IAE",
            "target_73_IAE",
            "combined_IAE",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(iae_first_screening_df, iae_first_cols)),
            use_container_width=True,
        )

    clean_wide_validation_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_clean_wide_validation_comparison_*.csv"
    )
    clean_wide_validation_df = read_csv_if_exists(clean_wide_validation_path)

    if clean_wide_validation_df is not None:
        st.markdown("**Direct Policy Clean-Structure Wide MLP Validation**")
        st.caption(f"Source: {clean_wide_validation_path.name}")
        st.caption(
            "Clean input features with a wider MLP looked promising in one run, "
            "but the three-run validation did not beat the clean Direct baseline."
        )
        clean_wide_cols = [
            "label",
            "n",
            "IAE",
            "IAE_std",
            "after_change_IAE",
            "after_change_IAE_std",
            "mean_pwm",
            "max_pwm",
            "max_pwm_std",
            "latency_generator_duration_sec_p90",
            "target_81_IAE",
            "target_100_IAE",
            "target_95_IAE",
            "target_90_IAE",
            "target_73_IAE",
        ]
        st.dataframe(
            pretty_dataframe(
                select_columns(clean_wide_validation_df, clean_wide_cols)
            ),
            use_container_width=True,
        )

    residual_db_screening_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_residual_db_screening_*.csv"
    )
    residual_db_screening_df = read_csv_if_exists(residual_db_screening_path)

    if residual_db_screening_df is not None:
        st.markdown("**Direct Policy Residual-DB Screening**")
        st.caption(f"Source: {residual_db_screening_path.name}")
        st.caption(
            "Discrete DB interpolation was used as the policy prior and the MLP "
            "predicted residual gain corrections. The first real screening did "
            "not beat the clean Direct baseline."
        )
        residual_db_cols = [
            "candidate",
            "scenario",
            "n",
            "IAE",
            "after_change_IAE",
            "mean_pwm",
            "max_pwm",
            "latency_generator_duration_sec_p90",
            "target_81_IAE",
            "target_100_IAE",
            "target_95_IAE",
            "target_90_IAE",
            "target_73_IAE",
            "combined_IAE",
        ]
        st.dataframe(
            pretty_dataframe(
                select_columns(residual_db_screening_df, residual_db_cols)
            ),
            use_container_width=True,
        )

    rank_weighted_screening_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_rank_weighted_screening_*.csv"
    )
    rank_weighted_screening_df = read_csv_if_exists(rank_weighted_screening_path)

    if rank_weighted_screening_df is not None:
        st.markdown("**Direct Policy Rank-Weighted Screening**")
        st.caption(f"Source: {rank_weighted_screening_path.name}")
        st.caption(
            "Multiple top-ranked gains per context were kept with soft weights. "
            "Offline gain MAE improved, but real control IAE did not beat baseline."
        )
        rank_weighted_cols = [
            "candidate",
            "scenario",
            "n",
            "IAE",
            "after_change_IAE",
            "mean_pwm",
            "max_pwm",
            "latency_generator_duration_sec_p90",
            "target_81_IAE",
            "target_100_IAE",
            "target_95_IAE",
            "target_90_IAE",
            "target_73_IAE",
            "combined_IAE",
        ]
        st.dataframe(
            pretty_dataframe(
                select_columns(rank_weighted_screening_df, rank_weighted_cols)
            ),
            use_container_width=True,
        )

    surrogate_objective_validation_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_surrogate_objective_validation_comparison_*.csv"
    )
    surrogate_objective_validation_df = read_csv_if_exists(
        surrogate_objective_validation_path
    )

    if surrogate_objective_validation_df is not None:
        st.markdown("**Direct Policy Surrogate-Objective Validation**")
        st.caption(f"Source: {surrogate_objective_validation_path.name}")
        st.caption(
            "The direct policy was trained through a frozen MT-DL horizon model, "
            "optimizing predicted closed-loop IAE instead of matching gain labels."
        )
        surrogate_objective_cols = [
            "label",
            "n",
            "IAE",
            "IAE_std",
            "after_change_IAE",
            "after_change_IAE_std",
            "settling_time_after_change",
            "mean_pwm",
            "max_pwm",
            "max_pwm_std",
            "latency_generator_duration_sec_p90",
            "target_81_IAE",
            "target_100_IAE",
            "target_95_IAE",
            "target_90_IAE",
            "target_73_IAE",
        ]
        st.dataframe(
            pretty_dataframe(
                select_columns(
                    surrogate_objective_validation_df,
                    surrogate_objective_cols,
                )
            ),
            use_container_width=True,
        )

    surrogate_objective_extra_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_surrogate_objective_extra_robustness_*.csv"
    )
    surrogate_objective_extra_df = read_csv_if_exists(
        surrogate_objective_extra_path
    )

    if surrogate_objective_extra_df is not None:
        st.markdown("**Surrogate-Objective Extra Robustness Check**")
        st.caption(f"Source: {surrogate_objective_extra_path.name}")
        st.caption(
            "Additional seen and reverse-transition scenarios used to confirm "
            "that the surrogate-objective policy improvement generalizes beyond "
            "the first validation set."
        )
        surrogate_extra_cols = [
            "label",
            "n",
            "IAE",
            "IAE_std",
            "after_change_IAE",
            "after_change_IAE_std",
            "settling_time_after_change",
            "mean_pwm",
            "max_pwm",
            "max_pwm_std",
            "latency_generator_duration_sec_p90",
            "target_70_IAE",
            "target_85_IAE",
            "target_100_IAE",
        ]
        st.dataframe(
            pretty_dataframe(
                select_columns(surrogate_objective_extra_df, surrogate_extra_cols)
            ),
            use_container_width=True,
        )

    surrogate_v2_validation_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_surrogate_objective_v2_validation_comparison_*.csv"
    )
    surrogate_v2_validation_df = read_csv_if_exists(surrogate_v2_validation_path)

    if surrogate_v2_validation_df is not None:
        st.markdown("**Surrogate-Objective V2 PWM Penalty Validation**")
        st.caption(f"Source: {surrogate_v2_validation_path.name}")
        st.caption(
            "PWM/risk penalties were increased to reduce peak PWM. The tested "
            "mild V2 did not beat V1 on the four-scenario average."
        )
        surrogate_v2_cols = [
            "label",
            "scenario",
            "model",
            "n",
            "IAE",
            "after_change_IAE",
            "mean_pwm",
            "max_pwm",
            "delta_vs_v1_IAE",
            "delta_vs_v1_after_change_IAE",
            "delta_vs_v1_max_pwm",
        ]
        st.dataframe(
            pretty_dataframe(
                select_columns(surrogate_v2_validation_df, surrogate_v2_cols)
            ),
            use_container_width=True,
        )

    surrogate_v2_model_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_surrogate_objective_v2_model_summary_*.csv"
    )
    surrogate_v2_model_df = read_csv_if_exists(surrogate_v2_model_path)

    if surrogate_v2_model_df is not None:
        st.caption(f"V2 model summary source: {surrogate_v2_model_path.name}")
        st.dataframe(
            pretty_dataframe(surrogate_v2_model_df),
            use_container_width=True,
        )

    v1_robust_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_v1_robust_validation_summary_*.csv"
    )
    v1_robust_df = read_csv_if_exists(v1_robust_path)

    if v1_robust_df is not None:
        st.markdown("**V1 Surrogate-Objective Robust Validation**")
        st.caption(f"Source: {v1_robust_path.name}")
        st.caption(
            "Six-scenario validation for the selected V1 policy, including "
            "seen, unseen, dynamic, reverse, large-step, and mixed transitions."
        )
        v1_robust_cols = [
            "scenario",
            "n",
            "IAE",
            "IAE_std",
            "after_change_IAE",
            "after_change_IAE_std",
            "mean_pwm",
            "max_pwm",
            "max_pwm_std",
            "latency_generator_duration_sec_p90",
            "delta_IAE_vs_clean",
            "delta_after_change_IAE_vs_clean",
            "delta_max_pwm_vs_clean",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(v1_robust_df, v1_robust_cols)),
            use_container_width=True,
        )

    v1_robust_model_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_v1_robust_validation_model_summary_*.csv"
    )
    v1_robust_model_df = read_csv_if_exists(v1_robust_model_path)

    if v1_robust_model_df is not None:
        st.caption(f"V1 robust model summary source: {v1_robust_model_path.name}")
        st.dataframe(
            pretty_dataframe(v1_robust_model_df),
            use_container_width=True,
        )

    final5_manifest_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_v1_final5_scenario_manifest_*.csv"
    )
    final5_manifest_df = read_csv_if_exists(final5_manifest_path)

    if final5_manifest_df is not None:
        st.markdown("**V1 Final 5-Repeat Scenario Set**")
        st.caption(f"Source: {final5_manifest_path.name}")
        manifest_cols = [
            "scenario",
            "category",
            "target_sequence",
            "target_change_times",
            "sim_time",
            "repeats",
            "inference_delay_sec",
            "chunk_horizon_steps",
            "description",
        ]
        st.dataframe(
            select_columns(final5_manifest_df, manifest_cols),
            use_container_width=True,
        )

    final5_summary_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_v1_final5_validation_summary_*.csv"
    )
    final5_summary_df = read_csv_if_exists(final5_summary_path)

    if final5_summary_df is not None:
        st.markdown("**V1 Final 5-Repeat Validation Results**")
        st.caption(f"Source: {final5_summary_path.name}")
        final5_cols = [
            "scenario",
            "category",
            "n",
            "IAE",
            "IAE_std",
            "after_change_IAE",
            "after_change_IAE_std",
            "mean_pwm",
            "max_pwm",
            "max_pwm_std",
            "latency_generator_duration_sec_p90",
            "delta_IAE_vs_clean",
            "delta_after_change_IAE_vs_clean",
            "delta_max_pwm_vs_clean",
        ]
        st.dataframe(
            pretty_dataframe(select_columns(final5_summary_df, final5_cols)),
            use_container_width=True,
        )

    final5_model_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_v1_final5_validation_model_summary_*.csv"
    )
    final5_model_df = read_csv_if_exists(final5_model_path)

    if final5_model_df is not None:
        st.caption(f"V1 final 5-repeat model summary source: {final5_model_path.name}")
        st.dataframe(
            pretty_dataframe(final5_model_df),
            use_container_width=True,
        )

    if raw_final_df is not None:
        st.markdown("**Representative Trajectory: Direct Policy vs MT-DL**")
        scenarios = list(raw_final_df["scenario"].dropna().unique())
        selected_scenario = st.selectbox(
            "Scenario for trajectory comparison",
            scenarios,
            index=0,
        )

        direct_row, direct_log = load_representative_log(
            raw_final_df, selected_scenario, "Direct policy MLP"
        )
        mtdl_row, mtdl_log = load_representative_log(
            raw_final_df, selected_scenario, "MT-DL surrogate"
        )

        if direct_log is None or mtdl_log is None:
            st.warning("Representative log pair not found for this scenario.")
        else:
            metric_cols = [
                "method",
                "rep",
                "IAE",
                "after_change_IAE",
                "mean_pwm",
                "max_pwm",
                "latency_generator_duration_sec_p90",
                "metrics_file",
            ]
            pair_df = pd.DataFrame([direct_row, mtdl_row])
            st.dataframe(
                pretty_dataframe(select_columns(pair_df, metric_cols)),
                use_container_width=True,
            )

            rpm_chart = comparison_series(direct_log, mtdl_log, ["target", "current"])
            if rpm_chart is not None:
                st.caption("Target and RPM")
                st.line_chart(rpm_chart)

            error_chart = comparison_series(direct_log, mtdl_log, ["error"])
            if error_chart is not None:
                st.caption("Tracking Error")
                st.line_chart(error_chart)

            pwm_chart = comparison_series(direct_log, mtdl_log, ["pwm"])
            if pwm_chart is not None:
                st.caption("PWM")
                st.line_chart(pwm_chart)

            gain_chart = comparison_series(direct_log, mtdl_log, ["kp", "ki", "kd"])
            if gain_chart is not None:
                st.caption("Applied Gains")
                st.line_chart(gain_chart)

    horizon_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_chunk_horizon_sweep_dynamic_*.csv"
    )
    horizon_df = read_csv_if_exists(horizon_path)

    if horizon_df is not None:
        best_horizon = horizon_df.sort_values("IAE_mean").head(1)
        st.markdown("**Chunk Horizon Sweep**")
        st.caption(f"Source: {horizon_path.name}")
        show_metric_cards(
            best_horizon,
            [
                ("Best Horizon", "horizon_steps", " steps"),
                ("IAE", "IAE_mean", ""),
                ("After-change IAE", "after_change_IAE_mean", ""),
                ("Max PWM", "max_pwm_mean", ""),
            ],
        )

        horizon_cols = [
            "horizon_steps",
            "horizon_sec",
            "IAE_mean",
            "after_change_IAE_mean",
            "mean_pwm_mean",
            "max_pwm_mean",
            "schedule_chunk_accepted_count_mean",
            "schedule_fallback_count_mean",
            "latency_source_to_apply_sec_p90_mean",
            "latency_generator_duration_sec_p90_mean",
        ]
        horizon_view = select_columns(horizon_df, horizon_cols)
        st.dataframe(pretty_dataframe(horizon_view), use_container_width=True)

        chart_cols = [
            col
            for col in ["IAE_mean", "after_change_IAE_mean", "max_pwm_mean"]
            if col in horizon_df.columns
        ]
        if chart_cols:
            st.line_chart(horizon_df.set_index("horizon_steps")[chart_cols])
    else:
        st.info("No chunk horizon sweep summary found.")

    delay_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_inference_delay_sweep_dynamic_*.csv"
    )
    delay_df = read_csv_if_exists(delay_path)

    if delay_df is not None:
        best_delay = delay_df.sort_values("IAE_mean").head(1)
        st.markdown("**Inference Delay Sweep**")
        st.caption(f"Source: {delay_path.name}")
        show_metric_cards(
            best_delay,
            [
                ("Best Delay", "delay_sec", " s"),
                ("IAE", "IAE_mean", ""),
                ("After-change IAE", "after_change_IAE_mean", ""),
                ("Max PWM", "max_pwm_mean", ""),
            ],
        )

        delay_cols = [
            "delay_sec",
            "IAE_mean",
            "after_change_IAE_mean",
            "mean_pwm_mean",
            "max_pwm_mean",
            "schedule_chunk_accepted_count_mean",
            "latency_source_to_accept_sec_p50_mean",
            "latency_source_to_apply_sec_p90_mean",
            "latency_generator_duration_sec_p90_mean",
        ]
        delay_view = select_columns(delay_df, delay_cols)
        st.dataframe(pretty_dataframe(delay_view), use_container_width=True)

        chart_cols = [
            col for col in ["IAE_mean", "after_change_IAE_mean", "max_pwm_mean"]
            if col in delay_df.columns
        ]
        if chart_cols:
            st.line_chart(delay_df.set_index("delay_sec")[chart_cols])
    else:
        st.info("No inference delay sweep summary found.")

    label_path = get_latest_file(
        SUMMARY_DIR, "direct_policy_pareto_iae_label_comparison_*.csv"
    )
    label_df = read_csv_if_exists(label_path)

    if label_df is not None:
        st.markdown("**Direct Policy Label Comparison**")
        st.caption(f"Source: {label_path.name}")
        label_cols = [
            "label",
            "n",
            "IAE",
            "after_change_IAE",
            "mean_pwm",
            "max_pwm",
            "latency_generator_duration_sec_p90",
            "target_95_IAE",
            "target_90_IAE",
            "target_73_IAE",
        ]
        label_view = select_columns(label_df, label_cols)
        st.dataframe(pretty_dataframe(label_view), use_container_width=True)
    else:
        st.info("No direct policy label comparison summary found.")

with logs_tab:
    st.subheader("Recent Kafka Control Metrics")

    metrics_files = sorted(
        KAFKA_CONTROL_DIR.glob("local_kafka_controller_metrics_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not metrics_files:
        st.info("No Kafka metrics files found.")
    else:
        selected = st.selectbox(
            "Metrics file",
            metrics_files,
            format_func=lambda path: path.name,
        )
        metrics_df = read_csv_if_exists(selected)
        if metrics_df is not None:
            st.caption(str(selected))
            cols = [
                "IAE",
                "after_change_IAE",
                "settling_time_after_change",
                "mean_pwm",
                "max_pwm",
                "schedule_chunk_accepted_count",
                "schedule_fallback_count",
                "latency_source_to_accept_sec_p50",
                "latency_source_to_apply_sec_p90",
                "latency_generator_duration_sec_p90",
            ]
            metric_view = select_columns(metrics_df, cols)
            st.dataframe(pretty_dataframe(metric_view), use_container_width=True)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
