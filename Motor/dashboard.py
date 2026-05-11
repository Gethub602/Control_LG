from pathlib import Path

import pandas as pd
import streamlit as st

from config import LOG_DIR


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts" / "final_ddim20"
SUMMARY_DIR = ARTIFACT_DIR / "summary"
MODEL_DIR = ARTIFACT_DIR / "models"


st.set_page_config(
    page_title="Diffusion Gain-Chunk Control",
    page_icon="DC",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #d9dee8;
            --panel: #f7f8fb;
            --accent: #ea6a2a;
            --accent-soft: #fff1e8;
            --blue: #18277a;
            --blue-soft: #eef1ff;
            --green: #16865a;
        }
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }
        h1 {
            font-size: 2.05rem !important;
            margin-bottom: 0.25rem !important;
        }
        h2 {
            font-size: 1.35rem !important;
            margin-top: 1.2rem !important;
        }
        h3 {
            font-size: 1.05rem !important;
        }
        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, #ffffff 0%, #f7f8fb 100%);
            padding: 1.1rem 1.25rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 760;
            color: var(--ink);
            margin: 0;
        }
        .hero-subtitle {
            color: var(--muted);
            margin-top: 0.25rem;
            font-size: 0.98rem;
        }
        .pill-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin-top: 0.85rem;
        }
        .pill {
            border: 1px solid #f0c6ac;
            background: var(--accent-soft);
            color: #9b3d0f;
            border-radius: 999px;
            padding: 0.22rem 0.6rem;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .note {
            border-left: 4px solid var(--accent);
            padding: 0.65rem 0.8rem;
            background: #fff8f4;
            color: #4c2a17;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        .section-caption {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: -0.2rem;
            margin-bottom: 0.75rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            background: #ffffff;
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
        }
        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-size: 0.78rem !important;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink) !important;
            font-size: 1.35rem !important;
        }
        .stDataFrame {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value, digits=2, suffix=""):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def find_file(pattern: str) -> Path | None:
    files = sorted(SUMMARY_DIR.glob(pattern))
    return files[-1] if files else None


def final_model_path() -> Path:
    return MODEL_DIR / (
        "diffusion_gain_chunk_unet_balanced1000_global_topk_full_"
        "20260508_193250.joblib"
    )


def style_numeric(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns
    return df.style.format({col: "{:.3f}" for col in numeric_cols})


def metric_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, suffix = item
        col.metric(label, fmt(value, suffix=suffix))


def compact_method_name(name: str) -> str:
    return (
        str(name)
        .replace("Full_Diffusion_", "")
        .replace("_5run", "")
        .replace("_", " ")
    )


def method_order_key(name: str) -> int:
    text = str(name)
    for idx, token in enumerate(["DDIM20", "DDIM30", "DDIM40", "DDIM50"]):
        if token in text:
            return idx
    return 99


def get_latest_file(directory: Path, pattern: str):
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def get_latest_log_file():
    if not LOG_DIR.exists():
        return None
    return get_latest_file(LOG_DIR, "*.csv")


inject_css()

comparison_path = find_file("full_diffusion_ddim20_30_40_50_comparison_*.csv")
repeats_path = find_file("full_diffusion_ddim20_5run_repeats_*.csv")
segment_path = find_file("full_diffusion_ddim20_30_50_segment_compare_*.csv")
gain_path = find_file("full_diffusion_ddim20_30_50_decel_gain_compare_*.csv")
light_path = find_file("light32_ddim20_after_isaac_off_5run_summary_*.csv")
bench_path = find_file("diffusion_unet_lightweight_ddim20_inference_benchmark_*.csv")

comparison_df = read_csv(comparison_path) if comparison_path else pd.DataFrame()
repeats_df = read_csv(repeats_path) if repeats_path else pd.DataFrame()
segment_df = read_csv(segment_path) if segment_path else pd.DataFrame()
gain_df = read_csv(gain_path) if gain_path else pd.DataFrame()
light_df = read_csv(light_path) if light_path else pd.DataFrame()
bench_df = read_csv(bench_path) if bench_path else pd.DataFrame()

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">Diffusion-Based PID Gain Chunk Scheduler</div>
      <div class="hero-subtitle">
        Real-motor validation of asynchronous server-assisted gain scheduling
        with a DDIM20 diffusion U-Net.
      </div>
      <div class="pill-row">
        <span class="pill">ESP32 local PID</span>
        <span class="pill">Kafka async schedule</span>
        <span class="pill">20-step gain chunk</span>
        <span class="pill">2.0 s horizon</span>
        <span class="pill">DDIM20 selected</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Final Study")
st.sidebar.caption("Artifact-backed dashboard")
st.sidebar.write(f"`{ARTIFACT_DIR.relative_to(BASE_DIR)}`")
st.sidebar.divider()
st.sidebar.markdown("**Scenario**")
st.sidebar.write("70 -> 95 -> 90 -> 73 RPM")
st.sidebar.write("Changes: 5, 10, 15 s")
st.sidebar.divider()
st.sidebar.markdown("**Selected Model**")
st.sidebar.write("Full Diffusion U-Net")
st.sidebar.write("DDIM steps: 20")
st.sidebar.write("Chunk: 20 x 0.1 s")

overview_tab, performance_tab, chunks_tab, realtime_tab, artifacts_tab = st.tabs(
    [
        "Overview",
        "Performance",
        "Gain Chunk Behavior",
        "Realtime Monitor",
        "Artifacts",
    ]
)

with overview_tab:
    st.subheader("Selected Result")
    st.markdown(
        '<div class="section-caption">Final method selected from DDIM step and lightweight ablations.</div>',
        unsafe_allow_html=True,
    )

    if comparison_df.empty:
        st.warning("Final comparison CSV was not found.")
    else:
        final_row = comparison_df[
            comparison_df["method"].eq("Full_Diffusion_DDIM20_5run")
        ].iloc[0]
        metric_row(
            [
                ("IAE", final_row.get("IAE_mean"), ""),
                ("After-change IAE", final_row.get("after_change_IAE_mean"), ""),
                ("Generator p90", final_row.get("latency_generator_duration_sec_p90_mean"), " s"),
                ("Source-to-apply p90", final_row.get("latency_source_to_apply_sec_p90_mean"), " s"),
            ]
        )

        st.markdown("")
        st.markdown(
            """
            <div class="note">
            The PC/Kafka server does not compute PWM at every control step.
            It asynchronously publishes future PID gain chunks, while the ESP32
            keeps the low-level PID loop local and continues with the latest
            valid gain schedule when communication is delayed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Control Architecture")
    arch_cols = st.columns([1.0, 1.0, 1.0, 1.0])
    arch_cols[0].markdown("**1. Local State**")
    arch_cols[0].write("Encoder RPM, target, error, PWM, PID terms, and recent history are published as `motor_state`.")
    arch_cols[1].markdown("**2. Diffusion Policy**")
    arch_cols[1].write("A conditional U-Net denoises a 20-step `[Kp, Ki, Kd]` chunk using DDIM20.")
    arch_cols[2].markdown("**3. Schedule Buffer**")
    arch_cols[2].write("Kafka-delivered chunks are inserted into a delay-aware FIFO schedule.")
    arch_cols[3].markdown("**4. ESP32 PID**")
    arch_cols[3].write("The local controller applies time-varying gains while preserving real-time PWM control.")

    st.subheader("Runtime Configuration")
    config_df = pd.DataFrame(
        [
            ["Final generator", "Conditional diffusion U-Net"],
            ["Sampler", "DDIM20"],
            ["Chunk horizon", "20 steps / 2.0 s"],
            ["Step interval", "0.1 s"],
            ["Inference delay assumption", "0.5 s"],
            ["Validation target sequence", "70 -> 95 -> 90 -> 73 RPM"],
            ["Kafka topics", "motor_state, motor_schedule_chunk, motor_gain_command"],
        ],
        columns=["Item", "Value"],
    )
    st.dataframe(config_df, hide_index=True, use_container_width=True)

with performance_tab:
    st.subheader("DDIM Step Ablation")
    st.markdown(
        '<div class="section-caption">Full diffusion model tested on the same real-motor transition scenario.</div>',
        unsafe_allow_html=True,
    )

    if comparison_df.empty:
        st.warning("No comparison data available.")
    else:
        display = comparison_df.copy()
        display["label"] = display["method"].map(compact_method_name)
        display = display.sort_values("method", key=lambda col: col.map(method_order_key))

        metric_cols = [
            "label",
            "runs",
            "IAE_mean",
            "IAE_std",
            "after_change_IAE_mean",
            "after_change_IAE_std",
            "mean_pwm_mean",
            "max_pwm_mean",
            "schedule_chunk_accepted_count_mean",
            "schedule_gain_applied_count_mean",
            "schedule_fallback_count_mean",
            "latency_generator_duration_sec_p90_mean",
            "latency_source_to_apply_sec_p90_mean",
        ]
        st.dataframe(
            style_numeric(display[[c for c in metric_cols if c in display.columns]]),
            use_container_width=True,
            hide_index=True,
        )

        chart_df = display.set_index("label")
        c1, c2 = st.columns(2)
        c1.markdown("**Tracking Error**")
        c1.bar_chart(chart_df[["IAE_mean", "after_change_IAE_mean"]])
        c2.markdown("**Timing**")
        c2.bar_chart(
            chart_df[
                [
                    "latency_generator_duration_sec_p90_mean",
                    "latency_source_to_apply_sec_p90_mean",
                ]
            ]
        )

    st.subheader("Lightweight Model Check")
    st.markdown(
        '<div class="section-caption">Model-size reduction was evaluated after the final DDIM20 result.</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    if not light_df.empty:
        light_row = light_df.iloc[0]
        light_summary = pd.DataFrame(
            [
                {
                    "method": "Light32 DDIM20",
                    "runs": light_row.get("runs"),
                    "IAE_mean": light_row.get("IAE_mean"),
                    "after_change_IAE_mean": light_row.get("after_change_IAE_mean"),
                    "mean_pwm_mean": light_row.get("mean_pwm_mean"),
                    "generator_p90_mean": light_row.get(
                        "latency_generator_duration_sec_p90_mean"
                    ),
                    "source_to_apply_p90_mean": light_row.get(
                        "latency_source_to_apply_sec_p90_mean"
                    ),
                }
            ]
        )
        c1.dataframe(style_numeric(light_summary), use_container_width=True, hide_index=True)
    else:
        c1.info("No light32 real-motor summary found.")

    if not bench_df.empty:
        bench_show = bench_df[["model", "mean_sec", "p50_sec", "p90_sec", "max_sec"]]
        c2.dataframe(style_numeric(bench_show), use_container_width=True, hide_index=True)
        c2.bar_chart(bench_df.set_index("model")[["p90_sec"]])
    else:
        c2.info("No inference benchmark found.")

    st.subheader("Repeat-Level Stability")
    if not repeats_df.empty:
        repeat_cols = [
            "method",
            "IAE",
            "after_change_IAE",
            "mean_pwm",
            "max_pwm",
            "schedule_chunk_accepted_count",
            "schedule_gain_applied_count",
            "latency_generator_duration_sec_p90",
            "latency_source_to_apply_sec_p90",
        ]
        st.dataframe(
            style_numeric(repeats_df[[c for c in repeat_cols if c in repeats_df.columns]]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Repeat-level file is not available.")

with chunks_tab:
    st.subheader("Segment-Level Behavior")
    st.markdown(
        '<div class="section-caption">How the selected method allocates gains and PWM across target transitions.</div>',
        unsafe_allow_html=True,
    )

    if segment_df.empty:
        st.warning("No segment comparison data available.")
    else:
        selected_methods = st.multiselect(
            "Methods",
            options=sorted(segment_df["method"].unique()),
            default=["Full_Diffusion_DDIM20"],
        )
        seg = segment_df[segment_df["method"].isin(selected_methods)].copy()
        seg_mean = (
            seg.groupby(["method", "segment"], as_index=False)
            .agg(
                IAE=("IAE", "mean"),
                mean_pwm=("mean_pwm", "mean"),
                kp_mean=("kp_mean", "mean"),
                ki_mean=("ki_mean", "mean"),
                kd_mean=("kd_mean", "mean"),
            )
        )
        st.dataframe(style_numeric(seg_mean), hide_index=True, use_container_width=True)

        c1, c2 = st.columns(2)
        if not seg_mean.empty:
            c1.markdown("**Segment IAE**")
            c1.bar_chart(seg_mean.pivot(index="segment", columns="method", values="IAE"))
            c2.markdown("**Segment Mean PWM**")
            c2.bar_chart(
                seg_mean.pivot(index="segment", columns="method", values="mean_pwm")
            )

        st.markdown("**Mean Gain by Segment**")
        gain_mean = seg_mean.set_index(["segment", "method"])[
            ["kp_mean", "ki_mean", "kd_mean"]
        ]
        st.bar_chart(gain_mean)

    st.subheader("90 -> 73 RPM Deceleration Snapshot")
    if gain_df.empty:
        st.warning("No deceleration gain snapshot found.")
    else:
        dd20 = gain_df[gain_df["method"].eq("Full_Diffusion_DDIM20")].copy()
        summary = pd.DataFrame(
            [
                {
                    "window": "0.0-0.5 s",
                    "Kp": dd20["kp_0_5s"].mean(),
                    "Ki": dd20["ki_0_5s"].mean(),
                    "Kd": dd20["kd_0_5s"].mean(),
                },
                {
                    "window": "1.0-2.0 s",
                    "Kp": dd20["kp_1_2s"].mean(),
                    "Ki": dd20["ki_1_2s"].mean(),
                    "Kd": dd20["kd_1_2s"].mean(),
                },
            ]
        )
        c1, c2 = st.columns([1.1, 1.0])
        c1.dataframe(style_numeric(summary), hide_index=True, use_container_width=True)
        c1.bar_chart(summary.set_index("window")[["Kp", "Ki", "Kd"]])

        rpm_pwm = pd.DataFrame(
            [
                ["RPM at 1 s", dd20["rpm_1s"].mean()],
                ["RPM at 2 s", dd20["rpm_2s_end"].mean()],
                ["Mean PWM", dd20["pwm_mean"].mean()],
            ],
            columns=["Metric", "Value"],
        )
        c2.dataframe(style_numeric(rpm_pwm), hide_index=True, use_container_width=True)
        c2.markdown(
            """
            <div class="note">
            The final DDIM20 policy consistently used high Ki and max Kd during
            the 90 -> 73 RPM deceleration segment. This data-driven gain pattern
            is one reason the full DDIM20 model outperformed longer DDIM runs and
            the lightweight model.
            </div>
            """,
            unsafe_allow_html=True,
        )

with realtime_tab:
    st.subheader("Realtime Monitor")
    st.markdown(
        '<div class="section-caption">Optional live view for new controller logs. Final results above do not depend on this tab.</div>',
        unsafe_allow_html=True,
    )

    if st.button("Refresh realtime data", type="secondary"):
        st.rerun()

    log_file = get_latest_log_file()
    if log_file is None:
        st.info("No realtime log CSV found in the configured log directory.")
    else:
        st.caption(f"Current log file: `{log_file}`")
        try:
            live_df = pd.read_csv(log_file)
        except Exception as exc:
            st.error(f"Failed to read realtime log: {exc}")
            live_df = pd.DataFrame()

        if live_df.empty:
            st.warning("Realtime log is empty.")
        else:
            latest = live_df.iloc[-1]
            live_metrics = []
            for label, col, suffix in [
                ("Target", "target", " RPM"),
                ("Current", "current", " RPM"),
                ("Error", "error", ""),
                ("PWM", "pwm", ""),
                ("Kp", "kp", ""),
                ("Ki", "ki", ""),
                ("Kd", "kd", ""),
            ]:
                if col in live_df.columns:
                    live_metrics.append((label, latest[col], suffix))
            if live_metrics:
                metric_row(live_metrics[:4])
                if len(live_metrics) > 4:
                    metric_row(live_metrics[4:])

            if "time" in live_df.columns:
                indexed = live_df.set_index("time")
                c1, c2 = st.columns(2)
                rpm_cols = [c for c in ["target", "current", "rpm"] if c in indexed.columns]
                if rpm_cols:
                    c1.markdown("**RPM Tracking**")
                    c1.line_chart(indexed[rpm_cols])
                pwm_cols = [c for c in ["pwm", "raw_pwm"] if c in indexed.columns]
                if pwm_cols:
                    c2.markdown("**PWM**")
                    c2.line_chart(indexed[pwm_cols])
                gain_cols = [c for c in ["kp", "ki", "kd"] if c in indexed.columns]
                if gain_cols:
                    st.markdown("**Applied Gains**")
                    st.line_chart(indexed[gain_cols])
            st.dataframe(live_df.tail(30), use_container_width=True)

with artifacts_tab:
    st.subheader("Final Artifact Bundle")
    st.markdown(
        '<div class="section-caption">Files committed for the compact final result.</div>',
        unsafe_allow_html=True,
    )

    files = []
    for path in sorted(ARTIFACT_DIR.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(BASE_DIR)),
                    "size_kb": path.stat().st_size / 1024,
                }
            )
    st.dataframe(style_numeric(pd.DataFrame(files)), hide_index=True, use_container_width=True)

    st.subheader("Reproduce Final Runtime")
    st.code(
        "\n".join(
            [
                "call C:\\Users\\Jaewook\\anaconda3\\Scripts\\activate.bat tensorflow",
                "python src\\gain_recommender_server.py ^",
                "  --generator diffusion_unet_gain_chunk ^",
                "  --diffusion-unet-model-path artifacts\\final_ddim20\\models\\diffusion_gain_chunk_unet_balanced1000_global_topk_full_20260508_193250.joblib ^",
                "  --diffusion-ddim-steps 20 ^",
                "  --diffusion-sample-count 1 ^",
                "  --inference-delay 0.5 ^",
                "  --disable-artificial-inference-sleep ^",
                "  --chunk-horizon-steps 20 ^",
                "  --command-min-interval 0.5",
            ]
        ),
        language="bat",
    )
    st.code(
        "\n".join(
            [
                "python src\\local_kafka_controller.py ^",
                "  --schedule-apply-mode delay_aware ^",
                "  --sim-time 20 ^",
                "  --target-sequence \"70,95,90,73\" ^",
                "  --target-change-times \"5,10,15\" ^",
                "  --run-label final_ddim20",
            ]
        ),
        language="bat",
    )
