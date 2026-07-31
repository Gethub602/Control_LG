import sys
from pathlib import Path
from datetime import datetime
import time
import uuid
import argparse

import numpy as np
import pandas as pd

try:
    from kafka import KafkaProducer, KafkaConsumer, TopicPartition
except ImportError:
    KafkaProducer = None
    KafkaConsumer = None

# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from pid_controller import PIDController
from motor_interface import SimMotorInterface, ESP32MotorInterface
from config import (
    DT,
    PWM_MIN,
    PWM_MAX,
    RESULTS_DIR,

    MOTOR_BACKEND,
    ESP32_PORT,
    ESP32_BAUDRATE,
    ESP32_TIMEOUT,
    REAL_PWM_MIN,
    REAL_PWM_MAX,
    REAL_PWM_SOFT_LIMIT,

    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,

    PID_GAIN_DB,
    GAIN_DB_MODE,

    ESP32_REAL_PID_GAIN_DB,
    ESP32_REAL_GAIN_DB_MODE,

    KP_MIN,
    KP_MAX,
    KI_MIN,
    KI_MAX,
    KD_MIN,
    KD_MAX,

    PWM_SOFT_LIMIT,
    SATURATION_CONSECUTIVE_STEPS,
    SATURATION_ERROR_THRESHOLD_RATIO,
    SATURATION_KP_DECAY,
    SATURATION_KI_DECAY,
    SATURATION_KD_DECAY,
    SATURATION_MIN_GAIN_SCALE,
    SATURATION_RECOVERY_RATE,
)

from kafka_config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TOPIC_MOTOR_STATE,
    TOPIC_GAIN_COMMAND,
    TOPIC_SCHEDULE_CHUNK,
    DEVICE_ID,
    LOCAL_CONTROLLER_GROUP_ID,
)

from message_schema import (
    json_serializer,
    json_deserializer,
    make_motor_state_message,
    is_valid_gain_command,
)

from schedule_buffer import ScheduleBuffer
from schedule_schema import PAYLOAD_KIND_GAIN


# ============================================================
# Experiment settings
# ============================================================

RESULT_DIR = RESULTS_DIR / "kafka_control"

SIM_TIME = 10.0

# ESP32 serial 제어는 10 Hz 기준으로 고정
CONTROL_DT = 0.10 if MOTOR_BACKEND == "esp32" else DT
N_STEPS = int(SIM_TIME / CONTROL_DT)

# ESP32 실물 보호용 PWM 변화율 제한
# 0.10 s마다 PWM이 최대 20만 변하게 제한
ESP32_LOCAL_PWM_RATE_LIMIT = 50.0

# ESP32 server gain safety guard
ESP32_SAFE_KP_MAX = 2.0
ESP32_SAFE_KI_MAX = 2.5
ESP32_SAFE_KD_MAX = 0.5

# Backend별 target 및 disturbance 설정
if MOTOR_BACKEND == "esp32":
    TARGET_BEFORE = 70.0
    TARGET_AFTER = 100.0
    TARGET_CHANGE_TIME = 3.0

    USE_DISTURBANCE = False
    DISTURBANCE_MODE = "none"
    DISTURBANCE_START_TIME = 5.0
    DISTURBANCE_END_TIME = 7.0
    DISTURBANCE_MAGNITUDE = 0.0

    # ESP32_REAL_PID_GAIN_DB가 server recommender에 연결된 뒤 True로 사용
    APPLY_SERVER_GAIN_COMMAND = True

elif MOTOR_BACKEND == "simulation":
    TARGET_BEFORE = 100.0
    TARGET_AFTER = 200.0
    TARGET_CHANGE_TIME = 3.0

    USE_DISTURBANCE = True
    DISTURBANCE_MODE = "pulse"
    DISTURBANCE_START_TIME = 5.0
    DISTURBANCE_END_TIME = 7.0
    DISTURBANCE_MAGNITUDE = 20.0

    APPLY_SERVER_GAIN_COMMAND = True

else:
    raise ValueError(
        f"Unknown MOTOR_BACKEND: {MOTOR_BACKEND}. "
        "Use 'simulation' or 'esp32'."
    )

TARGET_SEQUENCE = []
TARGET_CHANGE_TIMES = []

USE_SATURATION_AWARE_GAIN = True

RUN_ID = (
    f"kafka_{MOTOR_BACKEND}_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
    f"{uuid.uuid4().hex[:6]}"
)

STATE_PUBLISH_EVERY_STEPS = 1

# Prefer delay-aware chunk schedules when available. Legacy single gain commands
# remain available as a fallback/baseline path.
APPLY_SERVER_SCHEDULE_CHUNK = True
SCHEDULE_APPLY_MODE = "delay_aware"  # "delay_aware" or "naive"
SCHEDULE_BUFFER_MAX_CHUNKS = 8


# ============================================================
# Backend utility
# ============================================================

def get_backend_pwm_limits():
    """
    Backend에 따라 사용할 PWM 제한값을 반환한다.
    """

    if MOTOR_BACKEND == "esp32":
        return REAL_PWM_MIN, REAL_PWM_MAX, REAL_PWM_SOFT_LIMIT

    if MOTOR_BACKEND == "simulation":
        return PWM_MIN, PWM_MAX, PWM_SOFT_LIMIT

    raise ValueError(
        f"Unknown MOTOR_BACKEND: {MOTOR_BACKEND}. "
        "Use 'simulation' or 'esp32'."
    )


def create_motor_backend():
    """
    config.py의 MOTOR_BACKEND 값에 따라 motor interface 생성.
    """

    control_pwm_min, control_pwm_max, _ = get_backend_pwm_limits()

    if MOTOR_BACKEND == "simulation":
        motor = SimMotorInterface(
            initial_value=0.0,
            dt=CONTROL_DT,
            pwm_min=control_pwm_min,
            pwm_max=control_pwm_max,
            use_disturbance=USE_DISTURBANCE,
            disturbance_mode=DISTURBANCE_MODE,
            disturbance_start_time=DISTURBANCE_START_TIME,
            disturbance_end_time=DISTURBANCE_END_TIME,
            disturbance_magnitude=DISTURBANCE_MAGNITUDE,
            disturbance_freq=1.0,
        )
        return motor

    if MOTOR_BACKEND == "esp32":
        motor = ESP32MotorInterface(
            port=ESP32_PORT,
            baudrate=ESP32_BAUDRATE,
            timeout=ESP32_TIMEOUT,
            pwm_min=control_pwm_min,
            pwm_max=control_pwm_max,
        )
        return motor

    raise ValueError(
        f"Unknown MOTOR_BACKEND: {MOTOR_BACKEND}. "
        "Use 'simulation' or 'esp32'."
    )


def apply_pwm_rate_limit(pwm_cmd: float, prev_pwm: float, rate_limit: float) -> float:
    """
    PWM 변화율 제한.
    ESP32 실물 모터에서 급격한 PWM 변화를 줄이기 위해 사용한다.
    """

    delta = pwm_cmd - prev_pwm
    delta = np.clip(delta, -rate_limit, rate_limit)
    return float(prev_pwm + delta)


# ============================================================
# Target / gain DB utility
# ============================================================

def get_target_at_time(t: float) -> float:
    if TARGET_SEQUENCE:
        target = float(TARGET_SEQUENCE[0])
        for change_time, next_target in zip(TARGET_CHANGE_TIMES, TARGET_SEQUENCE[1:]):
            if t >= float(change_time):
                target = float(next_target)
            else:
                break
        return target

    if t < TARGET_CHANGE_TIME:
        return TARGET_BEFORE
    return TARGET_AFTER


def parse_float_list(text: str):
    return [
        float(item.strip())
        for item in str(text).split(",")
        if item.strip()
    ]


def get_target_profile_label() -> str:
    if TARGET_SEQUENCE:
        return "->".join(f"{target:g}" for target in TARGET_SEQUENCE)
    return f"{TARGET_BEFORE:g}->{TARGET_AFTER:g}"


def get_first_target_change_time() -> float:
    if TARGET_SEQUENCE and TARGET_CHANGE_TIMES:
        return float(TARGET_CHANGE_TIMES[0])
    return float(TARGET_CHANGE_TIME)


def interpolate_gain_from_db(
    target: float,
    gain_db: dict,
    mode: str,
    fallback_gain: tuple,
):
    """
    공통 gain DB lookup 함수.
    mode='linear'이면 target 사이를 선형 보간한다.
    """

    if not gain_db:
        return fallback_gain

    target = float(target)
    db_targets = sorted([float(k) for k in gain_db.keys()])

    if target in db_targets:
        gains = gain_db[target]
        return gains["kp"], gains["ki"], gains["kd"]

    if mode == "nearest":
        nearest_target = min(db_targets, key=lambda x: abs(x - target))
        gains = gain_db[nearest_target]
        return gains["kp"], gains["ki"], gains["kd"]

    if mode == "linear":
        if target <= db_targets[0]:
            gains = gain_db[db_targets[0]]
            return gains["kp"], gains["ki"], gains["kd"]

        if target >= db_targets[-1]:
            gains = gain_db[db_targets[-1]]
            return gains["kp"], gains["ki"], gains["kd"]

        for i in range(len(db_targets) - 1):
            t_low = db_targets[i]
            t_high = db_targets[i + 1]

            if t_low <= target <= t_high:
                ratio = (target - t_low) / (t_high - t_low)

                g_low = gain_db[t_low]
                g_high = gain_db[t_high]

                kp = g_low["kp"] + ratio * (g_high["kp"] - g_low["kp"])
                ki = g_low["ki"] + ratio * (g_high["ki"] - g_low["ki"])
                kd = g_low["kd"] + ratio * (g_high["kd"] - g_low["kd"])

                return kp, ki, kd

    nearest_target = min(db_targets, key=lambda x: abs(x - target))
    gains = gain_db[nearest_target]
    return gains["kp"], gains["ki"], gains["kd"]


def get_gain_from_db(target: float):
    """
    Backend별 gain DB에서 target 기반 gain을 가져온다.

    - simulation: PID_GAIN_DB 사용
    - esp32: ESP32_REAL_PID_GAIN_DB 사용
    """

    if MOTOR_BACKEND == "esp32":
        return interpolate_gain_from_db(
            target=target,
            gain_db=ESP32_REAL_PID_GAIN_DB,
            mode=ESP32_REAL_GAIN_DB_MODE,
            fallback_gain=(1.2, 0.7, 0.0),
        )

    return interpolate_gain_from_db(
        target=target,
        gain_db=PID_GAIN_DB,
        mode=GAIN_DB_MODE,
        fallback_gain=(DEFAULT_KP, DEFAULT_KI, DEFAULT_KD),
    )


def is_safe_esp32_gain(kp: float, ki: float, kd: float) -> bool:
    """
    ESP32 실물 모터 보호용 gain guard.
    서버가 실수로 simulation용 큰 gain을 보내도 local에서 적용하지 않음.
    """

    if kp > ESP32_SAFE_KP_MAX:
        return False

    if ki > ESP32_SAFE_KI_MAX:
        return False

    if kd > ESP32_SAFE_KD_MAX:
        return False

    if kp < 0.0 or ki < 0.0 or kd < 0.0:
        return False

    return True


def set_pid_gains_preserving_integral_term(
    pid: PIDController,
    kp: float,
    ki: float,
    kd: float,
    preserve_integral_term: bool = False,
):
    """
    Apply new gains without creating a step change in the I contribution.

    A server/schedule gain update can increase Ki while the controller already
    has accumulated integral from the previous target. Rescaling integral keeps
    old_ki * integral approximately equal to new_ki * integral.
    """

    if preserve_integral_term:
        old_ki = float(pid.ki)
        new_ki = float(ki)

        if abs(new_ki) > 1e-12:
            if abs(old_ki) > 1e-12:
                pid.integral *= old_ki / new_ki
            else:
                pid.integral = 0.0

    pid.set_gains(kp, ki, kd)


# ============================================================
# Local saturation-aware safety layer
# ============================================================

class LocalSaturationAwareGainManager:
    """
    Local safety layer.

    서버 gain command가 늦거나 없어도, local에서 PWM soft-limit 위험을 감지하면
    base gain에 곱해지는 scale을 조정한다.

    중요:
    - PID 현재 gain에 다시 scale을 곱하지 않는다.
    - 항상 base_gain × scale 구조로만 적용한다.
    """

    def __init__(self, pwm_soft_limit: float):
        self.pwm_soft_limit = float(pwm_soft_limit)

        self.kp_scale = 1.0
        self.ki_scale = 1.0
        self.kd_scale = 1.0

        self.saturation_counter = 0
        self.saturation_active = False

        self.gain_update_flag = False
        self.last_update_reason = "init"

    def update(self, target: float, error: float, pwm: float):
        self.gain_update_flag = False
        self.last_update_reason = "none"

        error_ratio = abs(error) / max(abs(target), 1e-6)

        high_pwm_risk = pwm >= self.pwm_soft_limit
        meaningful_error = error_ratio >= SATURATION_ERROR_THRESHOLD_RATIO

        if high_pwm_risk and meaningful_error:
            self.saturation_counter += 1
        else:
            self.saturation_counter = 0

        if self.saturation_counter >= SATURATION_CONSECUTIVE_STEPS:
            old_kp_scale = self.kp_scale
            old_ki_scale = self.ki_scale
            old_kd_scale = self.kd_scale

            self.kp_scale = max(
                SATURATION_MIN_GAIN_SCALE,
                self.kp_scale * SATURATION_KP_DECAY,
            )
            self.ki_scale = max(
                SATURATION_MIN_GAIN_SCALE,
                self.ki_scale * SATURATION_KI_DECAY,
            )
            self.kd_scale = max(
                SATURATION_MIN_GAIN_SCALE,
                self.kd_scale * SATURATION_KD_DECAY,
            )

            self.saturation_active = True

            if (
                abs(self.kp_scale - old_kp_scale) > 1e-12
                or abs(self.ki_scale - old_ki_scale) > 1e-12
                or abs(self.kd_scale - old_kd_scale) > 1e-12
            ):
                self.gain_update_flag = True
                self.last_update_reason = "local_saturation_gain_reduction"

        else:
            if self.kp_scale < 1.0 or self.ki_scale < 1.0 or self.kd_scale < 1.0:
                self.kp_scale = min(1.0, self.kp_scale + SATURATION_RECOVERY_RATE)
                self.ki_scale = min(1.0, self.ki_scale + SATURATION_RECOVERY_RATE)
                self.kd_scale = min(1.0, self.kd_scale + SATURATION_RECOVERY_RATE)

                self.gain_update_flag = True
                self.last_update_reason = "local_saturation_recovery"

            self.saturation_active = False

    def apply_scale(self, base_kp: float, base_ki: float, base_kd: float):
        kp = float(np.clip(base_kp * self.kp_scale, KP_MIN, KP_MAX))
        ki = float(np.clip(base_ki * self.ki_scale, KI_MIN, KI_MAX))
        kd = float(np.clip(base_kd * self.kd_scale, KD_MIN, KD_MAX))

        return kp, ki, kd

    def get_state(self):
        return {
            "kp_scale": self.kp_scale,
            "ki_scale": self.ki_scale,
            "kd_scale": self.kd_scale,
            "saturation_counter": self.saturation_counter,
            "saturation_active": self.saturation_active,
            "gain_update_flag": self.gain_update_flag,
            "last_update_reason": self.last_update_reason,
        }


# ============================================================
# Kafka
# ============================================================

def create_producer():
    if KafkaProducer is None:
        raise ImportError(
            "kafka-python is required to run the Kafka controller. "
            "Install it in the active Python environment."
        )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=json_serializer,
    )
    return producer


def _subscribe_latest(consumer, topic: str):
    """
    Attach a consumer to every partition of `topic`, positioned at the end.

    Consumer-group subscription is deliberately avoided. With kafka-python 3.0.9
    against a Kafka 3.9 broker the group join never completes when poll() is
    called with the very short timeouts this control loop uses, so the consumer
    is never assigned a partition and silently receives nothing. Manual
    assignment also removes rebalance pauses, which a real-time loop does not
    want anyway.

    The priming poll matters: without one, the fetcher has not issued its first
    request yet and subsequent 1 ms polls return empty forever.
    """
    partitions = consumer.partitions_for_topic(topic)
    if not partitions:
        raise RuntimeError(
            f"Topic {topic!r} has no partitions. Is the broker running and the "
            "topic created?"
        )
    consumer.assign([TopicPartition(topic, p) for p in sorted(partitions)])
    consumer.seek_to_end()
    consumer.poll(timeout_ms=1)
    return consumer


def create_gain_command_consumer():
    if KafkaConsumer is None:
        raise ImportError(
            "kafka-python is required to run the Kafka controller. "
            "Install it in the active Python environment."
        )

    consumer = KafkaConsumer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=json_deserializer,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        consumer_timeout_ms=1,
    )
    return _subscribe_latest(consumer, TOPIC_GAIN_COMMAND)


def create_schedule_chunk_consumer():
    if KafkaConsumer is None:
        raise ImportError(
            "kafka-python is required to run the Kafka controller. "
            "Install it in the active Python environment."
        )

    consumer = KafkaConsumer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=json_deserializer,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        consumer_timeout_ms=1,
    )
    return _subscribe_latest(consumer, TOPIC_SCHEDULE_CHUNK)


def poll_latest_gain_command(consumer):
    """
    Kafka에서 도착한 gain command 중 가장 최신 메시지 하나만 반환.
    없으면 None.
    """

    latest_command = None

    records = consumer.poll(timeout_ms=1)

    if not records:
        return None

    for _, messages in records.items():
        for msg in messages:
            latest_command = msg.value

    return latest_command


def poll_schedule_chunks(consumer):
    chunks = []

    records = consumer.poll(timeout_ms=1)

    if not records:
        return chunks

    for _, messages in records.items():
        for msg in messages:
            chunks.append(msg.value)

    return chunks


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local Kafka motor controller."
    )

    parser.add_argument(
        "--schedule-apply-mode",
        choices=["delay_aware", "naive"],
        default=SCHEDULE_APPLY_MODE,
        help="How to apply received schedule chunks.",
    )
    parser.add_argument(
        "--disable-schedule-chunk",
        action="store_true",
        help="Disable schedule chunk consumption and use legacy/fallback behavior.",
    )
    parser.add_argument(
        "--disable-legacy-gain-command",
        action="store_true",
        help="Disable legacy single gain command fallback from the server.",
    )
    parser.add_argument(
        "--sim-time",
        type=float,
        default=SIM_TIME,
        help="Experiment duration in seconds.",
    )
    parser.add_argument(
        "--target-before",
        type=float,
        default=TARGET_BEFORE,
        help="Target before the step change.",
    )
    parser.add_argument(
        "--target-after",
        type=float,
        default=TARGET_AFTER,
        help="Target after the step change.",
    )
    parser.add_argument(
        "--target-change-time",
        type=float,
        default=TARGET_CHANGE_TIME,
        help="Time of target step change in seconds.",
    )
    parser.add_argument(
        "--target-sequence",
        type=str,
        default="",
        help=(
            "Comma-separated multi-step target profile, e.g. '70,85,100'. "
            "When set, this overrides --target-before/--target-after."
        ),
    )
    parser.add_argument(
        "--target-change-times",
        type=str,
        default="",
        help=(
            "Comma-separated change times for --target-sequence, e.g. '3,6'. "
            "Count must be one less than the target sequence length."
        ),
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Optional label appended to output filenames.",
    )

    return parser.parse_args()


def apply_runtime_args(args):
    global SIM_TIME
    global N_STEPS
    global APPLY_SERVER_SCHEDULE_CHUNK
    global APPLY_SERVER_GAIN_COMMAND
    global SCHEDULE_APPLY_MODE
    global TARGET_BEFORE
    global TARGET_AFTER
    global TARGET_CHANGE_TIME
    global TARGET_SEQUENCE
    global TARGET_CHANGE_TIMES

    SIM_TIME = float(args.sim_time)
    N_STEPS = int(SIM_TIME / CONTROL_DT)
    SCHEDULE_APPLY_MODE = args.schedule_apply_mode
    TARGET_BEFORE = float(args.target_before)
    TARGET_AFTER = float(args.target_after)
    TARGET_CHANGE_TIME = float(args.target_change_time)
    TARGET_SEQUENCE = []
    TARGET_CHANGE_TIMES = []

    if args.target_sequence:
        TARGET_SEQUENCE = parse_float_list(args.target_sequence)
        TARGET_CHANGE_TIMES = parse_float_list(args.target_change_times)

        if len(TARGET_SEQUENCE) < 2:
            raise ValueError("--target-sequence must contain at least two targets")

        if len(TARGET_CHANGE_TIMES) != len(TARGET_SEQUENCE) - 1:
            raise ValueError(
                "--target-change-times count must be one less than "
                "--target-sequence count"
            )

        if any(
            TARGET_CHANGE_TIMES[i] >= TARGET_CHANGE_TIMES[i + 1]
            for i in range(len(TARGET_CHANGE_TIMES) - 1)
        ):
            raise ValueError("--target-change-times must be strictly increasing")

        TARGET_BEFORE = float(TARGET_SEQUENCE[0])
        TARGET_AFTER = float(TARGET_SEQUENCE[-1])
        TARGET_CHANGE_TIME = float(TARGET_CHANGE_TIMES[0])

    if args.disable_schedule_chunk:
        APPLY_SERVER_SCHEDULE_CHUNK = False

    if args.disable_legacy_gain_command:
        APPLY_SERVER_GAIN_COMMAND = False


# ============================================================
# Metrics
# ============================================================

def compute_dt(time_arr: np.ndarray):
    if len(time_arr) <= 1:
        return np.zeros_like(time_arr)

    dt_arr = np.diff(time_arr, prepend=time_arr[0])
    dt_arr[0] = 0.0
    return dt_arr


def finite_series_stats(df: pd.DataFrame, column: str, prefix: str) -> dict:
    if column not in df.columns:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
        }

    values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_max": float(np.max(values)),
    }


def calculate_metrics(df: pd.DataFrame) -> dict:
    control_pwm_min, control_pwm_max, _ = get_backend_pwm_limits()

    time_arr = df["time"].to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    current = df["current"].to_numpy(dtype=float)

    # metrics는 저장된 current 기준으로 measured error를 항상 재계산한다.
    measured_error = target - current

    pwm = df["pwm"].to_numpy(dtype=float)
    disturbance = df["disturbance"].to_numpy(dtype=float)

    dt_arr = compute_dt(time_arr)
    abs_error = np.abs(measured_error)
    abs_pwm = np.abs(pwm)

    iae = float(np.sum(abs_error * dt_arr))
    final_error = float(abs_error[-1])
    mean_abs_error = float(np.mean(abs_error))

    first_change_time = get_first_target_change_time()
    after_change_mask = time_arr >= first_change_time

    if after_change_mask.any():
        after_change_iae = float(
            np.sum(abs_error[after_change_mask] * dt_arr[after_change_mask])
        )
        after_change_max_error = float(np.max(abs_error[after_change_mask]))
    else:
        after_change_iae = np.nan
        after_change_max_error = np.nan

    disturbance_mask = np.abs(disturbance) > 1e-9

    if disturbance_mask.any():
        disturbance_iae = float(
            np.sum(abs_error[disturbance_mask] * dt_arr[disturbance_mask])
        )
        disturbance_max_error = float(np.max(abs_error[disturbance_mask]))
        disturbance_min_current = float(np.min(current[disturbance_mask]))
    else:
        disturbance_iae = np.nan
        disturbance_max_error = np.nan
        disturbance_min_current = np.nan

    high_saturation = pwm >= control_pwm_max - 1e-9
    saturation_ratio_percent = float(np.mean(high_saturation) * 100.0)
    saturation_duration = float(np.sum(dt_arr[high_saturation]))

    near_high_saturation = pwm >= control_pwm_max - 0.1 * (
        control_pwm_max - control_pwm_min
    )
    near_high_saturation_ratio_percent = float(np.mean(near_high_saturation) * 100.0)

    tolerance = 0.02 * abs(TARGET_AFTER)
    settling_time_after_change = np.nan

    after_indices = np.where(after_change_mask)[0]

    for idx in after_indices:
        if np.all(np.abs(target[idx:] - current[idx:]) <= tolerance):
            settling_time_after_change = float(time_arr[idx] - first_change_time)
            break

    segment_metrics = {}
    for segment_target in sorted(set(float(value) for value in target)):
        segment_mask = np.isclose(target, segment_target)
        if not segment_mask.any():
            continue

        label = f"target_{segment_target:g}".replace(".", "p")
        segment_metrics[f"{label}_IAE"] = float(
            np.sum(abs_error[segment_mask] * dt_arr[segment_mask])
        )
        segment_metrics[f"{label}_mean_abs_error"] = float(
            np.mean(abs_error[segment_mask])
        )
        segment_metrics[f"{label}_max_error"] = float(
            np.max(abs_error[segment_mask])
        )

    schedule_df = df[df["schedule_source"] == "schedule_chunk"].copy()
    latency_metrics = {}
    latency_metrics.update(
        finite_series_stats(
            schedule_df,
            "schedule_source_to_accept_sec",
            "latency_source_to_accept_sec",
        )
    )
    latency_metrics.update(
        finite_series_stats(
            schedule_df,
            "schedule_source_to_apply_sec",
            "latency_source_to_apply_sec",
        )
    )
    latency_metrics.update(
        finite_series_stats(
            schedule_df,
            "schedule_generated_to_accept_sec",
            "latency_generated_to_accept_sec",
        )
    )
    latency_metrics.update(
        finite_series_stats(
            schedule_df,
            "schedule_generator_duration_sec",
            "latency_generator_duration_sec",
        )
    )
    latency_metrics.update(
        finite_series_stats(
            schedule_df,
            "schedule_timing_slack_sec",
            "latency_timing_slack_sec",
        )
    )

    return {
        "backend": MOTOR_BACKEND,
        "control_dt": CONTROL_DT,
        "target_before": TARGET_BEFORE,
        "target_after": TARGET_AFTER,
        "target_profile": get_target_profile_label(),
        "target_change_times": ",".join(
            f"{change_time:g}" for change_time in TARGET_CHANGE_TIMES
        )
        if TARGET_CHANGE_TIMES
        else f"{TARGET_CHANGE_TIME:g}",

        "IAE": iae,
        "mean_abs_error": mean_abs_error,
        "final_error": final_error,

        "after_change_IAE": after_change_iae,
        "after_change_max_error": after_change_max_error,

        "disturbance_IAE": disturbance_iae,
        "disturbance_max_error": disturbance_max_error,
        "disturbance_min_current": disturbance_min_current,

        "settling_time_after_change": settling_time_after_change,

        "mean_pwm": float(np.mean(abs_pwm)),
        "total_pwm": float(np.sum(abs_pwm * dt_arr)),
        "max_pwm": float(np.max(pwm)),

        "saturation_ratio_percent": saturation_ratio_percent,
        "saturation_duration": saturation_duration,
        "near_high_saturation_ratio_percent": near_high_saturation_ratio_percent,

        "server_gain_applied_count": int(
            (df["gain_update_reason"] == "server_gain_applied").sum()
        ),
        "schedule_gain_applied_count": int(
            (df["gain_update_reason"] == "schedule_chunk_gain_applied").sum()
        ),
        "schedule_apply_mode": str(df["schedule_apply_mode"].dropna().iloc[-1])
        if "schedule_apply_mode" in df.columns and not df["schedule_apply_mode"].dropna().empty
        else "unknown",
        "schedule_chunk_accepted_count": int(
            df["schedule_chunk_accept_count"].sum()
        )
        if "schedule_chunk_accept_count" in df.columns
        else 0,
        "schedule_fallback_count": int(df["schedule_fallback_used"].sum())
        if "schedule_fallback_used" in df.columns
        else 0,
        "server_gain_discarded_count": int(df["gain_command_discarded"].sum()),
        "duplicate_gain_discard_count": int(
            (df["gain_command_discard_reason"] == "duplicate_gain_command").sum()
        ),
        "unsafe_gain_discard_count": int(
            (df["gain_command_discard_reason"] == "unsafe_esp32_server_gain").sum()
        ),

        "local_gain_reduction_count": int(
            (df["local_update_reason"] == "local_saturation_gain_reduction").sum()
        ),
        "local_gain_recovery_count": int(
            (df["local_update_reason"] == "local_saturation_recovery").sum()
        ),
        "min_kp_scale": float(df["kp_scale"].min()),
        "min_ki_scale": float(df["ki_scale"].min()),
        **latency_metrics,
        **segment_metrics,
    }


# ============================================================
# Main control loop
# ============================================================

def run_local_kafka_controller(run_label: str = ""):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    control_pwm_min, control_pwm_max, control_pwm_soft_limit = get_backend_pwm_limits()

    print("=" * 80)
    print("Local Kafka Controller Start")
    print("=" * 80)
    print(f"Run ID: {RUN_ID}")
    print(f"Device ID: {DEVICE_ID}")
    print(f"Kafka broker: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Publish topic: {TOPIC_MOTOR_STATE}")
    print(f"Consume topic: {TOPIC_GAIN_COMMAND}")
    print(f"Consume schedule topic: {TOPIC_SCHEDULE_CHUNK}")
    print(f"Motor backend: {MOTOR_BACKEND}")
    print(f"Control DT: {CONTROL_DT}")
    print(
        f"Target profile: {get_target_profile_label()} "
        f"at {TARGET_CHANGE_TIMES if TARGET_CHANGE_TIMES else [TARGET_CHANGE_TIME]}s"
    )
    print(f"Disturbance: {USE_DISTURBANCE}, mode={DISTURBANCE_MODE}")
    print(f"Control PWM limit: {control_pwm_min} ~ {control_pwm_max}")
    print(f"Control PWM soft limit: {control_pwm_soft_limit}")
    print(f"Apply server gain command: {APPLY_SERVER_GAIN_COMMAND}")
    print(f"Apply server schedule chunk: {APPLY_SERVER_SCHEDULE_CHUNK}")
    print(f"Schedule apply mode: {SCHEDULE_APPLY_MODE}")
    print(f"Saturation-aware local safety layer: {USE_SATURATION_AWARE_GAIN}")
    if MOTOR_BACKEND == "esp32":
        print(f"ESP32 PWM rate limit: {ESP32_LOCAL_PWM_RATE_LIMIT}")
        print(
            f"ESP32 safe gain guard: "
            f"Kp<={ESP32_SAFE_KP_MAX}, Ki<={ESP32_SAFE_KI_MAX}, Kd<={ESP32_SAFE_KD_MAX}"
        )
    print("=" * 80)

    producer = create_producer()
    command_consumer = create_gain_command_consumer()
    schedule_consumer = create_schedule_chunk_consumer()
    schedule_buffer = ScheduleBuffer(
        run_id=RUN_ID,
        device_id=DEVICE_ID,
        max_chunks=SCHEDULE_BUFFER_MAX_CHUNKS,
    )

    motor = create_motor_backend()

    pid = PIDController(
        dt=CONTROL_DT,
        output_min=control_pwm_min,
        output_max=control_pwm_max,
    )

    target0 = TARGET_BEFORE

    base_kp, base_ki, base_kd = get_gain_from_db(target0)
    last_applied_server_target = target0
    last_applied_server_gain = (base_kp, base_ki, base_kd)

    safety_layer = LocalSaturationAwareGainManager(
        pwm_soft_limit=control_pwm_soft_limit
    )

    kp_applied, ki_applied, kd_applied = safety_layer.apply_scale(
        base_kp,
        base_ki,
        base_kd,
    )
    pid.set_gains(kp_applied, ki_applied, kd_applied)

    motor_state = motor.get_state()
    current = motor_state.current

    # prev_error는 직전 측정 상태 기준 error로 관리
    prev_error = target0 - current

    prev_pwm = 0.0

    latest_seq = 0
    latest_applied_command_seq = -1

    rows = []

    try:
        for step in range(N_STEPS):
            loop_start = time.time()

            t = step * CONTROL_DT
            target = get_target_at_time(t)

            # ------------------------------------------------
            # Kafka schedule chunk consume
            # ------------------------------------------------

            schedule_chunk_accept_reason = "none"
            schedule_chunk_accept_count = 0

            if APPLY_SERVER_SCHEDULE_CHUNK:
                for chunk in poll_schedule_chunks(schedule_consumer):
                    accepted, accept_reason = schedule_buffer.add_chunk(
                        chunk,
                        accepted_control_time=t,
                    )
                    schedule_chunk_accept_reason = accept_reason
                    if accepted:
                        schedule_chunk_accept_count += 1

            if SCHEDULE_APPLY_MODE == "delay_aware":
                schedule_lookup = schedule_buffer.get_item(
                    control_time=t,
                    payload_kind=PAYLOAD_KIND_GAIN,
                )
            elif SCHEDULE_APPLY_MODE == "naive":
                schedule_lookup = schedule_buffer.get_item_naive(
                    control_time=t,
                    payload_kind=PAYLOAD_KIND_GAIN,
                )
            else:
                raise ValueError(
                    "SCHEDULE_APPLY_MODE must be 'delay_aware' or 'naive'"
                )

            schedule_source = "fallback"
            schedule_chunk_id = ""
            schedule_chunk_index = -1
            schedule_discarded_steps = 0
            schedule_chunk_age_sec = np.nan
            schedule_source_to_accept_sec = np.nan
            schedule_source_to_apply_sec = np.nan
            schedule_generated_to_accept_sec = np.nan
            schedule_generated_to_apply_sec = np.nan
            schedule_publish_to_accept_sec = np.nan
            schedule_generator_duration_sec = np.nan
            schedule_state_to_generation_sec = np.nan
            schedule_latency_assumption_sec = np.nan
            schedule_timing_slack_sec = np.nan
            schedule_accepted_control_lag_sec = np.nan
            schedule_model_family = ""
            schedule_generator_mode = ""
            schedule_fallback_used = True

            # ------------------------------------------------
            # Control state before applying new PWM
            # ------------------------------------------------

            control_current = current
            control_error = target - control_current
            error_derivative = (control_error - prev_error) / CONTROL_DT

            # ------------------------------------------------
            # Delay-aware schedule gain application
            # ------------------------------------------------

            gain_update_reason = "none"

            if APPLY_SERVER_SCHEDULE_CHUNK and schedule_lookup.found:
                schedule_item = schedule_lookup.item

                if all(key in schedule_item for key in ["kp", "ki", "kd"]):
                    server_kp = float(schedule_item["kp"])
                    server_ki = float(schedule_item["ki"])
                    server_kd = float(schedule_item["kd"])

                    if MOTOR_BACKEND == "esp32" and not is_safe_esp32_gain(
                        server_kp,
                        server_ki,
                        server_kd,
                    ):
                        schedule_source = "discarded_unsafe_gain"
                    else:
                        new_gain = (server_kp, server_ki, server_kd)
                        base_gain_changed = any(
                            abs(a - b) > 1e-9
                            for a, b in zip(new_gain, last_applied_server_gain)
                        )
                        base_kp, base_ki, base_kd = new_gain
                        last_applied_server_target = float(
                            schedule_item.get("target", target)
                        )
                        last_applied_server_gain = new_gain
                        latest_applied_command_seq = max(
                            latest_applied_command_seq,
                            int(schedule_lookup.chunk.get("source_seq", -1)),
                        )

                        schedule_source = "schedule_chunk"
                        schedule_chunk_id = str(schedule_lookup.chunk["chunk_id"])
                        schedule_chunk_index = int(schedule_lookup.chunk_index)
                        schedule_discarded_steps = int(schedule_lookup.discarded_steps)
                        schedule_chunk_age_sec = float(schedule_lookup.chunk_age_sec)
                        chunk = schedule_lookup.chunk
                        chunk_metadata = dict(chunk.get("metadata", {}))
                        accepted_at_wall_time = schedule_lookup.accepted_at_wall_time
                        accepted_at_control_time = (
                            schedule_lookup.accepted_at_control_time
                        )
                        now_wall_time = time.time()
                        source_timestamp = float(chunk.get("source_timestamp", np.nan))
                        generated_at = float(chunk.get("generated_at", np.nan))
                        publish_start = float(
                            chunk_metadata.get(
                                "schedule_publish_start_wall_time",
                                np.nan,
                            )
                        )
                        schedule_generator_duration_sec = float(
                            chunk_metadata.get("generator_duration_sec", np.nan)
                        )
                        schedule_state_to_generation_sec = float(
                            chunk_metadata.get("state_to_generation_sec", np.nan)
                        )
                        schedule_latency_assumption_sec = float(
                            chunk_metadata.get("latency_assumption_sec", np.nan)
                        )
                        schedule_model_family = str(
                            chunk_metadata.get("model_family", "")
                        )
                        schedule_generator_mode = str(
                            chunk_metadata.get("generator_mode", "")
                        )

                        if accepted_at_wall_time is not None:
                            schedule_source_to_accept_sec = (
                                float(accepted_at_wall_time) - source_timestamp
                            )
                            schedule_generated_to_accept_sec = (
                                float(accepted_at_wall_time) - generated_at
                            )
                            schedule_publish_to_accept_sec = (
                                float(accepted_at_wall_time) - publish_start
                            )

                        if accepted_at_control_time is not None:
                            schedule_accepted_control_lag_sec = (
                                t - float(accepted_at_control_time)
                            )

                        schedule_source_to_apply_sec = now_wall_time - source_timestamp
                        schedule_generated_to_apply_sec = now_wall_time - generated_at
                        schedule_timing_slack_sec = (
                            float(chunk.get("schedule_start_time", np.nan)) - t
                        )
                        schedule_fallback_used = False
                        gain_update_reason = "schedule_chunk_gain_applied"

                        kp_applied, ki_applied, kd_applied = safety_layer.apply_scale(
                            base_kp,
                            base_ki,
                            base_kd,
                        )
                        set_pid_gains_preserving_integral_term(
                            pid,
                            kp_applied,
                            ki_applied,
                            kd_applied,
                            preserve_integral_term=base_gain_changed,
                        )

            # ------------------------------------------------
            # PID compute using pre-step current
            # ------------------------------------------------

            raw_pwm = pid.compute(target, control_current)

            # ESP32 실물에서는 PWM 변화율 제한 적용
            if MOTOR_BACKEND == "esp32":
                pwm = apply_pwm_rate_limit(
                    pwm_cmd=raw_pwm,
                    prev_pwm=prev_pwm,
                    rate_limit=ESP32_LOCAL_PWM_RATE_LIMIT,
                )
                pwm = float(np.clip(pwm, control_pwm_min, control_pwm_max))
            else:
                pwm = raw_pwm

            # ------------------------------------------------
            # Local saturation-aware safety layer
            # based on control_error and current PWM command
            # ------------------------------------------------

            local_update_reason = "none"

            if USE_SATURATION_AWARE_GAIN:
                safety_layer.update(
                    target=target,
                    error=control_error,
                    pwm=pwm,
                )

                local_update_reason = safety_layer.get_state()["last_update_reason"]

            kp_applied, ki_applied, kd_applied = safety_layer.apply_scale(
                base_kp,
                base_ki,
                base_kd,
            )
            pid.set_gains(kp_applied, ki_applied, kd_applied)

            # ------------------------------------------------
            # Kafka gain command consume
            # server command updates base gain only
            # ------------------------------------------------

            gain_command_discarded = False
            gain_command_discard_reason = "none"

            command = poll_latest_gain_command(command_consumer)

            if (
                command is not None
                and APPLY_SERVER_GAIN_COMMAND
                and schedule_fallback_used
            ):
                valid, reason = is_valid_gain_command(
                    command=command,
                    current_run_id=RUN_ID,
                    current_device_id=DEVICE_ID,
                    current_target=target,
                    latest_seq=latest_applied_command_seq,
                )

                if valid:
                    server_kp = float(command["kp"])
                    server_ki = float(command["ki"])
                    server_kd = float(command["kd"])
                    server_target = float(command["target"])

                    # ------------------------------------------------
                    # ESP32 실물 보호용 gain guard
                    # ------------------------------------------------

                    if MOTOR_BACKEND == "esp32" and not is_safe_esp32_gain(
                        server_kp,
                        server_ki,
                        server_kd,
                    ):
                        gain_command_discarded = True
                        gain_command_discard_reason = "unsafe_esp32_server_gain"

                        latest_applied_command_seq = max(
                            latest_applied_command_seq,
                            int(command["source_seq"]),
                        )

                        print(
                            f"[DISCARD] unsafe ESP32 server gain: "
                            f"Kp={server_kp:.3f}, Ki={server_ki:.3f}, Kd={server_kd:.3f}"
                        )

                    else:
                        new_gain = (server_kp, server_ki, server_kd)

                        is_duplicate_gain = (
                            abs(server_target - float(last_applied_server_target)) < 1e-9
                            and all(
                                abs(a - b) < 1e-9
                                for a, b in zip(new_gain, last_applied_server_gain)
                            )
                        )

                        if is_duplicate_gain:
                            gain_command_discarded = True
                            gain_command_discard_reason = "duplicate_gain_command"

                            latest_applied_command_seq = max(
                                latest_applied_command_seq,
                                int(command["source_seq"]),
                            )

                        else:
                            old_gain = last_applied_server_gain
                            base_kp, base_ki, base_kd = new_gain
                            last_applied_server_target = server_target
                            last_applied_server_gain = new_gain

                            latest_applied_command_seq = int(command["source_seq"])
                            gain_update_reason = "server_gain_applied"

                            kp_applied, ki_applied, kd_applied = safety_layer.apply_scale(
                                base_kp,
                                base_ki,
                                base_kd,
                            )
                            set_pid_gains_preserving_integral_term(
                                pid,
                                kp_applied,
                                ki_applied,
                                kd_applied,
                                preserve_integral_term=any(
                                    abs(a - b) > 1e-9
                                    for a, b in zip(new_gain, old_gain)
                                ),
                            )

                            print(
                                f"[APPLY] t={t:.2f}, seq={latest_seq}, "
                                f"source_seq={command['source_seq']}, "
                                f"target={target:.1f}, "
                                f"base=({base_kp:.3f}, {base_ki:.3f}, {base_kd:.3f}), "
                                f"applied=({kp_applied:.3f}, {ki_applied:.3f}, {kd_applied:.3f}), "
                                f"scale=({safety_layer.kp_scale:.3f}, {safety_layer.ki_scale:.3f})"
                            )

                else:
                    gain_command_discarded = True
                    gain_command_discard_reason = reason

            # ------------------------------------------------
            # Motor update
            # ------------------------------------------------

            motor_state = motor.step(pwm)
            measured_current = motor_state.current
            disturbance = motor_state.disturbance

            # step 이후 측정값 기준 error
            measured_error = target - measured_current

            # 다음 loop에서 사용할 current 갱신
            current = measured_current

            # ------------------------------------------------
            # Current PID / safety state
            # ------------------------------------------------

            pid_state = pid.get_state()
            safety_state = safety_layer.get_state()

            # ------------------------------------------------
            # Publish motor state
            # Kafka에는 step 이후 측정값 기준 current/error를 보낸다.
            # ------------------------------------------------

            if step % STATE_PUBLISH_EVERY_STEPS == 0:
                state_message = make_motor_state_message(
                    run_id=RUN_ID,
                    device_id=DEVICE_ID,
                    seq=step,
                    mode=f"local_kafka_controller_{MOTOR_BACKEND}",
                    target=target,
                    current=measured_current,
                    error=measured_error,
                    error_derivative=error_derivative,
                    pwm=pwm,
                    kp=pid_state["kp"],
                    ki=pid_state["ki"],
                    kd=pid_state["kd"],
                    kp_scale=safety_state["kp_scale"],
                    ki_scale=safety_state["ki_scale"],
                    saturation_active=safety_state["saturation_active"],
                    control_time=t,
                    measured_time=t + CONTROL_DT,
                    dt=CONTROL_DT,
                    prev_pwm=prev_pwm,
                    integral=pid_state["integral"],
                    backend=MOTOR_BACKEND,
                    encoder=(
                        motor_state.raw.get("encoder")
                        if isinstance(motor_state.raw, dict)
                        and "encoder" in motor_state.raw
                        else None
                    ),
                )

                producer.send(TOPIC_MOTOR_STATE, value=state_message)

                latest_seq = step

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            rows.append(
                {
                    "run_id": RUN_ID,
                    "device_id": DEVICE_ID,
                    "backend": MOTOR_BACKEND,
                    "step": step,
                    "time": t,

                    "target": target,

                    # control 기준
                    "control_current": control_current,
                    "control_error": control_error,

                    # measurement 기준
                    "current": measured_current,
                    "measured_error": measured_error,

                    # backward compatibility: error는 measured_error로 저장
                    "error": measured_error,

                    "error_derivative": error_derivative,

                    "raw_pwm": raw_pwm,
                    "pwm": pwm,
                    "prev_pwm": prev_pwm,

                    "disturbance": disturbance,

                    "base_kp": base_kp,
                    "base_ki": base_ki,
                    "base_kd": base_kd,

                    "kp": pid_state["kp"],
                    "ki": pid_state["ki"],
                    "kd": pid_state["kd"],
                    "integral": pid_state["integral"],

                    "kp_scale": safety_state["kp_scale"],
                    "ki_scale": safety_state["ki_scale"],
                    "kd_scale": safety_state["kd_scale"],
                    "saturation_counter": safety_state["saturation_counter"],
                    "saturation_active": safety_state["saturation_active"],
                    "local_update_reason": local_update_reason,

                    "gain_update_reason": gain_update_reason,
                    "gain_command_discarded": gain_command_discarded,
                    "gain_command_discard_reason": gain_command_discard_reason,
                    "latest_applied_command_seq": latest_applied_command_seq,

                    "schedule_source": schedule_source,
                    "schedule_apply_mode": SCHEDULE_APPLY_MODE,
                    "schedule_chunk_id": schedule_chunk_id,
                    "schedule_chunk_index": schedule_chunk_index,
                    "schedule_discarded_steps": schedule_discarded_steps,
                    "schedule_chunk_age_sec": schedule_chunk_age_sec,
                    "schedule_source_to_accept_sec": schedule_source_to_accept_sec,
                    "schedule_source_to_apply_sec": schedule_source_to_apply_sec,
                    "schedule_generated_to_accept_sec": schedule_generated_to_accept_sec,
                    "schedule_generated_to_apply_sec": schedule_generated_to_apply_sec,
                    "schedule_publish_to_accept_sec": schedule_publish_to_accept_sec,
                    "schedule_generator_duration_sec": schedule_generator_duration_sec,
                    "schedule_state_to_generation_sec": schedule_state_to_generation_sec,
                    "schedule_latency_assumption_sec": schedule_latency_assumption_sec,
                    "schedule_timing_slack_sec": schedule_timing_slack_sec,
                    "schedule_accepted_control_lag_sec": schedule_accepted_control_lag_sec,
                    "schedule_model_family": schedule_model_family,
                    "schedule_generator_mode": schedule_generator_mode,
                    "schedule_fallback_used": schedule_fallback_used,
                    "schedule_chunk_accept_count": schedule_chunk_accept_count,
                    "schedule_chunk_accept_reason": schedule_chunk_accept_reason,
                    "schedule_buffer_size": len(schedule_buffer),
                }
            )

            if step % 10 == 0:
                print(
                    f"step={step:04d}, "
                    f"t={t:.2f}, "
                    f"target={target:.1f}, "
                    f"control={control_current:.2f}, "
                    f"measured={measured_current:.2f}, "
                    f"err={measured_error:.2f}, "
                    f"raw_pwm={raw_pwm:.2f}, "
                    f"pwm={pwm:.2f}, "
                    f"base=({base_kp:.3f}, {base_ki:.3f}), "
                    f"applied=({pid_state['kp']:.3f}, {pid_state['ki']:.3f}), "
                    f"scale=({safety_state['kp_scale']:.3f}, {safety_state['ki_scale']:.3f}), "
                    f"gain_reason={gain_update_reason}, "
                    f"schedule={schedule_source}, "
                    f"idx={schedule_chunk_index}, "
                    f"discard={gain_command_discard_reason}, "
                    f"local_reason={local_update_reason}"
                )

            # 다음 step derivative 기준은 step 이후 measured_error
            prev_error = measured_error
            prev_pwm = pwm

            # ------------------------------------------------
            # Maintain loop time
            # ------------------------------------------------

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, CONTROL_DT - elapsed)
            time.sleep(sleep_time)

    finally:
        motor.close()
        producer.flush()
        producer.close()
        command_consumer.close()
        schedule_consumer.close()

    result_df = pd.DataFrame(rows)
    metrics = calculate_metrics(result_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    label_parts = [MOTOR_BACKEND, SCHEDULE_APPLY_MODE]
    if run_label:
        safe_label = "".join(
            ch if ch.isalnum() or ch in ["-", "_"] else "_"
            for ch in str(run_label)
        )
        label_parts.append(safe_label)

    file_label = "_".join(label_parts)

    log_path = RESULT_DIR / f"local_kafka_controller_log_{file_label}_{timestamp}.csv"
    metric_path = RESULT_DIR / f"local_kafka_controller_metrics_{file_label}_{timestamp}.csv"

    result_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metric_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Local Kafka Controller Finished")
    print(f"Saved log: {log_path}")
    print(f"Saved metrics: {metric_path}")
    print("=" * 80)
    print(metrics)


if __name__ == "__main__":
    cli_args = parse_args()
    apply_runtime_args(cli_args)
    run_local_kafka_controller(run_label=cli_args.run_label)
