import sys
from pathlib import Path
import time
import signal
import threading
import argparse
import json
from collections import defaultdict, deque

try:
    from kafka import KafkaConsumer, KafkaProducer
except ImportError:
    KafkaConsumer = None
    KafkaProducer = None


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import (
    PID_GAIN_DB,
    GAIN_DB_MODE,
    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,

    ESP32_REAL_PID_GAIN_DB,
    ESP32_REAL_GAIN_DB_MODE,
    ESP32_SWEEP_KP_LIST,
    ESP32_SWEEP_KI_LIST,
    ESP32_SWEEP_KD_LIST,
    MODEL_DIR,
    RESULTS_DIR,
)

from kafka_config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TOPIC_MOTOR_STATE,
    TOPIC_GAIN_COMMAND,
    TOPIC_SCHEDULE_CHUNK,
    GAIN_RECOMMENDER_GROUP_ID,
    GAIN_COMMAND_TTL_SEC,
    SCHEDULE_CHUNK_TTL_SEC,
    SCHEDULE_CHUNK_DT,
    SCHEDULE_CHUNK_HORIZON_STEPS,
)

from message_schema import (
    json_serializer,
    json_deserializer,
    make_gain_command_message,
)

from schedule_generators import (
    DirectGainChunkPolicyGenerator,
    DirectPolicyMlpChunkGenerator,
    DirectPolicySequenceMlpChunkGenerator,
    DiffusionUnetGainChunkGenerator,
    DbGainChunkGenerator,
    MlpCostChunkGenerator,
    MultiTaskMlpCostChunkGenerator,
    RandomForestCostChunkGenerator,
)


# ============================================================
# Server settings
# ============================================================

# 서버 AI 추론 + Kafka 왕복 지연을 포함한 schedule lead time.
# RF 실측 p90 latency 기준 기본값이며, DL/다른 모델은 CLI에서 덮어쓴다.
INFERENCE_DELAY_SEC = 0.5
ARTIFICIAL_INFERENCE_SLEEP = True

# 너무 많은 state에 대해 매번 command를 보내지 않기 위한 최소 발행 간격
COMMAND_MIN_INTERVAL_SEC = 0.5

# 동일 target에서 너무 작은 변화는 command 반복 발행 방지
TARGET_TOL = 1e-9

# 같은 device라도 backend가 바뀌면 별도로 command frequency를 관리하기 위한 key
USE_MODE_IN_RATE_LIMIT_KEY = True

GENERATOR_MODE = "db"  # "db" or "rf_cost"
HISTORY_MAXLEN = 50
RF_MODEL_PATH = MODEL_DIR / "esp32_horizon_cost_random_forest_latest.joblib"
MLP_MODEL_PATH = MODEL_DIR / "esp32_horizon_cost_mlp_latest.joblib"
MULTITASK_MLP_MODEL_PATH = MODEL_DIR / "esp32_horizon_multitask_mlp_latest.joblib"
DIRECT_POLICY_MLP_MODEL_PATH = MODEL_DIR / "esp32_direct_policy_mlp_latest.joblib"
GAIN_CHUNK_POLICY_MODEL_PATH = (
    MODEL_DIR / "diffusion_gain_chunk_mlp_balanced1000_global_tracking_topk_20260508_185112.joblib"
)
DIFFUSION_UNET_MODEL_PATH = (
    MODEL_DIR / "diffusion_gain_chunk_unet_balanced1000_global_topk_full_20260508_193250.joblib"
)
GAIN_CHUNK_COST_SURROGATE_MODEL_PATH = ""
GAIN_CHUNK_COST_SELECTION_METRIC = "label_cost"
RESPONSE_SURROGATE_MODEL_PATH = ""
RESPONSE_SCORE_MODE = "iae"
DIFFUSION_CANDIDATE_MODE = "sample"
DIFFUSION_DETERMINISTIC_SEED = 0
TWO_PHASE_BOOST_SCALE = 1.0
TWO_PHASE_KI_DECAY_SCALE = 1.0
TWO_PHASE_KD_BOOST_SCALE = 1.0
DIFFUSION_DDIM_STEPS = 20
DIFFUSION_SAMPLE_COUNT = 1
SCHEDULE_CHUNK_MESSAGE_LOG_PATH = None


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run gain/schedule recommender server."
    )

    parser.add_argument(
        "--inference-delay",
        type=float,
        default=INFERENCE_DELAY_SEC,
        help="Simulated server inference delay in seconds.",
    )
    parser.add_argument(
        "--disable-artificial-inference-sleep",
        action="store_true",
        help=(
            "Do not sleep before generation. The inference-delay value is still "
            "used as schedule lead time, but actual model runtime is measured "
            "instead of simulated."
        ),
    )
    parser.add_argument(
        "--command-min-interval",
        type=float,
        default=COMMAND_MIN_INTERVAL_SEC,
        help="Minimum publish interval for repeated recommendations.",
    )
    parser.add_argument(
        "--chunk-horizon-steps",
        type=int,
        default=SCHEDULE_CHUNK_HORIZON_STEPS,
        help="Number of steps in each generated schedule chunk.",
    )
    parser.add_argument(
        "--chunk-ttl",
        type=float,
        default=SCHEDULE_CHUNK_TTL_SEC,
        help="Wall-clock validity duration for schedule chunks.",
    )
    parser.add_argument(
        "--generator",
        choices=[
            "db",
            "rf_cost",
            "mlp_cost",
            "multitask_mlp_cost",
            "direct_policy_mlp",
            "direct_policy_sequence_mlp",
            "direct_gain_chunk_policy",
            "diffusion_unet_gain_chunk",
        ],
        default=GENERATOR_MODE,
        help="Schedule generator backend.",
    )
    parser.add_argument(
        "--rf-model-path",
        type=str,
        default=str(RF_MODEL_PATH),
        help="Path to trained ESP32 horizon-cost RandomForest model.",
    )
    parser.add_argument(
        "--mlp-model-path",
        type=str,
        default=str(MLP_MODEL_PATH),
        help="Path to trained ESP32 horizon-cost TensorFlow MLP metadata.",
    )
    parser.add_argument(
        "--multitask-mlp-model-path",
        type=str,
        default=str(MULTITASK_MLP_MODEL_PATH),
        help="Path to trained ESP32 multi-task TensorFlow MLP metadata.",
    )
    parser.add_argument(
        "--direct-policy-mlp-model-path",
        type=str,
        default=str(DIRECT_POLICY_MLP_MODEL_PATH),
        help="Path to trained ESP32 direct-policy TensorFlow MLP metadata.",
    )
    parser.add_argument(
        "--gain-chunk-policy-model-path",
        type=str,
        default=str(GAIN_CHUNK_POLICY_MODEL_PATH),
        help="Path to trained supervised gain-chunk TensorFlow metadata.",
    )
    parser.add_argument(
        "--diffusion-unet-model-path",
        type=str,
        default=str(DIFFUSION_UNET_MODEL_PATH),
        help="Path to trained diffusion U-Net gain-chunk metadata.",
    )
    parser.add_argument(
        "--diffusion-ddim-steps",
        type=int,
        default=DIFFUSION_DDIM_STEPS,
        help="Number of DDIM denoising steps for diffusion gain chunks.",
    )
    parser.add_argument(
        "--diffusion-sample-count",
        type=int,
        default=DIFFUSION_SAMPLE_COUNT,
        help="Number of generated diffusion samples averaged into one chunk.",
    )
    parser.add_argument(
        "--gain-chunk-cost-surrogate-model-path",
        type=str,
        default=str(GAIN_CHUNK_COST_SURROGATE_MODEL_PATH),
        help="Optional gain-chunk cost surrogate metadata for diffusion candidate selection.",
    )
    parser.add_argument(
        "--gain-chunk-cost-selection-metric",
        type=str,
        default=GAIN_CHUNK_COST_SELECTION_METRIC,
        help="Cost surrogate output column used for diffusion candidate selection.",
    )
    parser.add_argument(
        "--response-surrogate-model-path",
        type=str,
        default=str(RESPONSE_SURROGATE_MODEL_PATH),
        help=(
            "Optional closed-loop response surrogate metadata. When supplied "
            "with diffusion-sample-count > 1, candidates are selected by "
            "predicted RPM/PWM response score."
        ),
    )
    parser.add_argument(
        "--response-score-mode",
        type=str,
        default=RESPONSE_SCORE_MODE,
        choices=["iae", "settling_overshoot"],
        help="Response surrogate objective used for diffusion candidate selection.",
    )
    parser.add_argument(
        "--diffusion-candidate-mode",
        type=str,
        default=DIFFUSION_CANDIDATE_MODE,
        choices=["sample", "two_phase"],
        help=(
            "sample keeps the original diffusion candidates. two_phase creates "
            "deterministic approach/stabilization candidates from one base chunk."
        ),
    )
    parser.add_argument(
        "--diffusion-deterministic-seed",
        type=int,
        default=DIFFUSION_DETERMINISTIC_SEED,
        help="Set >0 to reseed diffusion noise before every chunk generation.",
    )
    parser.add_argument(
        "--two-phase-boost-scale",
        type=float,
        default=TWO_PHASE_BOOST_SCALE,
        help="Scale for early Kp/Ki boost in two_phase diffusion candidate mode.",
    )
    parser.add_argument(
        "--two-phase-ki-decay-scale",
        type=float,
        default=TWO_PHASE_KI_DECAY_SCALE,
        help="Scale for late Ki decay in two_phase diffusion candidate mode.",
    )
    parser.add_argument(
        "--two-phase-kd-boost-scale",
        type=float,
        default=TWO_PHASE_KD_BOOST_SCALE,
        help="Scale for late Kd boost in two_phase diffusion candidate mode.",
    )
    parser.add_argument(
        "--history-maxlen",
        type=int,
        default=HISTORY_MAXLEN,
        help="Number of recent telemetry states kept per device/backend.",
    )

    return parser.parse_args()


def apply_runtime_args(args):
    global INFERENCE_DELAY_SEC
    global ARTIFICIAL_INFERENCE_SLEEP
    global COMMAND_MIN_INTERVAL_SEC
    global SCHEDULE_CHUNK_HORIZON_STEPS
    global SCHEDULE_CHUNK_TTL_SEC
    global GENERATOR_MODE
    global RF_MODEL_PATH
    global MLP_MODEL_PATH
    global MULTITASK_MLP_MODEL_PATH
    global DIRECT_POLICY_MLP_MODEL_PATH
    global GAIN_CHUNK_POLICY_MODEL_PATH
    global DIFFUSION_UNET_MODEL_PATH
    global GAIN_CHUNK_COST_SURROGATE_MODEL_PATH
    global GAIN_CHUNK_COST_SELECTION_METRIC
    global RESPONSE_SURROGATE_MODEL_PATH
    global RESPONSE_SCORE_MODE
    global DIFFUSION_CANDIDATE_MODE
    global DIFFUSION_DETERMINISTIC_SEED
    global TWO_PHASE_BOOST_SCALE
    global TWO_PHASE_KI_DECAY_SCALE
    global TWO_PHASE_KD_BOOST_SCALE
    global DIFFUSION_DDIM_STEPS
    global DIFFUSION_SAMPLE_COUNT
    global SCHEDULE_CHUNK_MESSAGE_LOG_PATH
    global HISTORY_MAXLEN

    INFERENCE_DELAY_SEC = float(args.inference_delay)
    ARTIFICIAL_INFERENCE_SLEEP = not bool(args.disable_artificial_inference_sleep)
    COMMAND_MIN_INTERVAL_SEC = float(args.command_min_interval)
    SCHEDULE_CHUNK_HORIZON_STEPS = int(args.chunk_horizon_steps)
    SCHEDULE_CHUNK_TTL_SEC = float(args.chunk_ttl)
    GENERATOR_MODE = str(args.generator)
    RF_MODEL_PATH = args.rf_model_path
    MLP_MODEL_PATH = args.mlp_model_path
    MULTITASK_MLP_MODEL_PATH = args.multitask_mlp_model_path
    DIRECT_POLICY_MLP_MODEL_PATH = args.direct_policy_mlp_model_path
    GAIN_CHUNK_POLICY_MODEL_PATH = args.gain_chunk_policy_model_path
    DIFFUSION_UNET_MODEL_PATH = args.diffusion_unet_model_path
    GAIN_CHUNK_COST_SURROGATE_MODEL_PATH = args.gain_chunk_cost_surrogate_model_path
    GAIN_CHUNK_COST_SELECTION_METRIC = args.gain_chunk_cost_selection_metric
    RESPONSE_SURROGATE_MODEL_PATH = args.response_surrogate_model_path
    RESPONSE_SCORE_MODE = args.response_score_mode
    DIFFUSION_CANDIDATE_MODE = args.diffusion_candidate_mode
    DIFFUSION_DETERMINISTIC_SEED = int(args.diffusion_deterministic_seed)
    TWO_PHASE_BOOST_SCALE = float(args.two_phase_boost_scale)
    TWO_PHASE_KI_DECAY_SCALE = float(args.two_phase_ki_decay_scale)
    TWO_PHASE_KD_BOOST_SCALE = float(args.two_phase_kd_boost_scale)
    DIFFUSION_DDIM_STEPS = int(args.diffusion_ddim_steps)
    DIFFUSION_SAMPLE_COUNT = int(args.diffusion_sample_count)
    HISTORY_MAXLEN = int(args.history_maxlen)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    SCHEDULE_CHUNK_MESSAGE_LOG_PATH = (
        RESULTS_DIR
        / "kafka_control"
        / f"gain_recommender_schedule_chunks_{GENERATOR_MODE}_{timestamp}.jsonl"
    )


# ============================================================
# Global stop flag
# ============================================================

stop_event = threading.Event()


def signal_handler(sig, frame):
    print("\nStop signal received. Shutting down gain recommender server...")
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_model_family(generator_mode: str) -> str:
    if generator_mode == "rf_cost":
        return "random_forest"
    if generator_mode == "mlp_cost":
        return "tensorflow_mlp"
    if generator_mode == "multitask_mlp_cost":
        return "tensorflow_multitask_mlp"
    if generator_mode in {"direct_policy_mlp", "direct_policy_sequence_mlp"}:
        return "tensorflow_direct_policy_mlp"
    if generator_mode == "direct_gain_chunk_policy":
        return "tensorflow_direct_gain_chunk_policy"
    if generator_mode == "diffusion_unet_gain_chunk":
        return "tensorflow_diffusion_unet_gain_chunk"
    return "gain_db"


# ============================================================
# Backend / gain DB utility
# ============================================================

def infer_backend_from_state(state: dict) -> str:
    """
    motor_state message에서 backend를 추정한다.

    현재 local_kafka_controller.py는 mode를 다음처럼 보낸다.
        local_kafka_controller_simulation
        local_kafka_controller_esp32

    향후 message에 backend 필드가 추가되면 그것을 우선 사용한다.
    """

    backend = state.get("backend", None)

    if backend is not None:
        backend = str(backend).lower().strip()
        if backend in ["simulation", "esp32"]:
            return backend

    mode = str(state.get("mode", "")).lower()

    if "esp32" in mode:
        return "esp32"

    if "simulation" in mode or "sim" in mode:
        return "simulation"

    # fallback
    return "simulation"


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

    # Exact match
    if target in db_targets:
        gains = gain_db[target]
        return gains["kp"], gains["ki"], gains["kd"]

    # Nearest mode
    if mode == "nearest":
        nearest_target = min(db_targets, key=lambda x: abs(x - target))
        gains = gain_db[nearest_target]
        return gains["kp"], gains["ki"], gains["kd"]

    # Linear interpolation mode
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

    # Fallback
    nearest_target = min(db_targets, key=lambda x: abs(x - target))
    gains = gain_db[nearest_target]
    return gains["kp"], gains["ki"], gains["kd"]


def get_gain_from_db(target: float, backend: str):
    """
    Backend별 gain DB에서 target 기반 gain을 가져온다.

    - simulation: PID_GAIN_DB 사용
    - esp32: ESP32_REAL_PID_GAIN_DB 사용
    """

    backend = str(backend).lower().strip()

    if backend == "esp32":
        kp, ki, kd = interpolate_gain_from_db(
            target=target,
            gain_db=ESP32_REAL_PID_GAIN_DB,
            mode=ESP32_REAL_GAIN_DB_MODE,
            fallback_gain=(1.2, 0.7, 0.0),
        )
        db_name = "ESP32_REAL_PID_GAIN_DB"
        return kp, ki, kd, db_name

    kp, ki, kd = interpolate_gain_from_db(
        target=target,
        gain_db=PID_GAIN_DB,
        mode=GAIN_DB_MODE,
        fallback_gain=(DEFAULT_KP, DEFAULT_KI, DEFAULT_KD),
    )
    db_name = "PID_GAIN_DB"
    return kp, ki, kd, db_name


def build_candidate_gains(backend: str):
    backend = str(backend).lower().strip()

    if backend == "esp32":
        candidates = set()

        for kp in ESP32_SWEEP_KP_LIST:
            for ki in ESP32_SWEEP_KI_LIST:
                for kd in ESP32_SWEEP_KD_LIST:
                    candidates.add((float(kp), float(ki), float(kd)))

        for gains in ESP32_REAL_PID_GAIN_DB.values():
            candidates.add(
                (
                    float(gains["kp"]),
                    float(gains["ki"]),
                    float(gains["kd"]),
                )
            )

        return sorted(candidates)

    candidates = set()
    for gains in PID_GAIN_DB.values():
        candidates.add(
            (
                float(gains["kp"]),
                float(gains["ki"]),
                float(gains["kd"]),
            )
        )

    return sorted(candidates)


def create_schedule_generator(backend: str, generator_cache: dict = None):
    backend = str(backend).lower().strip()
    if GENERATOR_MODE == "mlp_cost":
        model_path_key = MLP_MODEL_PATH
    elif GENERATOR_MODE == "multitask_mlp_cost":
        model_path_key = MULTITASK_MLP_MODEL_PATH
    elif GENERATOR_MODE in {"direct_policy_mlp", "direct_policy_sequence_mlp"}:
        model_path_key = DIRECT_POLICY_MLP_MODEL_PATH
    elif GENERATOR_MODE == "direct_gain_chunk_policy":
        model_path_key = GAIN_CHUNK_POLICY_MODEL_PATH
    elif GENERATOR_MODE == "diffusion_unet_gain_chunk":
        model_path_key = (
            f"{DIFFUSION_UNET_MODEL_PATH}:"
            f"ddim{DIFFUSION_DDIM_STEPS}:n{DIFFUSION_SAMPLE_COUNT}:"
            f"cost{GAIN_CHUNK_COST_SURROGATE_MODEL_PATH}:"
            f"metric{GAIN_CHUNK_COST_SELECTION_METRIC}:"
            f"response{RESPONSE_SURROGATE_MODEL_PATH}:"
            f"score{RESPONSE_SCORE_MODE}:"
            f"candidate{DIFFUSION_CANDIDATE_MODE}:"
            f"seed{DIFFUSION_DETERMINISTIC_SEED}:"
            f"twophase{TWO_PHASE_BOOST_SCALE}-{TWO_PHASE_KI_DECAY_SCALE}-{TWO_PHASE_KD_BOOST_SCALE}"
        )
    else:
        model_path_key = RF_MODEL_PATH
    cache_key = f"{backend}:{GENERATOR_MODE}:{model_path_key}"

    if generator_cache is not None and cache_key in generator_cache:
        return generator_cache[cache_key]

    if backend == "esp32":
        db_generator = DbGainChunkGenerator(
            gain_db=ESP32_REAL_PID_GAIN_DB,
            mode=ESP32_REAL_GAIN_DB_MODE,
            fallback_gain=(1.2, 0.7, 0.0),
            backend_name="esp32",
        )

        if GENERATOR_MODE == "rf_cost":
            generator = RandomForestCostChunkGenerator(
                model_path=RF_MODEL_PATH,
                candidate_gains=build_candidate_gains(backend),
                backend_name="esp32",
                fallback_generator=db_generator,
            )
            if generator_cache is not None:
                generator_cache[cache_key] = generator
            return generator

        if GENERATOR_MODE == "mlp_cost":
            generator = MlpCostChunkGenerator(
                model_path=MLP_MODEL_PATH,
                candidate_gains=build_candidate_gains(backend),
                backend_name="esp32",
                fallback_generator=db_generator,
            )
            if generator_cache is not None:
                generator_cache[cache_key] = generator
            return generator

        if GENERATOR_MODE == "multitask_mlp_cost":
            generator = MultiTaskMlpCostChunkGenerator(
                model_path=MULTITASK_MLP_MODEL_PATH,
                candidate_gains=build_candidate_gains(backend),
                backend_name="esp32",
                fallback_generator=db_generator,
            )
            if generator_cache is not None:
                generator_cache[cache_key] = generator
            return generator

        if GENERATOR_MODE == "direct_policy_mlp":
            generator = DirectPolicyMlpChunkGenerator(
                model_path=DIRECT_POLICY_MLP_MODEL_PATH,
                backend_name="esp32",
                fallback_generator=db_generator,
            )
            if generator_cache is not None:
                generator_cache[cache_key] = generator
            return generator

        if GENERATOR_MODE == "direct_policy_sequence_mlp":
            generator = DirectPolicySequenceMlpChunkGenerator(
                model_path=DIRECT_POLICY_MLP_MODEL_PATH,
                backend_name="esp32",
                fallback_generator=db_generator,
            )
            if generator_cache is not None:
                generator_cache[cache_key] = generator
            return generator

        if GENERATOR_MODE == "direct_gain_chunk_policy":
            generator = DirectGainChunkPolicyGenerator(
                model_path=GAIN_CHUNK_POLICY_MODEL_PATH,
                backend_name="esp32",
                fallback_generator=db_generator,
            )
            if generator_cache is not None:
                generator_cache[cache_key] = generator
            return generator

        if GENERATOR_MODE == "diffusion_unet_gain_chunk":
            generator = DiffusionUnetGainChunkGenerator(
                model_path=DIFFUSION_UNET_MODEL_PATH,
                backend_name="esp32",
                fallback_generator=db_generator,
                ddim_steps=DIFFUSION_DDIM_STEPS,
                sample_count=DIFFUSION_SAMPLE_COUNT,
                cost_surrogate_model_path=GAIN_CHUNK_COST_SURROGATE_MODEL_PATH,
                cost_selection_metric=GAIN_CHUNK_COST_SELECTION_METRIC,
                response_surrogate_model_path=RESPONSE_SURROGATE_MODEL_PATH,
                response_score_mode=RESPONSE_SCORE_MODE,
                diffusion_candidate_mode=DIFFUSION_CANDIDATE_MODE,
                diffusion_deterministic_seed=DIFFUSION_DETERMINISTIC_SEED,
                two_phase_boost_scale=TWO_PHASE_BOOST_SCALE,
                two_phase_ki_decay_scale=TWO_PHASE_KI_DECAY_SCALE,
                two_phase_kd_boost_scale=TWO_PHASE_KD_BOOST_SCALE,
            )
            if generator_cache is not None:
                generator_cache[cache_key] = generator
            return generator

        generator = db_generator
        if generator_cache is not None:
            generator_cache[cache_key] = generator
        return generator

    db_generator = DbGainChunkGenerator(
        gain_db=PID_GAIN_DB,
        mode=GAIN_DB_MODE,
        fallback_gain=(DEFAULT_KP, DEFAULT_KI, DEFAULT_KD),
        backend_name="simulation",
    )

    if generator_cache is not None:
        generator_cache[cache_key] = db_generator

    return db_generator


def make_rate_limit_key(device_id: str, backend: str):
    if USE_MODE_IN_RATE_LIMIT_KEY:
        return f"{device_id}:{backend}"
    return device_id


def append_state_history(history_by_key, key: str, state: dict):
    history_by_key[key].append(dict(state))
    state["_history"] = list(history_by_key[key])
    return state


def append_schedule_chunk_message_log(schedule_chunk: dict):
    if SCHEDULE_CHUNK_MESSAGE_LOG_PATH is None:
        return
    try:
        SCHEDULE_CHUNK_MESSAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SCHEDULE_CHUNK_MESSAGE_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(schedule_chunk, ensure_ascii=False, default=str))
            handle.write("\n")
    except Exception as exc:
        print(f"[SCHEDULE_LOG_ERROR] error={exc}")


# ============================================================
# Kafka
# ============================================================

def create_consumer():
    if KafkaConsumer is None:
        raise ImportError(
            "kafka-python is required to run the recommender server. "
            "Install it in the active Python environment."
        )

    consumer = KafkaConsumer(
        TOPIC_MOTOR_STATE,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=GAIN_RECOMMENDER_GROUP_ID,
        value_deserializer=json_deserializer,
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    return consumer


def create_producer():
    if KafkaProducer is None:
        raise ImportError(
            "kafka-python is required to run the recommender server. "
            "Install it in the active Python environment."
        )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=json_serializer,
    )

    return producer


# ============================================================
# Server loop
# ============================================================

def run_gain_recommender_server():
    print("=" * 80)
    print("Gain Recommender Server Start")
    print("=" * 80)
    print(f"Kafka bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Consume topic: {TOPIC_MOTOR_STATE}")
    print(f"Produce topic: {TOPIC_GAIN_COMMAND}")
    print(f"Produce schedule topic: {TOPIC_SCHEDULE_CHUNK}")
    print(f"Inference delay: {INFERENCE_DELAY_SEC} s")
    print(f"Artificial inference sleep: {ARTIFICIAL_INFERENCE_SLEEP}")
    print(f"Command min interval: {COMMAND_MIN_INTERVAL_SEC} s")
    print(f"Schedule horizon steps: {SCHEDULE_CHUNK_HORIZON_STEPS}")
    print(f"Generator mode: {GENERATOR_MODE}")
    if GENERATOR_MODE == "rf_cost":
        print(f"RF model path: {RF_MODEL_PATH}")
    if GENERATOR_MODE == "mlp_cost":
        print(f"MLP model path: {MLP_MODEL_PATH}")
    if GENERATOR_MODE == "multitask_mlp_cost":
        print(f"Multi-task MLP model path: {MULTITASK_MLP_MODEL_PATH}")
    if GENERATOR_MODE == "direct_gain_chunk_policy":
        print(f"Gain chunk policy model path: {GAIN_CHUNK_POLICY_MODEL_PATH}")
    if GENERATOR_MODE == "diffusion_unet_gain_chunk":
        print(f"Diffusion U-Net model path: {DIFFUSION_UNET_MODEL_PATH}")
        print(f"Diffusion DDIM steps: {DIFFUSION_DDIM_STEPS}")
        print(f"Diffusion sample count: {DIFFUSION_SAMPLE_COUNT}")
        print(f"Gain chunk cost surrogate model path: {GAIN_CHUNK_COST_SURROGATE_MODEL_PATH}")
        print(f"Gain chunk cost selection metric: {GAIN_CHUNK_COST_SELECTION_METRIC}")
        print(f"Response surrogate model path: {RESPONSE_SURROGATE_MODEL_PATH}")
        print(f"Response score mode: {RESPONSE_SCORE_MODE}")
        print(f"Diffusion candidate mode: {DIFFUSION_CANDIDATE_MODE}")
        print(f"Diffusion deterministic seed: {DIFFUSION_DETERMINISTIC_SEED}")
        print(
            "Two-phase scales: "
            f"boost={TWO_PHASE_BOOST_SCALE}, "
            f"ki_decay={TWO_PHASE_KI_DECAY_SCALE}, "
            f"kd_boost={TWO_PHASE_KD_BOOST_SCALE}"
        )
    print(f"Schedule chunk message log: {SCHEDULE_CHUNK_MESSAGE_LOG_PATH}")
    print(f"History maxlen: {HISTORY_MAXLEN}")
    print("=" * 80)

    consumer = create_consumer()
    producer = create_producer()

    last_command_time_by_key = {}
    last_target_by_key = {}
    history_by_key = defaultdict(lambda: deque(maxlen=HISTORY_MAXLEN))
    generator_cache = {}

    try:
        while not stop_event.is_set():
            records = consumer.poll(timeout_ms=100)

            if not records:
                continue

            for _, messages in records.items():
                for msg in messages:
                    state = msg.value

                    try:
                        run_id = state["run_id"]
                        device_id = state["device_id"]
                        seq = int(state["seq"])
                        target = float(state["target"])
                        mode = str(state.get("mode", "unknown"))

                        backend = infer_backend_from_state(state)
                        rate_key = make_rate_limit_key(device_id, backend)
                        state = append_state_history(history_by_key, rate_key, state)

                        now = time.time()

                        # ----------------------------------------------------
                        # command 발행 빈도 제한
                        # ----------------------------------------------------

                        last_time = last_command_time_by_key.get(rate_key, 0.0)
                        last_target = last_target_by_key.get(rate_key, None)

                        same_target = (
                            last_target is not None
                            and abs(float(last_target) - target) <= TARGET_TOL
                        )

                        # 같은 backend + 같은 target에 대해 너무 자주 command 발행하지 않음
                        if same_target and (now - last_time) < COMMAND_MIN_INTERVAL_SEC:
                            continue

                        # ----------------------------------------------------
                        # inference delay 모사
                        # ----------------------------------------------------

                        if ARTIFICIAL_INFERENCE_SLEEP:
                            time.sleep(INFERENCE_DELAY_SEC)

                        # ----------------------------------------------------
                        # gain recommendation
                        # ----------------------------------------------------

                        kp, ki, kd, db_name = get_gain_from_db(
                            target=target,
                            backend=backend,
                        )

                        command = make_gain_command_message(
                            run_id=run_id,
                            device_id=device_id,
                            source_seq=seq,
                            target=target,
                            kp=kp,
                            ki=ki,
                            kd=kd,
                            confidence=1.0,
                            valid_for_sec=GAIN_COMMAND_TTL_SEC,
                            reason=f"{backend}_{db_name}_recommendation",
                        )

                        command_future = producer.send(TOPIC_GAIN_COMMAND, value=command)

                        # ----------------------------------------------------
                        # baseline schedule chunk
                        # ----------------------------------------------------

                        chunk_dt = float(state.get("dt", SCHEDULE_CHUNK_DT))
                        source_control_time = float(state.get("control_time", 0.0))
                        schedule_start_time = source_control_time + INFERENCE_DELAY_SEC

                        schedule_chunk = None
                        schedule_error = None

                        try:
                            generator_start_wall_time = time.time()
                            generator = create_schedule_generator(
                                backend,
                                generator_cache=generator_cache,
                            )
                            schedule_chunk = generator.generate(
                                state=state,
                                schedule_start_time=schedule_start_time,
                                dt=chunk_dt,
                                horizon_steps=SCHEDULE_CHUNK_HORIZON_STEPS,
                            )
                            generator_end_wall_time = time.time()

                            schedule_chunk.setdefault("metadata", {})
                            schedule_chunk["metadata"].update(
                                {
                                    "generator_mode": GENERATOR_MODE,
                                    "model_family": get_model_family(GENERATOR_MODE),
                                    "latency_assumption_sec": float(
                                        INFERENCE_DELAY_SEC
                                    ),
                                    "server_received_at": float(now),
                                    "generator_start_wall_time": float(
                                        generator_start_wall_time
                                    ),
                                    "generator_end_wall_time": float(
                                        generator_end_wall_time
                                    ),
                                    "generator_duration_sec": float(
                                        generator_end_wall_time
                                        - generator_start_wall_time
                                    ),
                                    "state_to_generation_sec": float(
                                        generator_end_wall_time
                                        - float(state.get("timestamp", now))
                                    ),
                                }
                            )
                            schedule_chunk["valid_until"] = (
                                float(schedule_chunk["generated_at"])
                                + float(SCHEDULE_CHUNK_TTL_SEC)
                            )

                            schedule_publish_start_wall_time = time.time()
                            schedule_chunk["metadata"].update(
                                {
                                    "schedule_publish_start_wall_time": float(
                                        schedule_publish_start_wall_time
                                    ),
                                }
                            )
                            schedule_future = producer.send(
                                TOPIC_SCHEDULE_CHUNK,
                                value=schedule_chunk,
                            )
                            schedule_future.get(timeout=5)
                            append_schedule_chunk_message_log(schedule_chunk)
                            schedule_publish_end_wall_time = time.time()

                            print(
                                f"[SCHEDULE] generator={schedule_chunk['generator_id']}, "
                                f"chunk={schedule_chunk['chunk_id'][:8]}, "
                                f"start={schedule_start_time:.2f}, "
                                f"items={schedule_chunk['horizon_steps']}, "
                                f"reason={schedule_chunk['reason']}, "
                                f"gen_ms={(generator_end_wall_time - generator_start_wall_time) * 1000.0:.1f}"
                            )

                        except Exception as schedule_exc:
                            schedule_error = schedule_exc
                            print(
                                f"[SCHEDULE_ERROR] backend={backend}, "
                                f"device={device_id}, seq={seq}, "
                                f"generator_mode={GENERATOR_MODE}, "
                                f"error={schedule_exc}"
                            )

                            try:
                                fallback_generator = DbGainChunkGenerator(
                                    gain_db=ESP32_REAL_PID_GAIN_DB
                                    if backend == "esp32"
                                    else PID_GAIN_DB,
                                    mode=ESP32_REAL_GAIN_DB_MODE
                                    if backend == "esp32"
                                    else GAIN_DB_MODE,
                                    fallback_gain=(1.2, 0.7, 0.0)
                                    if backend == "esp32"
                                    else (DEFAULT_KP, DEFAULT_KI, DEFAULT_KD),
                                    backend_name=f"{backend}_fallback",
                                )
                                schedule_chunk = fallback_generator.generate(
                                    state=state,
                                    schedule_start_time=schedule_start_time,
                                    dt=chunk_dt,
                                    horizon_steps=SCHEDULE_CHUNK_HORIZON_STEPS,
                                )
                                fallback_end_wall_time = time.time()
                                schedule_chunk["reason"] = "fallback_after_schedule_error"
                                schedule_chunk["metadata"]["schedule_error"] = str(
                                    schedule_exc
                                )
                                schedule_chunk["metadata"].update(
                                    {
                                        "generator_mode": GENERATOR_MODE,
                                        "model_family": "fallback_gain_db",
                                        "latency_assumption_sec": float(
                                            INFERENCE_DELAY_SEC
                                        ),
                                        "server_received_at": float(now),
                                        "generator_duration_sec": float(
                                            fallback_end_wall_time - now
                                        ),
                                        "state_to_generation_sec": float(
                                            fallback_end_wall_time
                                            - float(state.get("timestamp", now))
                                        ),
                                    }
                                )
                                schedule_chunk["valid_until"] = (
                                    float(schedule_chunk["generated_at"])
                                    + float(SCHEDULE_CHUNK_TTL_SEC)
                                )
                                producer.send(
                                    TOPIC_SCHEDULE_CHUNK,
                                    value=schedule_chunk,
                                ).get(timeout=5)
                                append_schedule_chunk_message_log(schedule_chunk)
                                print(
                                    f"[SCHEDULE_FALLBACK] "
                                    f"chunk={schedule_chunk['chunk_id'][:8]}, "
                                    f"reason={schedule_chunk['reason']}"
                                )
                            except Exception as fallback_exc:
                                print(
                                    f"[SCHEDULE_FALLBACK_ERROR] "
                                    f"backend={backend}, device={device_id}, "
                                    f"seq={seq}, error={fallback_exc}"
                                )

                        command_future.get(timeout=5)
                        producer.flush()

                        last_command_time_by_key[rate_key] = time.time()
                        last_target_by_key[rate_key] = target

                        print(
                            f"[COMMAND] backend={backend}, "
                            f"mode={mode}, "
                            f"device={device_id}, "
                            f"seq={seq}, target={target:.1f}, "
                            f"DB={db_name}, "
                            f"Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}, "
                            f"schedule_ok={schedule_chunk is not None and schedule_error is None}"
                        )

                    except Exception as e:
                        print(f"Failed to process motor_state message: {e}")
                        print(f"Message: {state}")

    finally:
        print("Closing Kafka consumer/producer...")
        consumer.close()
        producer.close()
        print("Gain recommender server stopped.")


if __name__ == "__main__":
    cli_args = parse_args()
    apply_runtime_args(cli_args)
    run_gain_recommender_server()
