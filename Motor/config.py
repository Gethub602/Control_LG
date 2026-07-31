from pathlib import Path

# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SIMULATION_DATA_DIR = DATA_DIR / "simulation"

RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
FIGURE_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"

SIMULINK_DIR = ROOT_DIR / "Simulink" / "Motor"


# ============================================================
# Control settings
# ============================================================

DT = 0.01  # sampling time [s]

PWM_MIN = 0
PWM_MAX = 255

TARGET_MIN = 0.0
TARGET_MAX = 3000.0  # 예: RPM 기준이면 나중에 수정


# ============================================================
# Default PID gains
# ============================================================

DEFAULT_KP = 0.5
DEFAULT_KI = 0.0
DEFAULT_KD = 0.0


# ============================================================
# PID gain limits
# ============================================================

KP_MIN = 0.0
KP_MAX = 10.0

KI_MIN = 0.0
KI_MAX = 10.0

KD_MIN = 0.0
KD_MAX = 10.0


# ============================================================
# Gain update limits
# ============================================================

MAX_DELTA_KP = 0.5
MAX_DELTA_KI = 0.5
MAX_DELTA_KD = 0.5


# ============================================================
# Safety settings
# ============================================================

MAX_ERROR = 3000.0
MAX_ERROR_DERIVATIVE = 10000.0
OSCILLATION_THRESHOLD = 5
FALLBACK_ENABLED = True


# ============================================================
# Serial communication
# ============================================================

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 2.0

# ============================================================
# Experiment settings
# ============================================================

RUN_MODE = "fixed_pid"  # "fixed_pid", "adaptive_pid", or "simulink_pid"
ENV_TYPE = "simple_motor"

TEST_TARGET = 100.0
TEST_STEPS = 1000

# ============================================================
# Gain scheduler settings
# ============================================================

GAIN_UPDATE_INTERVAL = 10  # update gains every 10 control steps

# ============================================================
# Simulink settings
# ============================================================

SIMULINK_MODEL_NAME = "Motor"
SIMULINK_STOP_TIME = 10.0
SIMULINK_MAT_FILE = "matlab.mat"

# ============================================================
# Simulink gain sweep setting
# ============================================================

TARGET_LIST = [50, 100, 150, 200]

SWEEP_KP_LIST = [3.4, 3.6, 3.8, 4.0, 4.2, 4.4]
SWEEP_KI_LIST = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
SWEEP_KD_LIST = [0.0]

SIMULINK_SWEEP_STOP_TIME = 10.0

# ============================================================
# Target-based PID gain database
# ============================================================

USE_GAIN_DB = True

# "nearest" 또는 "linear"
# nearest: 가장 가까운 target의 gain 사용
# linear : target 사이를 선형 보간
GAIN_DB_MODE = "linear"

PID_GAIN_DB = {
    50.0:  {"kp": 4.300, "ki": 8.250, "kd": 0.000},
    75.0:  {"kp": 4.250, "ki": 6.850, "kd": 0.000},
    100.0: {"kp": 4.200, "ki": 6.000, "kd": 0.000},
    125.0: {"kp": 4.000, "ki": 5.000, "kd": 0.000},
    150.0: {"kp": 4.400, "ki": 4.000, "kd": 0.000},
    175.0: {"kp": 3.700, "ki": 3.250, "kd": 0.000},
    200.0: {"kp": 4.400, "ki": 2.500, "kd": 0.000},
}

# ============================================================
# Disturbance settings
# ============================================================

USE_DISTURBANCE = True

# disturbance mode:
# "none", "step", "pulse", "sin"
DISTURBANCE_MODE = "pulse"

# disturbance가 시작되는 시간 [s]
DISTURBANCE_START_TIME = 3.0

# disturbance가 끝나는 시간 [s]
DISTURBANCE_END_TIME = 6.0

# disturbance 크기
# SimpleMotorEnv에서 current 또는 rpm을 감소시키는 부하항으로 사용
DISTURBANCE_MAGNITUDE = 20.0

# sinusoidal disturbance용
DISTURBANCE_FREQ = 1.0

# ============================================================
# Score function settings
# ============================================================

OVERSHOOT_WEIGHT = 30.0
PWM_WEIGHT = 0.01
SATURATION_WEIGHT = 20.0

# PWM saturation 판단 기준
# exact: PWM_MAX에 거의 도달한 경우만 saturation
PWM_SATURATION_TOL = 1e-9

# ============================================================
# Saturation-aware adaptive gain settings
# ============================================================

USE_SATURATION_AWARE_GAIN = True

# PWM_MAX=255이지만, 실제 제어에서는 이 값 이상을 saturation 위험 구간으로 간주
PWM_SOFT_LIMIT = 240.0

# saturation이 몇 step 연속 발생해야 gain 완화할지
SATURATION_CONSECUTIVE_STEPS = 3

# target 대비 error가 너무 작으면 saturation이어도 gain을 줄이지 않음
SATURATION_ERROR_THRESHOLD_RATIO = 0.02

# saturation 발생 시 gain scale 감소율
SATURATION_KP_DECAY = 0.99
SATURATION_KI_DECAY = 0.98
SATURATION_KD_DECAY = 1.00

# gain scale의 최소값
SATURATION_MIN_GAIN_SCALE = 0.85

# saturation이 없을 때 원래 DB gain으로 천천히 복귀하는 속도
SATURATION_RECOVERY_RATE = 0.004

# ============================================================
# Experiment tag
# ============================================================

EXPERIMENT_TAG = "adaptive_saturation_aware"

# ============================================================
# Motor backend settings
# ============================================================

MOTOR_BACKEND = "esp32"  # "simulation" or "esp32"

# ============================================================
# ESP32 serial settings
# ============================================================

ESP32_PORT = "/dev/ttyUSB0"
ESP32_BAUDRATE = 115200
ESP32_TIMEOUT = 0.2

# Raised from 140/120: this unit needs ~1.4x the PWM of the original
# testbed for the same rpm, so 140 put the 95 rpm target at saturation.
REAL_PWM_MIN = 0.0
REAL_PWM_MAX = 200.0
REAL_PWM_SOFT_LIMIT = 170.0

# ============================================================
# Real ESP32 motor PID gain DB
# ============================================================

ESP32_REAL_PID_GAIN_DB = {
    30.0: {"kp": 1.2000, "ki": 0.7000, "kd": 0.0000},
    50.0: {"kp": 1.2000, "ki": 0.7000, "kd": 0.0000},
    70.0: {"kp": 0.6000, "ki": 1.6000, "kd": 0.0300},
    85.0: {"kp": 1.0000, "ki": 1.9000, "kd": 0.1000},
    100.0: {"kp": 1.0000, "ki": 1.6000, "kd": 0.1000},
}

ESP32_REAL_GAIN_DB_MODE = "linear"

# ============================================================
# Real ESP32 motor gain sweep settings
# ============================================================

ESP32_SWEEP_TARGET_LIST = [30.0, 50.0, 70.0]

ESP32_SWEEP_KP_LIST = [0.6, 0.8, 1.0, 1.2, 1.4]
ESP32_SWEEP_KI_LIST = [0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5]
ESP32_SWEEP_KD_LIST = [0.0, 0.03, 0.06, 0.1]

ESP32_SWEEP_TEST_TIME = 20.0
ESP32_SWEEP_REST_TIME = 2.0

ESP32_SWEEP_PWM_MIN = 0.0
ESP32_SWEEP_PWM_MAX = 200.0
ESP32_SWEEP_PWM_RATE_LIMIT = 20.0

