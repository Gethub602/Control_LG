# Motor/src/message_schema.py

import time
import json


def now_timestamp() -> float:
    return time.time()


def make_motor_state_message(
    run_id: str,
    device_id: str,
    seq: int,
    mode: str,
    target: float,
    current: float,
    error: float,
    error_derivative: float,
    pwm: float,
    kp: float,
    ki: float,
    kd: float,
    kp_scale: float = 1.0,
    ki_scale: float = 1.0,
    saturation_active: bool = False,
    control_time: float = None,
    measured_time: float = None,
    dt: float = None,
    prev_pwm: float = None,
    integral: float = None,
    backend: str = None,
    encoder: int = None,
    metadata: dict = None,
):
    """
    Local controller -> server recommender
    """

    message = {
        "timestamp": now_timestamp(),
        "run_id": run_id,
        "device_id": device_id,
        "seq": int(seq),
        "mode": mode,

        "target": float(target),
        "current": float(current),
        "error": float(error),
        "error_derivative": float(error_derivative),
        "pwm": float(pwm),

        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),

        "kp_scale": float(kp_scale),
        "ki_scale": float(ki_scale),
        "saturation_active": bool(saturation_active),
    }

    optional_float_fields = {
        "control_time": control_time,
        "measured_time": measured_time,
        "dt": dt,
        "prev_pwm": prev_pwm,
        "integral": integral,
    }

    for key, value in optional_float_fields.items():
        if value is not None:
            message[key] = float(value)

    if backend is not None:
        message["backend"] = str(backend)

    if encoder is not None:
        message["encoder"] = int(encoder)

    if metadata:
        message["metadata"] = dict(metadata)

    return message


def make_gain_command_message(
    run_id: str,
    device_id: str,
    source_seq: int,
    target: float,
    kp: float,
    ki: float,
    kd: float,
    confidence: float = 1.0,
    valid_for_sec: float = 1.0,
    reason: str = "server_recommendation",
):
    """
    Server recommender -> local controller
    """

    timestamp = now_timestamp()

    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "device_id": device_id,
        "source_seq": int(source_seq),

        "target": float(target),
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),

        "confidence": float(confidence),
        "valid_until": timestamp + float(valid_for_sec),
        "reason": reason,
    }


def is_valid_gain_command(
    command: dict,
    current_run_id: str,
    current_device_id: str,
    current_target: float,
    latest_seq: int,
    now: float = None,
) -> tuple[bool, str]:
    """
    stale command 방지.
    """

    if now is None:
        now = now_timestamp()

    if command.get("run_id") != current_run_id:
        return False, "run_id_mismatch"

    if command.get("device_id") != current_device_id:
        return False, "device_id_mismatch"

    if abs(float(command.get("target", -999999.0)) - float(current_target)) > 1e-9:
        return False, "target_mismatch"

    if int(command.get("source_seq", -1)) < int(latest_seq):
        return False, "stale_seq"

    if now > float(command.get("valid_until", 0.0)):
        return False, "expired"

    return True, "valid"


def json_serializer(data: dict):
    return json.dumps(data).encode("utf-8")


def json_deserializer(data: bytes):
    return json.loads(data.decode("utf-8"))
