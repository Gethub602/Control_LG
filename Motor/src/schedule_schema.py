import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple


MESSAGE_TYPE_SCHEDULE_CHUNK = "schedule_chunk"

PAYLOAD_KIND_GAIN = "gain"
PAYLOAD_KIND_PWM = "pwm"
PAYLOAD_KIND_FEEDFORWARD_PWM = "feedforward_pwm"
PAYLOAD_KIND_HYBRID = "hybrid"

SUPPORTED_PAYLOAD_KINDS = {
    PAYLOAD_KIND_GAIN,
    PAYLOAD_KIND_PWM,
    PAYLOAD_KIND_FEEDFORWARD_PWM,
    PAYLOAD_KIND_HYBRID,
}


def now_timestamp() -> float:
    return time.time()


def _maybe_float(value):
    if value is None:
        return None
    return float(value)


def make_schedule_item(
    step_index: int,
    control_time: float,
    target: Optional[float] = None,
    kp: Optional[float] = None,
    ki: Optional[float] = None,
    kd: Optional[float] = None,
    pwm: Optional[float] = None,
    u_ff: Optional[float] = None,
    pwm_ref: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build one schedule row.

    control_time is the local controller time in seconds, not wall-clock time.
    Optional fields let the same schema carry gain-only, PWM-only, feedforward,
    or hybrid schedules.
    """

    item = {
        "step_index": int(step_index),
        "control_time": float(control_time),
    }

    optional_fields = {
        "target": _maybe_float(target),
        "kp": _maybe_float(kp),
        "ki": _maybe_float(ki),
        "kd": _maybe_float(kd),
        "pwm": _maybe_float(pwm),
        "u_ff": _maybe_float(u_ff),
        "pwm_ref": _maybe_float(pwm_ref),
    }

    for key, value in optional_fields.items():
        if value is not None:
            item[key] = value

    if metadata:
        item["metadata"] = dict(metadata)

    return item


def make_constant_gain_items(
    schedule_start_time: float,
    dt: float,
    horizon_steps: int,
    kp: float,
    ki: float,
    kd: float,
    target: Optional[float] = None,
) -> List[Dict[str, Any]]:
    items = []

    for step_index in range(int(horizon_steps)):
        control_time = float(schedule_start_time) + step_index * float(dt)
        items.append(
            make_schedule_item(
                step_index=step_index,
                control_time=control_time,
                target=target,
                kp=kp,
                ki=ki,
                kd=kd,
            )
        )

    return items


def make_schedule_chunk_message(
    run_id: str,
    device_id: str,
    source_seq: int,
    source_timestamp: float,
    source_control_time: float,
    schedule_start_time: float,
    dt: float,
    items: Iterable[Dict[str, Any]],
    payload_kind: str = PAYLOAD_KIND_HYBRID,
    chunk_id: Optional[str] = None,
    generator_id: str = "unknown_generator",
    confidence: float = 1.0,
    valid_until: Optional[float] = None,
    valid_for_sec: Optional[float] = None,
    reason: str = "schedule_chunk",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Server -> local controller schedule chunk.

    schedule_start_time/source_control_time are local controller times in
    seconds. timestamp/generated_at/valid_until are wall-clock times.
    """

    if payload_kind not in SUPPORTED_PAYLOAD_KINDS:
        raise ValueError(f"Unsupported payload_kind: {payload_kind}")

    generated_at = now_timestamp()
    item_list = [dict(item) for item in items]

    if valid_until is None and valid_for_sec is not None:
        valid_until = generated_at + float(valid_for_sec)

    if valid_until is None:
        last_item_time = (
            max(float(item["control_time"]) for item in item_list)
            if item_list
            else float(schedule_start_time)
        )
        remaining_control_horizon = max(0.0, last_item_time - float(source_control_time))
        valid_until = generated_at + remaining_control_horizon + 1.0

    return {
        "message_type": MESSAGE_TYPE_SCHEDULE_CHUNK,
        "timestamp": generated_at,
        "generated_at": generated_at,
        "run_id": run_id,
        "device_id": device_id,
        "chunk_id": chunk_id or uuid.uuid4().hex,
        "generator_id": generator_id,
        "source_seq": int(source_seq),
        "source_timestamp": float(source_timestamp),
        "source_control_time": float(source_control_time),
        "schedule_start_time": float(schedule_start_time),
        "dt": float(dt),
        "horizon_steps": len(item_list),
        "payload_kind": payload_kind,
        "items": item_list,
        "confidence": float(confidence),
        "valid_until": float(valid_until),
        "reason": reason,
        "metadata": dict(metadata or {}),
    }


def is_schedule_chunk_message(message: Dict[str, Any]) -> bool:
    return message.get("message_type") == MESSAGE_TYPE_SCHEDULE_CHUNK


def validate_schedule_chunk_message(
    message: Dict[str, Any],
    current_run_id: Optional[str] = None,
    current_device_id: Optional[str] = None,
    now: Optional[float] = None,
) -> Tuple[bool, str]:
    if now is None:
        now = now_timestamp()

    if not is_schedule_chunk_message(message):
        return False, "not_schedule_chunk"

    if current_run_id is not None and message.get("run_id") != current_run_id:
        return False, "run_id_mismatch"

    if current_device_id is not None and message.get("device_id") != current_device_id:
        return False, "device_id_mismatch"

    if str(message.get("payload_kind", "")) not in SUPPORTED_PAYLOAD_KINDS:
        return False, "unsupported_payload_kind"

    try:
        dt = float(message["dt"])
        horizon_steps = int(message["horizon_steps"])
        valid_until = float(message["valid_until"])
        items = message["items"]
    except (KeyError, TypeError, ValueError):
        return False, "missing_or_invalid_required_field"

    if dt <= 0.0:
        return False, "invalid_dt"

    if horizon_steps <= 0:
        return False, "empty_horizon"

    if not isinstance(items, list) or len(items) != horizon_steps:
        return False, "item_count_mismatch"

    if now > valid_until:
        return False, "expired"

    previous_time = None

    for expected_index, item in enumerate(items):
        try:
            step_index = int(item["step_index"])
            control_time = float(item["control_time"])
        except (KeyError, TypeError, ValueError):
            return False, "invalid_item"

        if step_index != expected_index:
            return False, "non_contiguous_step_index"

        if previous_time is not None and control_time <= previous_time:
            return False, "non_increasing_control_time"

        previous_time = control_time

    return True, "valid"
