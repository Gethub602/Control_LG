"""
Closed-loop comparison of gain-chunk generators under *measured* inference delay.

Unlike an offline accuracy benchmark, this runs the whole asynchronous loop:

    SimpleMotorEnv  --motor_state-->  real ScheduleGenerator  --chunk-->  ScheduleBuffer

The important detail is that the network/inference delay is not a guessed
constant. Each chunk is delivered to the buffer at

    arrival_control_time = request_control_time + measured_generate_sec + transport

where measured_generate_sec is the actual wall-clock cost of that generator call.
A slower generator therefore delivers later, misses more of its own schedule
window, and falls back more often -- which is exactly the effect that made
DDIM30 lose to DDIM20 on the real motor.

Kafka is replaced by an in-process delay queue; everything else (the generator,
the schedule schema, the delay-aware buffer, the PID loop) is the repository's
own code.
"""

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

from config import (  # noqa: E402
    ESP32_REAL_GAIN_DB_MODE,
    ESP32_REAL_PID_GAIN_DB,
    RESULTS_DIR,
)
from message_schema import make_motor_state_message  # noqa: E402
from motor_env import SimpleMotorEnv  # noqa: E402
from pid_controller import PIDController  # noqa: E402
from schedule_buffer import ScheduleBuffer  # noqa: E402
from schedule_generators import DbGainChunkGenerator  # noqa: E402
from schedule_schema import PAYLOAD_KIND_GAIN  # noqa: E402

SUMMARY_DIR = RESULTS_DIR / "summary"

CONTROL_DT = 0.01
CHUNK_DT = 0.1
RUN_ID = "closed_loop_sim"
DEVICE_ID = "sim_motor_01"


def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop generator comparison.")
    p.add_argument("--diffusion-model", default="")
    p.add_argument("--flow-model", default="")
    p.add_argument("--framework", default="torch", choices=["torch", "tensorflow"],
                   help="Which generator implementation to benchmark.")
    p.add_argument("--device", default="",
                   help="Torch device for generation, e.g. cuda or cpu. "
                        "Running on cpu reproduces an edge server whose inference "
                        "cost is large enough to interact with the chunk window.")
    p.add_argument("--ddim-steps", default="20,30")
    p.add_argument("--flow-steps", default="1,2,4")
    p.add_argument("--flow-solver", default="midpoint")
    p.add_argument("--sample-count", type=int, default=1)
    p.add_argument("--sim-time", type=float, default=20.0)
    p.add_argument("--targets", default="70,95,90,73")
    p.add_argument("--change-times", default="5,10,15")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--inference-delay", type=float, default=0.5,
                   help="Delay the server ASSUMES when placing schedule_start_time.")
    p.add_argument("--transport-sec", type=float, default=0.05,
                   help="Fixed one-way messaging cost added to measured inference.")
    p.add_argument("--latency-scale", type=float, default=1.0,
                   help="Multiplier on measured generation time. Models a heavier "
                        "deployment (larger backbone, loaded server) while keeping "
                        "each method's relative cost, which is set by its NFE count.")
    p.add_argument("--state-publish-every", type=int, default=10)
    p.add_argument("--server-min-interval", type=float, default=0.5)
    p.add_argument("--chunk-ttl", type=float, default=3.0)
    p.add_argument("--horizon-steps", type=int, default=20)
    p.add_argument("--pwm-min", type=float, default=0.0)
    p.add_argument("--pwm-max", type=float, default=140.0)
    p.add_argument("--pwm-rate-limit", type=float, default=20.0)
    p.add_argument("--tau-motor", type=float, default=0.4)
    p.add_argument("--k-motor", type=float, default=1.0)
    p.add_argument("--schedule-apply-mode", default="delay_aware",
                   choices=["delay_aware", "naive"])
    p.add_argument("--run-label", default="")
    return p.parse_args()


def parse_floats(text):
    return [float(v) for v in str(text).split(",") if str(v).strip()]


def parse_ints(text):
    return [int(v) for v in str(text).split(",") if str(v).strip()]


def make_db_generator():
    return DbGainChunkGenerator(
        gain_db=ESP32_REAL_PID_GAIN_DB,
        mode=ESP32_REAL_GAIN_DB_MODE,
        fallback_gain=(1.2, 0.7, 0.0),
        backend_name="sim",
    )


def db_gain_for(target):
    return make_db_generator()._interpolate_gain(float(target))


def set_gains_preserving_integral(pid, kp, ki, kd):
    """Keep Ki * integral continuous so a gain switch does not jolt the output."""
    old_ki = pid.ki
    if abs(ki) > 1e-9 and abs(old_ki) > 1e-9:
        pid.integral = pid.integral * (old_ki / ki)
    pid.set_gains(kp, ki, kd)


def apply_rate_limit(cmd, prev, limit):
    return float(np.clip(cmd, prev - limit, prev + limit))


def run_closed_loop(generator, args, targets, change_times, seed):
    rng = np.random.default_rng(seed)
    n_steps = int(args.sim_time / CONTROL_DT)

    env = SimpleMotorEnv(
        dt=CONTROL_DT,
        k_motor=args.k_motor,
        tau_motor=args.tau_motor,
        pwm_min=args.pwm_min,
        pwm_max=args.pwm_max,
        use_disturbance=False,
    )
    env.reset()

    pid = PIDController(dt=CONTROL_DT, output_min=args.pwm_min, output_max=args.pwm_max)
    kp0, ki0, kd0 = db_gain_for(targets[0])
    pid.set_gains(kp0, ki0, kd0)

    buf = ScheduleBuffer(run_id=RUN_ID, device_id=DEVICE_ID, max_chunks=8)
    inflight = deque()
    history = deque(maxlen=40)

    def target_at(t):
        idx = 0
        for i, ct in enumerate(change_times):
            if t >= ct:
                idx = i + 1
        return targets[min(idx, len(targets) - 1)]

    current = env.get_state()
    prev_pwm = 0.0
    prev_err = targets[0] - current
    last_server_t = -1e9
    last_server_target = None
    t0_wall = time.time()

    log = {k: [] for k in ["t", "target", "current", "pwm", "kp", "ki", "kd", "src"]}
    gen_times = []
    n_accepted = n_applied = n_fallback = 0
    n_generator_error = 0

    for step in range(n_steps):
        t = step * CONTROL_DT
        wall = t0_wall + t
        target = target_at(t)

        while inflight and inflight[0][0] <= t:
            _, chunk = inflight.popleft()
            ok, _ = buf.add_chunk(chunk, now=wall, accepted_control_time=t)
            if ok:
                n_accepted += 1

        if args.schedule_apply_mode == "delay_aware":
            look = buf.get_item(t, now=wall, payload_kind=PAYLOAD_KIND_GAIN)
        else:
            look = buf.get_item_naive(t, now=wall, payload_kind=PAYLOAD_KIND_GAIN)

        src = "fallback"
        if look.found:
            it = look.item
            set_gains_preserving_integral(pid, it["kp"], it["ki"], it["kd"])
            src = "chunk"
            n_applied += 1
        else:
            n_fallback += 1

        err = target - current
        raw = pid.compute(target, current)
        pwm = apply_rate_limit(raw, prev_pwm, args.pwm_rate_limit)
        pwm = float(np.clip(pwm, args.pwm_min, args.pwm_max))

        env.step(pwm)
        current = env.get_state()

        if step % args.state_publish_every == 0:
            same = (last_server_target is not None
                    and abs(last_server_target - target) < 1e-9)
            if not (same and (t - last_server_t) < args.server_min_interval):
                state = make_motor_state_message(
                    run_id=RUN_ID, device_id=DEVICE_ID, seq=step,
                    mode="local_kafka_controller_esp32",
                    target=target, current=current, error=target - current,
                    error_derivative=(err - prev_err) / CONTROL_DT,
                    pwm=pwm, kp=pid.kp, ki=pid.ki, kd=pid.kd,
                    control_time=t, dt=CHUNK_DT, prev_pwm=prev_pwm,
                    integral=pid.integral, backend="esp32",
                )
                state["timestamp"] = wall
                history.append(dict(state))
                state["_history"] = list(history)

                gen_t0 = time.perf_counter()
                try:
                    chunk = generator.generate(
                        state=state,
                        schedule_start_time=t + args.inference_delay,
                        dt=CHUNK_DT,
                        horizon_steps=args.horizon_steps,
                    )
                except Exception as exc:  # generator must never stop the loop
                    n_generator_error += 1
                    print(f"    [GEN_ERROR] t={t:.2f}: {exc}")
                    chunk = None
                gen_sec = time.perf_counter() - gen_t0

                if chunk is not None:
                    gen_sec *= float(args.latency_scale)
                    gen_times.append(gen_sec)
                    chunk["generated_at"] = wall + gen_sec
                    chunk["valid_until"] = wall + gen_sec + args.chunk_ttl
                    # delivery is driven by the REAL cost of this generator call
                    arrival = t + gen_sec + args.transport_sec
                    inflight.append((arrival, chunk))

                last_server_t = t
                last_server_target = target

        prev_pwm, prev_err = pwm, err
        log["t"].append(t); log["target"].append(target)
        log["current"].append(current); log["pwm"].append(pwm)
        log["kp"].append(pid.kp); log["ki"].append(pid.ki); log["kd"].append(pid.kd)
        log["src"].append(src)

    t_arr = np.array(log["t"])
    e_arr = np.array(log["target"]) - np.array(log["current"])
    after = np.zeros_like(t_arr, dtype=bool)
    for ct in change_times:
        after |= (t_arr >= ct) & (t_arr < ct + 2.0)

    return {
        "IAE": float(np.sum(np.abs(e_arr)) * CONTROL_DT),
        "after_change_IAE": float(np.sum(np.abs(e_arr[after])) * CONTROL_DT),
        "final_error": float(abs(e_arr[-1])),
        "max_pwm": float(np.max(log["pwm"])),
        "saturation_pct": float(np.mean(np.array(log["pwm"]) >= args.pwm_max - 1e-6) * 100),
        "chunks_accepted": n_accepted,
        "steps_applied": n_applied,
        "steps_fallback": n_fallback,
        "fallback_pct": 100.0 * n_fallback / max(n_steps, 1),
        "generator_calls": len(gen_times),
        "generator_errors": n_generator_error,
        "gen_mean_sec": float(np.mean(gen_times)) if gen_times else float("nan"),
        "gen_p90_sec": float(np.percentile(gen_times, 90)) if gen_times else float("nan"),
        "_log": log,
    }


def build_generators(args):
    """Return [(name, generator), ...]; models are loaded once and reused."""
    if args.framework == "torch":
        from torch_schedule_generators import (
            TorchDiffusionGainChunkGenerator as DiffusionGen,
            TorchFlowMatchingGainChunkGenerator as FlowGen,
        )
    else:
        from schedule_generators import (
            DiffusionUnetGainChunkGenerator as DiffusionGen,
            FlowMatchingGainChunkGenerator as FlowGen,
        )

    device_kwargs = (
        {"device": args.device} if (args.device and args.framework == "torch") else {}
    )
    db = make_db_generator()
    entries = [("db_baseline", db)]

    if args.diffusion_model:
        for steps in parse_ints(args.ddim_steps):
            entries.append((
                f"diffusion_ddim{steps}",
                DiffusionGen(
                    model_path=args.diffusion_model,
                    backend_name="esp32",
                    fallback_generator=db,
                    ddim_steps=steps,
                    sample_count=args.sample_count,
                    **device_kwargs,
                ),
            ))

    if args.flow_model:
        for steps in parse_ints(args.flow_steps):
            entries.append((
                f"flow_{args.flow_solver}{steps}",
                FlowGen(
                    model_path=args.flow_model,
                    flow_steps=steps,
                    flow_solver=args.flow_solver,
                    backend_name="esp32",
                    fallback_generator=db,
                    sample_count=args.sample_count,
                    **device_kwargs,
                ),
            ))

    return entries


def main():
    args = parse_args()
    targets = parse_floats(args.targets)
    change_times = parse_floats(args.change_times)

    print(f"Scenario: {targets} at {change_times}s, "
          f"apply_mode={args.schedule_apply_mode}, repeats={args.repeats}")
    print(f"Server latency assumption: {args.inference_delay}s "
          f"(actual delivery uses measured generate time + {args.transport_sec}s)")
    print()

    rows = []
    for name, generator in build_generators(args):
        runs = []
        for r in range(args.repeats):
            runs.append(run_closed_loop(generator, args, targets, change_times, seed=r))
        agg = {"method": name, "runs": args.repeats}
        for key in ["IAE", "after_change_IAE", "final_error", "saturation_pct",
                    "chunks_accepted", "steps_applied", "fallback_pct",
                    "gen_mean_sec", "gen_p90_sec", "generator_errors"]:
            vals = [r[key] for r in runs]
            agg[f"{key}_mean"] = float(np.mean(vals))
            if key in ("IAE", "after_change_IAE"):
                agg[f"{key}_std"] = float(np.std(vals))
        rows.append(agg)
        print(f"  {name:<22} IAE={agg['IAE_mean']:7.2f}+-{agg['IAE_std']:.2f}  "
              f"after={agg['after_change_IAE_mean']:6.2f}  "
              f"fallback={agg['fallback_pct_mean']:5.1f}%  "
              f"gen_p90={agg['gen_p90_sec_mean'] * 1000:6.1f}ms")

    out = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or "generator_closed_loop"
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / f"{label}_{timestamp}.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
