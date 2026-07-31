"""
PyTorch gain-chunk schedule generators for the asynchronous server.

These plug into the same server/controller path as the TensorFlow generators in
schedule_generators.py and emit identical schedule-chunk messages. Only model
execution differs, which is what lets the GPU (compute capability 12.0) be used
at all -- see torch_gain_chunk_common for why the TF stack cannot.

All the conditioning, candidate-selection and message-building helpers are
inherited from DiffusionUnetGainChunkGenerator's base class, so feature
construction stays byte-identical between frameworks.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))

from schedule_generators import (  # noqa: E402
    DirectGainChunkPolicyGenerator,
    ScheduleGenerator,
)
from schedule_schema import (  # noqa: E402
    PAYLOAD_KIND_GAIN,
    make_schedule_chunk_message,
    make_schedule_item,
)
from torch_gain_chunk_common import (  # noqa: E402
    GainChunkUNet,
    cosine_beta_schedule,
    get_device,
)


class TorchGainChunkGenerator(DirectGainChunkPolicyGenerator):
    """
    Base class for torch-backed chunk generators.

    Deliberately does not call super().__init__: the parent constructor loads a
    TensorFlow policy. Only the parent's stateless feature helpers are reused.
    """

    generator_id = "torch_gain_chunk_generator"
    payload_kind = PAYLOAD_KIND_GAIN

    def __init__(
        self,
        model_path,
        backend_name: str = "esp32",
        fallback_generator: Optional[ScheduleGenerator] = None,
        sample_count: int = 1,
        device: Optional[str] = None,
        deterministic_seed: int = 0,
        smooth_passes: int = 0,
    ):
        payload = joblib.load(model_path)
        arch = payload["architecture"]

        self.payload = payload
        self.model_path = str(model_path)
        self.weights_path = str(payload["weights_path"])
        self.backend_name = backend_name
        self.fallback_generator = fallback_generator
        self.sample_count = int(sample_count)
        self.deterministic_seed = int(deterministic_seed)
        self.smooth_passes = int(smooth_passes)

        self.obs_cols = list(payload["obs_cols"])
        self.static_feature_cols = list(payload["static_feature_cols"])
        self.gain_cols = list(payload["gain_cols"])
        self.gain_bounds = {
            str(k): (float(v[0]), float(v[1]))
            for k, v in payload["gain_bounds"].items()
        }
        self.obs_steps = int(payload["obs_steps"])
        self.trained_horizon_steps = int(payload["horizon_steps"])
        self.seq_scaler = payload["seq_scaler"]
        self.static_scaler = payload["static_scaler"]
        self.diffusion_steps = int(payload["diffusion_steps"])

        self.device = torch.device(device) if device else get_device()
        self.model = GainChunkUNet(
            obs_dim=len(self.obs_cols),
            static_dim=len(self.static_feature_cols),
            horizon_steps=self.trained_horizon_steps,
            gain_dim=len(self.gain_cols),
            base_filters=int(arch["base_filters"]),
            cond_dim=int(arch["condition_dim"]),
            time_embed_dim=int(arch["time_embed_dim"]),
            dropout=float(arch["dropout"]),
            norm=str(arch["norm"]),
            condition_mode=str(arch["condition_mode"]),
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)
        )
        self.model.eval()

    def warm_up(self):
        """
        Pay CUDA kernel compilation once, not on the first control step.

        Must be called by the concrete subclass after its sampler attributes are
        set; calling it from this base __init__ would run before ddim_steps /
        flow_steps exist.
        """
        obs = np.zeros((1, self.obs_steps, len(self.obs_cols)), dtype=np.float32)
        static = np.zeros((1, len(self.static_feature_cols)), dtype=np.float32)
        self._sample(obs, static)

    def _sample(self, x_seq: np.ndarray, x_static: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _to_device(self, x_seq, x_static):
        obs = torch.from_numpy(np.asarray(x_seq, dtype=np.float32)).to(self.device)
        static = torch.from_numpy(np.asarray(x_static, dtype=np.float32)).to(self.device)
        return (
            obs.repeat_interleave(self.sample_count, dim=0),
            static.repeat_interleave(self.sample_count, dim=0),
        )

    def _smooth(self, gain_actual: np.ndarray) -> np.ndarray:
        out = np.asarray(gain_actual, dtype=np.float32).copy()
        for _ in range(self.smooth_passes):
            out[1:-1] = (out[:-2] + out[1:-1] * 2.0 + out[2:]) / 4.0
        return out

    def generate(
        self,
        state: Dict[str, Any],
        schedule_start_time: float,
        dt: float,
        horizon_steps: int,
    ) -> Dict[str, Any]:
        try:
            history = state.get("_history", [])
            raw_seq, raw_static = self._build_raw_inputs_for_columns(
                state=state,
                history=history,
                obs_cols=self.obs_cols,
                static_feature_cols=self.static_feature_cols,
                obs_steps=self.obs_steps,
            )
            x_seq, x_static = self._scale_raw_inputs(
                obs=raw_seq,
                static=raw_static,
                seq_scaler=self.seq_scaler,
                static_scaler=self.static_scaler,
            )
            if self.deterministic_seed > 0:
                torch.manual_seed(self.deterministic_seed)

            samples = self._sample(x_seq, x_static)  # (1, S, H, G) in [-1, 1]
            pred = np.mean(samples[0], axis=0)
            pred_norm = np.clip((pred + 1.0) / 2.0, 0.0, 1.0)
            gain_sequence = self._decode_gain_sequence(pred_norm, int(horizon_steps))
        except Exception:
            if self.fallback_generator is None:
                raise
            return self.fallback_generator.generate(
                state=state,
                schedule_start_time=schedule_start_time,
                dt=dt,
                horizon_steps=horizon_steps,
            )

        target = float(state["target"])
        items = [
            make_schedule_item(
                step_index=i,
                control_time=float(schedule_start_time) + i * float(dt),
                target=target,
                kp=float(g["kp"]),
                ki=float(g["ki"]),
                kd=float(g["kd"]),
            )
            for i, g in enumerate(gain_sequence)
        ]

        return make_schedule_chunk_message(
            run_id=state["run_id"],
            device_id=state["device_id"],
            source_seq=int(state["seq"]),
            source_timestamp=float(state["timestamp"]),
            source_control_time=float(state.get("control_time", 0.0)),
            schedule_start_time=schedule_start_time,
            dt=dt,
            items=items,
            payload_kind=self.payload_kind,
            generator_id=f"{self.backend_name}_{self.generator_id}",
            reason=self.chunk_reason,
            metadata={
                "backend": self.backend_name,
                "framework": "pytorch",
                "device": str(self.device),
                "model_path": self.model_path,
                "target": target,
                "sample_count": self.sample_count,
                "sequence_length": len(gain_sequence),
                "first_gain": dict(gain_sequence[0]),
                "last_gain": dict(gain_sequence[-1]),
            },
        )


class TorchDiffusionGainChunkGenerator(TorchGainChunkGenerator):
    """DDIM sampling of the learned reverse diffusion process."""

    generator_id = "torch_diffusion_gain_chunk_generator"
    chunk_reason = "torch_diffusion_ddim_gain_chunk_prediction"

    def __init__(self, model_path, ddim_steps: int = 20, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        self.ddim_steps = int(ddim_steps)
        betas = cosine_beta_schedule(self.diffusion_steps)
        self.alpha_bars = np.cumprod(1.0 - betas).astype(np.float32)
        self.warm_up()

    @torch.no_grad()
    def _sample(self, x_seq, x_static):
        obs, static = self._to_device(x_seq, x_static)
        total = obs.shape[0]
        h, g = self.trained_horizon_steps, len(self.gain_cols)
        x = torch.randn(total, h, g, device=self.device)
        idx = np.linspace(self.diffusion_steps - 1, 0, self.ddim_steps, dtype=int)

        for i, t_idx in enumerate(idx):
            t = torch.full((total,), float(t_idx), device=self.device)
            eps = self.model(x, t, obs, static)
            a_t = float(self.alpha_bars[int(t_idx)])
            x0 = ((x - float(np.sqrt(max(1.0 - a_t, 1e-12))) * eps)
                  / float(np.sqrt(max(a_t, 1e-12)))).clamp(-1.5, 1.5)
            if i == len(idx) - 1:
                x = x0
            else:
                a_prev = float(self.alpha_bars[int(idx[i + 1])])
                x = (float(np.sqrt(max(a_prev, 1e-12))) * x0
                     + float(np.sqrt(max(1.0 - a_prev, 1e-12))) * eps)

        x = x.clamp(-1.0, 1.0).cpu().numpy()
        return x.reshape(-1, self.sample_count, h, g)


class TorchFlowMatchingGainChunkGenerator(TorchGainChunkGenerator):
    """ODE integration of the learned velocity field (rectified flow)."""

    generator_id = "torch_flow_matching_gain_chunk_generator"
    chunk_reason = "torch_flow_matching_ode_gain_chunk_prediction"

    def __init__(self, model_path, flow_steps: int = 2, flow_solver: str = "", **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        self.flow_steps = int(flow_steps)
        self.flow_solver = str(flow_solver or self.payload.get("flow_solver", "midpoint"))
        self.warm_up()

    @torch.no_grad()
    def _sample(self, x_seq, x_static):
        obs, static = self._to_device(x_seq, x_static)
        total = obs.shape[0]
        h, g = self.trained_horizon_steps, len(self.gain_cols)
        y = torch.randn(total, h, g, device=self.device)
        dt = 1.0 / float(self.flow_steps)
        scale = float(self.diffusion_steps - 1)

        def velocity(y_cur, t_scalar):
            t = torch.full((total,), float(t_scalar) * scale, device=self.device)
            return self.model(y_cur, t, obs, static)

        for i in range(self.flow_steps):
            t0 = i * dt
            if self.flow_solver == "euler":
                y = y + dt * velocity(y, t0)
            elif self.flow_solver == "midpoint":
                k1 = velocity(y, t0)
                y = y + dt * velocity(y + 0.5 * dt * k1, t0 + 0.5 * dt)
            else:  # heun
                k1 = velocity(y, t0)
                k2 = velocity(y + dt * k1, min(t0 + dt, 1.0))
                y = y + 0.5 * dt * (k1 + k2)

        y = y.clamp(-1.0, 1.0).cpu().numpy()
        return y.reshape(-1, self.sample_count, h, g)
