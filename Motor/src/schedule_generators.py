from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config import ESP32_REAL_PID_GAIN_DB
from schedule_schema import (
    PAYLOAD_KIND_GAIN,
    make_constant_gain_items,
    make_schedule_item,
    make_schedule_chunk_message,
)


class ScheduleGenerator(ABC):
    """
    Backend interface for chunk generation.

    Implementations can be DB-based, model-based, physics-guided, or hybrid.
    The local controller only needs the resulting schedule chunk message.
    """

    generator_id = "base_generator"
    payload_kind = PAYLOAD_KIND_GAIN

    @abstractmethod
    def generate(
        self,
        state: Dict[str, Any],
        schedule_start_time: float,
        dt: float,
        horizon_steps: int,
    ) -> Dict[str, Any]:
        pass


class DbGainChunkGenerator(ScheduleGenerator):
    """
    Baseline generator that repeats one DB-interpolated gain over the horizon.
    """

    generator_id = "db_gain_chunk_generator"
    payload_kind = PAYLOAD_KIND_GAIN

    def __init__(
        self,
        gain_db: Dict[float, Dict[str, float]],
        mode: str,
        fallback_gain: Tuple[float, float, float],
        backend_name: str = "generic",
    ):
        self.gain_db = gain_db
        self.mode = mode
        self.fallback_gain = fallback_gain
        self.backend_name = backend_name

    def generate(
        self,
        state: Dict[str, Any],
        schedule_start_time: float,
        dt: float,
        horizon_steps: int,
    ) -> Dict[str, Any]:
        target = float(state["target"])
        kp, ki, kd = self._interpolate_gain(target)

        items = make_constant_gain_items(
            schedule_start_time=schedule_start_time,
            dt=dt,
            horizon_steps=horizon_steps,
            kp=kp,
            ki=ki,
            kd=kd,
            target=target,
        )

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
            reason="db_gain_chunk",
            metadata={
                "backend": self.backend_name,
                "target": target,
                "base_gain": {"kp": kp, "ki": ki, "kd": kd},
            },
        )

    def _interpolate_gain(self, target: float) -> Tuple[float, float, float]:
        if not self.gain_db:
            return self.fallback_gain

        target = float(target)
        db_targets: List[float] = sorted(float(key) for key in self.gain_db.keys())

        if target in db_targets:
            gains = self.gain_db[target]
            return float(gains["kp"]), float(gains["ki"]), float(gains["kd"])

        if self.mode == "nearest":
            nearest_target = min(db_targets, key=lambda value: abs(value - target))
            gains = self.gain_db[nearest_target]
            return float(gains["kp"]), float(gains["ki"]), float(gains["kd"])

        if self.mode == "linear":
            if target <= db_targets[0]:
                gains = self.gain_db[db_targets[0]]
                return float(gains["kp"]), float(gains["ki"]), float(gains["kd"])

            if target >= db_targets[-1]:
                gains = self.gain_db[db_targets[-1]]
                return float(gains["kp"]), float(gains["ki"]), float(gains["kd"])

            for lower_target, upper_target in zip(db_targets[:-1], db_targets[1:]):
                if lower_target <= target <= upper_target:
                    ratio = (target - lower_target) / (upper_target - lower_target)
                    lower_gain = self.gain_db[lower_target]
                    upper_gain = self.gain_db[upper_target]

                    kp = lower_gain["kp"] + ratio * (upper_gain["kp"] - lower_gain["kp"])
                    ki = lower_gain["ki"] + ratio * (upper_gain["ki"] - lower_gain["ki"])
                    kd = lower_gain["kd"] + ratio * (upper_gain["kd"] - lower_gain["kd"])
                    return float(kp), float(ki), float(kd)

        nearest_target = min(db_targets, key=lambda value: abs(value - target))
        gains = self.gain_db[nearest_target]
        return float(gains["kp"]), float(gains["ki"]), float(gains["kd"])


class RandomForestCostChunkGenerator(ScheduleGenerator):
    """
    Select a constant gain chunk by minimizing model-predicted horizon cost.

    The model is trained on state/history + candidate gain -> horizon_cost.
    """

    generator_id = "rf_cost_chunk_generator"
    payload_kind = PAYLOAD_KIND_GAIN

    def __init__(
        self,
        model_path,
        candidate_gains: List[Tuple[float, float, float]],
        backend_name: str = "esp32",
        fallback_generator: ScheduleGenerator = None,
    ):
        payload = joblib.load(model_path)

        self.model = payload["model"]
        self.feature_cols = list(payload["feature_cols"])
        self.target_log1p = bool(payload.get("target_log1p", False))
        self.model_path = str(model_path)
        self.candidate_gains = [
            (float(kp), float(ki), float(kd))
            for kp, ki, kd in candidate_gains
        ]
        self.backend_name = backend_name
        self.fallback_generator = fallback_generator

        if not self.candidate_gains:
            raise ValueError("candidate_gains must not be empty")

    def generate(
        self,
        state: Dict[str, Any],
        schedule_start_time: float,
        dt: float,
        horizon_steps: int,
    ) -> Dict[str, Any]:
        history = state.get("_history", [])

        candidate_df = self._build_candidate_features(
            state=state,
            history=history,
        )

        if candidate_df.empty:
            if self.fallback_generator is None:
                raise RuntimeError("No RF candidate rows and no fallback generator")
            return self.fallback_generator.generate(
                state=state,
                schedule_start_time=schedule_start_time,
                dt=dt,
                horizon_steps=horizon_steps,
            )

        pred = self._predict_cost(candidate_df)
        best_idx = int(np.argmin(pred))
        best_row = candidate_df.iloc[best_idx]

        kp = float(best_row["kp"])
        ki = float(best_row["ki"])
        kd = float(best_row["kd"])
        predicted_cost = float(pred[best_idx])

        items = make_constant_gain_items(
            schedule_start_time=schedule_start_time,
            dt=dt,
            horizon_steps=horizon_steps,
            kp=kp,
            ki=ki,
            kd=kd,
            target=float(state["target"]),
        )

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
            reason=f"{self.generator_id}_minimization",
            metadata={
                "backend": self.backend_name,
                "model_path": self.model_path,
                "target": float(state["target"]),
                "selected_gain": {"kp": kp, "ki": ki, "kd": kd},
                "predicted_cost": predicted_cost,
                "candidate_count": int(len(candidate_df)),
                "best_rank": best_idx,
                "top_candidates": self._top_candidate_records(
                    candidate_df,
                    pred,
                    top_n=5,
                ),
            },
        )

    def _predict_cost(self, candidate_df: pd.DataFrame):
        X = candidate_df[self.feature_cols]
        pred = self.model.predict(X)

        if self.target_log1p:
            pred = np.expm1(pred)

        return np.maximum(pred, 0.0)

    def _build_candidate_features(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        rows = []

        base = self._base_features(state, history)

        for kp, ki, kd in self.candidate_gains:
            row = dict(base)
            row["kp"] = kp
            row["ki"] = ki
            row["kd"] = kd
            rows.append(row)

        df = pd.DataFrame(rows)

        for feature in self.feature_cols:
            if feature not in df.columns:
                df[feature] = 0.0

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    def _base_features(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        target = float(state.get("target", 0.0))
        current = float(state.get("current", 0.0))
        error = float(state.get("error", target - current))
        pwm = float(state.get("pwm", 0.0))

        control_time = float(state.get("control_time", 0.0))
        source_start_time = self._first_control_time(history, default=control_time)
        last_target_change_time = self._last_target_change_time(
            history,
            target=target,
            default=source_start_time,
        )

        pwm_max = 140.0 if self.backend_name == "esp32" else 255.0

        base = {
            "target": target,
            "current": current,
            "error": error,
            "error_derivative": float(state.get("error_derivative", 0.0)),
            "pwm": pwm,
            "prev_pwm": float(state.get("prev_pwm", pwm)),
            "integral": float(state.get("integral", 0.0)),
            "kp_scale": float(state.get("kp_scale", 1.0)),
            "ki_scale": float(state.get("ki_scale", 1.0)),
            "time_since_start": max(0.0, control_time - source_start_time),
            "time_since_target_change": max(0.0, control_time - last_target_change_time),
            "error_ratio": abs(error) / max(abs(target), 1e-6),
            "pwm_ratio": abs(pwm) / max(pwm_max, 1e-6),
        }

        for lag in range(1, 4):
            lag_state = self._get_lag_state(history, lag=lag, fallback=state)
            lag_target = float(lag_state.get("target", target))
            lag_current = float(lag_state.get("current", current))
            lag_error = float(lag_state.get("error", lag_target - lag_current))

            base[f"current_lag_{lag}"] = lag_current
            base[f"error_lag_{lag}"] = lag_error
            base[f"pwm_lag_{lag}"] = float(lag_state.get("pwm", pwm))

        return base

    def _first_control_time(self, history, default: float):
        if not history:
            return float(default)

        return float(history[0].get("control_time", default))

    def _last_target_change_time(self, history, target: float, default: float):
        if not history:
            return float(default)

        last_change = float(history[0].get("control_time", default))
        previous_target = float(history[0].get("target", target))

        for item in history[1:]:
            item_target = float(item.get("target", previous_target))
            if abs(item_target - previous_target) > 1e-9:
                last_change = float(item.get("control_time", last_change))
                previous_target = item_target

        if abs(float(target) - previous_target) > 1e-9:
            return float(history[-1].get("control_time", last_change))

        return last_change

    def _get_lag_state(self, history, lag: int, fallback: Dict[str, Any]):
        if len(history) >= lag + 1:
            return history[-(lag + 1)]

        if len(history) >= 1:
            return history[0]

        return fallback

    def _top_candidate_records(self, candidate_df, pred, top_n: int):
        order = np.argsort(pred)[:top_n]
        records = []

        for idx in order:
            row = candidate_df.iloc[int(idx)]
            records.append(
                {
                    "kp": float(row["kp"]),
                    "ki": float(row["ki"]),
                    "kd": float(row["kd"]),
                    "predicted_cost": float(pred[int(idx)]),
                }
            )

        return records


class MlpCostChunkGenerator(RandomForestCostChunkGenerator):
    """
    Select a constant gain chunk by minimizing TensorFlow MLP-predicted cost.

    It intentionally shares the RF feature builder so RF/MLP comparisons differ
    only in the surrogate model family.
    """

    generator_id = "mlp_cost_chunk_generator"

    def __init__(
        self,
        model_path,
        candidate_gains: List[Tuple[float, float, float]],
        backend_name: str = "esp32",
        fallback_generator: ScheduleGenerator = None,
    ):
        payload = joblib.load(model_path)

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for MlpCostChunkGenerator. "
                "Run the server in the tensorflow conda environment."
            ) from exc

        keras_model_path = payload.get("keras_model_path")
        if not keras_model_path:
            raise ValueError(f"Missing keras_model_path in MLP payload: {model_path}")

        self.model = tf.keras.models.load_model(keras_model_path)
        self.scaler = payload["scaler"]
        self.feature_cols = list(payload["feature_cols"])
        self.target_log1p = bool(payload.get("target_log1p", True))
        self.model_path = str(model_path)
        self.keras_model_path = str(keras_model_path)
        self.candidate_gains = [
            (float(kp), float(ki), float(kd))
            for kp, ki, kd in candidate_gains
        ]
        self.backend_name = backend_name
        self.fallback_generator = fallback_generator

        if not self.candidate_gains:
            raise ValueError("candidate_gains must not be empty")

    def _predict_cost(self, candidate_df: pd.DataFrame):
        X = candidate_df[self.feature_cols]
        X_scaled = self.scaler.transform(X).astype(np.float32)
        pred = self.model.predict(
            X_scaled,
            batch_size=max(1, len(X_scaled)),
            verbose=0,
        ).reshape(-1)

        if self.target_log1p:
            pred = np.expm1(pred)

        return np.maximum(pred, 0.0)


class MultiTaskMlpCostChunkGenerator(MlpCostChunkGenerator):
    """
    Risk-aware MLP generator using multiple predicted closed-loop metrics.

    The policy does not use a DB prior or previous-gain proximity. It selects
    the candidate with the lowest weighted score computed from model-predicted
    future IAE, overshoot, saturation risk, PWM level, and PWM variation.
    """

    generator_id = "multitask_mlp_cost_chunk_generator"

    def __init__(
        self,
        model_path,
        candidate_gains: List[Tuple[float, float, float]],
        backend_name: str = "esp32",
        fallback_generator: ScheduleGenerator = None,
    ):
        payload = joblib.load(model_path)

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for MultiTaskMlpCostChunkGenerator. "
                "Run the server in the tensorflow conda environment."
            ) from exc

        keras_model_path = payload.get("keras_model_path")
        if not keras_model_path:
            raise ValueError(
                f"Missing keras_model_path in multi-task MLP payload: {model_path}"
            )

        self.model = tf.keras.models.load_model(keras_model_path)
        self.scaler = payload["scaler"]
        self.feature_cols = list(payload["feature_cols"])
        self.target_cols = list(payload["target_cols"])
        self.target_log1p = bool(payload.get("target_log1p", True))
        self.score_weights = {
            str(key): float(value)
            for key, value in payload.get("score_weights", {}).items()
        }
        self.model_path = str(model_path)
        self.keras_model_path = str(keras_model_path)
        self.candidate_gains = [
            (float(kp), float(ki), float(kd))
            for kp, ki, kd in candidate_gains
        ]
        self.backend_name = backend_name
        self.fallback_generator = fallback_generator

        if not self.candidate_gains:
            raise ValueError("candidate_gains must not be empty")

    def _predict_cost(self, candidate_df: pd.DataFrame):
        metrics = self._predict_metrics(candidate_df)
        weights = np.array(
            [self.score_weights.get(col, 0.0) for col in self.target_cols],
            dtype=float,
        )
        score = metrics @ weights
        self._last_predicted_metrics = metrics
        self._last_predicted_score = score
        return np.maximum(score, 0.0)

    def _predict_metrics(self, candidate_df: pd.DataFrame):
        X = candidate_df[self.feature_cols]
        X_scaled = self.scaler.transform(X).astype(np.float32)
        pred = self.model.predict(
            X_scaled,
            batch_size=max(1, len(X_scaled)),
            verbose=0,
        )

        if self.target_log1p:
            pred = np.expm1(pred)

        return np.maximum(pred, 0.0)

    def _top_candidate_records(self, candidate_df, pred, top_n: int):
        order = np.argsort(pred)[:top_n]
        records = []
        metrics = getattr(self, "_last_predicted_metrics", None)

        for idx in order:
            row = candidate_df.iloc[int(idx)]
            record = {
                "kp": float(row["kp"]),
                "ki": float(row["ki"]),
                "kd": float(row["kd"]),
                "predicted_cost": float(pred[int(idx)]),
            }
            if metrics is not None:
                for col_idx, col in enumerate(self.target_cols):
                    record[f"predicted_{col}"] = float(metrics[int(idx), col_idx])
            records.append(record)

        return records


class DirectPolicyMlpChunkGenerator(RandomForestCostChunkGenerator):
    """
    Direct TensorFlow policy generator.

    It predicts one continuous gain triplet from the current state, avoiding
    candidate-grid scoring during online inference.
    """

    generator_id = "direct_policy_mlp_chunk_generator"
    payload_kind = PAYLOAD_KIND_GAIN

    def __init__(
        self,
        model_path,
        backend_name: str = "esp32",
        fallback_generator: ScheduleGenerator = None,
    ):
        payload = joblib.load(model_path)

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for DirectPolicyMlpChunkGenerator. "
                "Run the server in the tensorflow conda environment."
            ) from exc

        keras_model_path = payload.get("keras_model_path")
        if not keras_model_path:
            raise ValueError(
                f"Missing keras_model_path in direct policy MLP payload: {model_path}"
            )

        self.model = tf.keras.models.load_model(keras_model_path)
        self.scaler = payload["scaler"]
        self.feature_cols = list(payload["feature_cols"])
        self.output_cols = list(payload.get("output_cols", ["kp", "ki", "kd"]))
        self.output_mode = str(payload.get("output_mode", "absolute"))
        self.residual_scales = {
            str(key): float(value)
            for key, value in payload.get("residual_scales", {}).items()
        }
        self.db_reference_targets = [
            float(value)
            for value in payload.get(
                "db_reference_targets", [30.0, 50.0, 70.0, 85.0, 100.0]
            )
        ]
        self.gain_bounds = {
            str(key): (float(value[0]), float(value[1]))
            for key, value in payload.get("gain_bounds", {}).items()
        }
        self.model_path = str(model_path)
        self.keras_model_path = str(keras_model_path)
        self.backend_name = backend_name
        self.fallback_generator = fallback_generator

    def generate(
        self,
        state: Dict[str, Any],
        schedule_start_time: float,
        dt: float,
        horizon_steps: int,
    ) -> Dict[str, Any]:
        try:
            features = self._build_features(state, state.get("_history", []))
            x_scaled = self.scaler.transform(features).astype(np.float32)
            pred_norm = np.asarray(self.model(x_scaled, training=False)).reshape(-1)
            kp, ki, kd = self._decode_gain(pred_norm, features)
        except Exception:
            if self.fallback_generator is None:
                raise
            return self.fallback_generator.generate(
                state=state,
                schedule_start_time=schedule_start_time,
                dt=dt,
                horizon_steps=horizon_steps,
            )

        items = make_constant_gain_items(
            schedule_start_time=schedule_start_time,
            dt=dt,
            horizon_steps=horizon_steps,
            kp=kp,
            ki=ki,
            kd=kd,
            target=float(state["target"]),
        )

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
            reason="direct_policy_mlp_prediction",
            metadata={
                "backend": self.backend_name,
                "model_path": self.model_path,
                "target": float(state["target"]),
                "selected_gain": {"kp": kp, "ki": ki, "kd": kd},
                "raw_prediction": {
                    col: float(pred_norm[idx])
                    for idx, col in enumerate(self.output_cols)
                    if idx < len(pred_norm)
                },
            },
        )

    def _decode_gain(
        self,
        pred_norm: np.ndarray,
        features: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, float, float]:
        values = {}
        for idx, col in enumerate(self.output_cols):
            lo, hi = self.gain_bounds.get(col, (0.0, 1.0))
            if self.output_mode == "residual_db" and features is not None:
                base_col = f"db_base_{col}"
                base = float(features[base_col].iloc[0]) if base_col in features else 0.0
                scale = max(float(self.residual_scales.get(col, 1.0)), 1e-9)
                value = base + float(np.clip(pred_norm[idx], -1.0, 1.0)) * scale
                values[col] = float(np.clip(value, lo, hi))
            else:
                norm = float(np.clip(pred_norm[idx], 0.0, 1.0))
                values[col] = lo + norm * (hi - lo)
        return float(values["kp"]), float(values["ki"]), float(values["kd"])

    def _build_features(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        base = RandomForestCostChunkGenerator._base_features(self, state, history)
        self._add_direct_policy_features(base, history)
        df = pd.DataFrame([base])

        for feature in self.feature_cols:
            if feature not in df.columns:
                df[feature] = 0.0

        return df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _add_direct_policy_features(
        self,
        base: Dict[str, float],
        history: List[Dict[str, Any]],
    ):
        error = float(base.get("error", 0.0))
        target = float(base.get("target", 0.0))
        current = float(base.get("current", 0.0))
        error_derivative = float(base.get("error_derivative", 0.0))

        base["abs_error"] = abs(error)
        base["signed_error_ratio"] = error / max(abs(target), 1e-6)
        base["accel_demand"] = 1.0 if error > 0.0 else 0.0
        base["decel_demand"] = 1.0 if error < 0.0 else 0.0
        base["speed_ratio"] = current / max(abs(target), 1e-6)
        base["abs_error_derivative"] = abs(error_derivative)
        base.update(self._db_prior_gain_features(target))
        base.update(self._transition_context_features(target, history))
        base.update(self._db_reference_features(target))

    def _transition_context_features(
        self,
        target: float,
        history: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not history:
            return {
                "previous_target": float(target),
                "target_delta": 0.0,
                "abs_target_delta": 0.0,
                "target_direction": 0.0,
                "target_change_count": 0.0,
            }

        previous_segment_target = float(history[0].get("target", target))
        current_segment_target = previous_segment_target
        target_change_count = 0

        for item in history:
            item_target = float(item.get("target", current_segment_target))
            if abs(item_target - current_segment_target) > 1e-9:
                previous_segment_target = current_segment_target
                current_segment_target = item_target
                target_change_count += 1

        if abs(float(target) - current_segment_target) > 1e-9:
            previous_segment_target = current_segment_target
            current_segment_target = float(target)
            target_change_count += 1

        delta = current_segment_target - previous_segment_target
        return {
            "previous_target": float(previous_segment_target),
            "target_delta": float(delta),
            "abs_target_delta": float(abs(delta)),
            "target_direction": float(np.sign(delta)),
            "target_change_count": float(target_change_count),
        }

    def _db_prior_gain_features(self, target: float) -> Dict[str, float]:
        targets = sorted(float(value) for value in ESP32_REAL_PID_GAIN_DB.keys())
        if not targets:
            return {"db_base_kp": 0.0, "db_base_ki": 0.0, "db_base_kd": 0.0}

        target = float(target)
        if target in targets:
            gains = ESP32_REAL_PID_GAIN_DB[target]
            return {
                "db_base_kp": float(gains["kp"]),
                "db_base_ki": float(gains["ki"]),
                "db_base_kd": float(gains["kd"]),
            }

        if target <= targets[0]:
            gains = ESP32_REAL_PID_GAIN_DB[targets[0]]
            return {
                "db_base_kp": float(gains["kp"]),
                "db_base_ki": float(gains["ki"]),
                "db_base_kd": float(gains["kd"]),
            }
        if target >= targets[-1]:
            gains = ESP32_REAL_PID_GAIN_DB[targets[-1]]
            return {
                "db_base_kp": float(gains["kp"]),
                "db_base_ki": float(gains["ki"]),
                "db_base_kd": float(gains["kd"]),
            }

        for lower, upper in zip(targets[:-1], targets[1:]):
            if lower <= target <= upper:
                ratio = (target - lower) / max(upper - lower, 1e-9)
                lower_gain = ESP32_REAL_PID_GAIN_DB[lower]
                upper_gain = ESP32_REAL_PID_GAIN_DB[upper]
                return {
                    f"db_base_{key}": float(lower_gain[key])
                    + ratio * (float(upper_gain[key]) - float(lower_gain[key]))
                    for key in ["kp", "ki", "kd"]
                }

        nearest = min(targets, key=lambda value: abs(value - target))
        gains = ESP32_REAL_PID_GAIN_DB[nearest]
        return {
            "db_base_kp": float(gains["kp"]),
            "db_base_ki": float(gains["ki"]),
            "db_base_kd": float(gains["kd"]),
        }

    def _db_reference_features(self, target: float) -> Dict[str, float]:
        targets = sorted(self.db_reference_targets)
        if not targets:
            return {
                "target_db_nearest": 0.0,
                "target_db_lower": 0.0,
                "target_db_upper": 0.0,
                "target_db_nearest_distance": 0.0,
                "target_db_interval_width": 0.0,
                "target_db_alpha": 0.0,
                "target_db_is_interpolation": 0.0,
            }

        nearest = min(targets, key=lambda value: abs(value - target))
        lower_candidates = [value for value in targets if value <= target]
        upper_candidates = [value for value in targets if value >= target]
        lower = max(lower_candidates) if lower_candidates else targets[0]
        upper = min(upper_candidates) if upper_candidates else targets[-1]
        width = max(upper - lower, 0.0)
        alpha = 0.0 if width <= 1e-9 else (target - lower) / width

        return {
            "target_db_nearest": float(nearest),
            "target_db_lower": float(lower),
            "target_db_upper": float(upper),
            "target_db_nearest_distance": float(abs(target - nearest)),
            "target_db_interval_width": float(width),
            "target_db_alpha": float(np.clip(alpha, 0.0, 1.0)),
            "target_db_is_interpolation": float(
                width > 1e-9 and lower < target < upper
            ),
        }


class DirectPolicySequenceMlpChunkGenerator(DirectPolicyMlpChunkGenerator):
    """
    Direct TensorFlow sequence policy generator.

    It predicts one gain triplet for each schedule step, so the chunk can carry
    a time-varying PID gain sequence instead of repeating a constant gain.
    """

    generator_id = "direct_policy_sequence_mlp_chunk_generator"
    payload_kind = PAYLOAD_KIND_GAIN

    def generate(
        self,
        state: Dict[str, Any],
        schedule_start_time: float,
        dt: float,
        horizon_steps: int,
    ) -> Dict[str, Any]:
        try:
            features = self._build_features(state, state.get("_history", []))
            x_scaled = self.scaler.transform(features).astype(np.float32)
            pred_norm = np.asarray(self.model(x_scaled, training=False))
            pred_norm = pred_norm.reshape(-1, 3)

            if len(pred_norm) < int(horizon_steps):
                pad = np.repeat(
                    pred_norm[-1:, :],
                    int(horizon_steps) - len(pred_norm),
                    axis=0,
                )
                pred_norm = np.vstack([pred_norm, pad])
            pred_norm = pred_norm[: int(horizon_steps)]

            gain_sequence = [
                self._decode_gain(pred_norm[step_idx], features)
                for step_idx in range(int(horizon_steps))
            ]
        except Exception:
            if self.fallback_generator is None:
                raise
            return self.fallback_generator.generate(
                state=state,
                schedule_start_time=schedule_start_time,
                dt=dt,
                horizon_steps=horizon_steps,
            )

        items = []
        target = float(state["target"])
        for step_index, (kp, ki, kd) in enumerate(gain_sequence):
            items.append(
                make_schedule_item(
                    step_index=step_index,
                    control_time=float(schedule_start_time) + step_index * float(dt),
                    target=target,
                    kp=kp,
                    ki=ki,
                    kd=kd,
                )
            )

        first_gain = gain_sequence[0]
        last_gain = gain_sequence[-1]

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
            reason="direct_policy_sequence_mlp_prediction",
            metadata={
                "backend": self.backend_name,
                "model_path": self.model_path,
                "target": target,
                "sequence_length": int(len(gain_sequence)),
                "first_gain": {
                    "kp": float(first_gain[0]),
                    "ki": float(first_gain[1]),
                    "kd": float(first_gain[2]),
                },
                "last_gain": {
                    "kp": float(last_gain[0]),
                    "ki": float(last_gain[1]),
                    "kd": float(last_gain[2]),
                },
            },
        )


class DirectGainChunkPolicyGenerator(ScheduleGenerator):
    """
    Direct supervised gain-chunk policy.

    Unlike DirectPolicyMlpChunkGenerator, this model predicts the full
    time-varying gain sequence in one forward pass from recent observation
    history plus the current static state.
    """

    generator_id = "direct_gain_chunk_policy_generator"
    payload_kind = PAYLOAD_KIND_GAIN

    def __init__(
        self,
        model_path,
        backend_name: str = "esp32",
        fallback_generator: ScheduleGenerator = None,
    ):
        payload = joblib.load(model_path)

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for DirectGainChunkPolicyGenerator. "
                "Run the server in the tensorflow conda environment."
            ) from exc

        keras_model_path = payload.get("keras_model_path")
        if not keras_model_path:
            raise ValueError(
                f"Missing keras_model_path in gain chunk policy payload: {model_path}"
            )

        self.model = tf.keras.models.load_model(keras_model_path)
        self.seq_scaler = payload["seq_scaler"]
        self.static_scaler = payload["static_scaler"]
        self.obs_cols = list(payload["obs_cols"])
        self.static_feature_cols = list(payload["static_feature_cols"])
        self.gain_cols = list(payload.get("gain_cols", ["kp", "ki", "kd"]))
        self.gain_bounds = {
            str(key): (float(value[0]), float(value[1]))
            for key, value in payload.get(
                "gain_bounds",
                {"kp": (0.55, 1.45), "ki": (0.70, 2.50), "kd": (0.0, 0.12)},
            ).items()
        }
        self.obs_steps = int(payload["obs_steps"])
        self.trained_horizon_steps = int(payload["horizon_steps"])
        self.model_path = str(model_path)
        self.keras_model_path = str(keras_model_path)
        self.backend_name = backend_name
        self.fallback_generator = fallback_generator

    def generate(
        self,
        state: Dict[str, Any],
        schedule_start_time: float,
        dt: float,
        horizon_steps: int,
    ) -> Dict[str, Any]:
        try:
            x_seq, x_static = self._build_inputs(state, state.get("_history", []))
            pred_norm = np.asarray(self.model([x_seq, x_static], training=False))
            pred_norm = pred_norm.reshape(-1, len(self.gain_cols))
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

        items = []
        target = float(state["target"])
        for step_index, gain in enumerate(gain_sequence):
            items.append(
                make_schedule_item(
                    step_index=step_index,
                    control_time=float(schedule_start_time) + step_index * float(dt),
                    target=target,
                    kp=float(gain["kp"]),
                    ki=float(gain["ki"]),
                    kd=float(gain["kd"]),
                )
            )

        first_gain = gain_sequence[0]
        last_gain = gain_sequence[-1]
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
            reason="direct_gain_chunk_policy_prediction",
            metadata={
                "backend": self.backend_name,
                "model_path": self.model_path,
                "target": target,
                "sequence_length": int(len(gain_sequence)),
                "trained_horizon_steps": self.trained_horizon_steps,
                "first_gain": dict(first_gain),
                "last_gain": dict(last_gain),
            },
        )

    def _build_inputs(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        obs, static = self._build_raw_inputs_for_columns(
            state=state,
            history=history,
            obs_cols=self.obs_cols,
            static_feature_cols=self.static_feature_cols,
            obs_steps=self.obs_steps,
        )
        obs_scaled, static_scaled = self._scale_raw_inputs(
            obs=obs,
            static=static,
            seq_scaler=self.seq_scaler,
            static_scaler=self.static_scaler,
        )
        return obs_scaled, static_scaled

    def _build_raw_inputs_for_columns(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        obs_cols: List[str],
        static_feature_cols: List[str],
        obs_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        records = list(history)[-max(obs_steps - 1, 0) :] + [state]
        if not records:
            records = [state]
        while len(records) < obs_steps:
            records.insert(0, records[0])
        records = records[-obs_steps :]

        obs = np.zeros((1, obs_steps, len(obs_cols)), dtype=np.float32)
        for row_idx, item in enumerate(records):
            features = self._runtime_features(item, history)
            for col_idx, col in enumerate(obs_cols):
                obs[0, row_idx, col_idx] = float(features.get(col, 0.0))

        static_base = self._runtime_features(state, history)
        static = np.zeros((1, len(static_feature_cols)), dtype=np.float32)
        for col_idx, col in enumerate(static_feature_cols):
            key = col[len("state_") :] if col.startswith("state_") else col
            static[0, col_idx] = float(static_base.get(key, static_base.get(col, 0.0)))

        return obs.astype(np.float32), static.astype(np.float32)

    def _scale_raw_inputs(
        self,
        obs: np.ndarray,
        static: np.ndarray,
        seq_scaler,
        static_scaler,
    ) -> Tuple[np.ndarray, np.ndarray]:
        obs_flat = obs.reshape(obs.shape[0], obs.shape[1] * obs.shape[2])
        obs_scaled = seq_scaler.transform(obs_flat).reshape(obs.shape)
        static_scaled = static_scaler.transform(static).astype(np.float32)
        return obs_scaled.astype(np.float32), static_scaled

    def _runtime_features(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        target = float(state.get("target", 0.0))
        current = float(state.get("current", state.get("rpm", 0.0)))
        error = float(state.get("error", target - current))
        prev_error = float(state.get("prev_error", error))
        error_derivative = float(state.get("error_derivative", error - prev_error))
        pwm = float(state.get("pwm", state.get("output", 0.0)))
        raw_pwm = float(state.get("raw_pwm", pwm))
        kp = float(state.get("kp", state.get("current_kp", 0.0)))
        ki = float(state.get("ki", state.get("current_ki", 0.0)))
        kd = float(state.get("kd", state.get("current_kd", 0.0)))
        integral = float(state.get("integral", 0.0))
        timestamp = float(state.get("timestamp", state.get("control_time", 0.0)))
        control_time = float(state.get("control_time", timestamp))

        previous_target = target
        target_change_count = 0.0
        if history:
            previous_target = float(history[0].get("target", target))
            last_target = previous_target
            for item in history:
                item_target = float(item.get("target", last_target))
                if abs(item_target - last_target) > 1e-9:
                    previous_target = last_target
                    last_target = item_target
                    target_change_count += 1.0
            if abs(target - last_target) > 1e-9:
                previous_target = last_target
                target_change_count += 1.0

        target_delta = target - previous_target
        return {
            "target": target,
            "current": current,
            "error": error,
            "error_derivative": error_derivative,
            "pwm": pwm,
            "prev_pwm": float(state.get("prev_pwm", pwm)),
            "raw_pwm": raw_pwm,
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "kp_scale": float(state.get("kp_scale", 1.0)),
            "ki_scale": float(state.get("ki_scale", 1.0)),
            "integral": integral,
            "pid_p_term": float(state.get("pid_p_term", kp * error)),
            "pid_i_term": float(state.get("pid_i_term", ki * integral)),
            "pid_d_term": float(state.get("pid_d_term", kd * error_derivative)),
            "time_since_start": control_time,
            "time_since_target_change": float(
                state.get("time_since_target_change", 0.0)
            ),
            "error_ratio": error / max(abs(target), 1e-6),
            "pwm_ratio": pwm / 140.0,
            "abs_error": abs(error),
            "signed_error_ratio": error / max(abs(target), 1e-6),
            "accel_demand": 1.0 if error > 0.0 else 0.0,
            "decel_demand": 1.0 if error < 0.0 else 0.0,
            "speed_ratio": current / max(abs(target), 1e-6),
            "abs_error_derivative": abs(error_derivative),
            "previous_target": previous_target,
            "target_delta": target_delta,
            "abs_target_delta": abs(target_delta),
            "target_direction": float(np.sign(target_delta)),
            "target_change_count": target_change_count,
        }

    def _decode_gain_sequence(
        self,
        pred_norm: np.ndarray,
        horizon_steps: int,
    ) -> List[Dict[str, float]]:
        pred_norm = np.clip(pred_norm, 0.0, 1.0)
        if len(pred_norm) < horizon_steps:
            pad = np.repeat(pred_norm[-1:, :], horizon_steps - len(pred_norm), axis=0)
            pred_norm = np.vstack([pred_norm, pad])
        pred_norm = pred_norm[:horizon_steps]

        sequence = []
        for row in pred_norm:
            item = {}
            for idx, col in enumerate(self.gain_cols):
                lo, hi = self.gain_bounds.get(col, (0.0, 1.0))
                item[col] = float(lo + float(row[idx]) * (hi - lo))
            for col in ["kp", "ki", "kd"]:
                item.setdefault(col, 0.0)
            sequence.append(item)
        return sequence


class DiffusionUnetGainChunkGenerator(DirectGainChunkPolicyGenerator):
    """
    Conditional diffusion U-Net gain-chunk generator.

    The model denoises a 20-step gain sequence in normalized gain space and
    publishes the resulting time-varying PID gain chunk.
    """

    generator_id = "diffusion_unet_gain_chunk_generator"
    payload_kind = PAYLOAD_KIND_GAIN

    def __init__(
        self,
        model_path,
        backend_name: str = "esp32",
        fallback_generator: ScheduleGenerator = None,
        ddim_steps: int = 20,
        sample_count: int = 1,
        cost_surrogate_model_path: Optional[str] = None,
        cost_selection_metric: str = "label_cost",
        response_surrogate_model_path: Optional[str] = None,
        response_score_iae_weight: float = 1.0,
        response_score_pwm_weight: float = 0.01,
        response_score_pwm_variation_weight: float = 0.02,
        response_score_saturation_weight: float = 50.0,
        response_score_mode: str = "iae",
        response_score_settling_weight: float = 2.0,
        response_score_overshoot_weight: float = 4.0,
        response_score_final_error_weight: float = 3.0,
        response_score_oscillation_weight: float = 0.2,
        diffusion_candidate_mode: str = "sample",
        diffusion_deterministic_seed: int = 0,
        two_phase_boost_scale: float = 1.0,
        two_phase_ki_decay_scale: float = 1.0,
        two_phase_kd_boost_scale: float = 1.0,
    ):
        payload = joblib.load(model_path)

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for DiffusionUnetGainChunkGenerator. "
                "Run the server in the tensorflow conda environment."
            ) from exc

        try:
            from types import SimpleNamespace
            from train_diffusion_gain_chunk_unet import (
                build_diffusion_constants,
                build_unet,
                ddim_sample,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import diffusion U-Net helpers. "
                "Run from the project root or ensure src is on PYTHONPATH."
            ) from exc

        arch = payload["architecture"]
        self.tf = tf
        self.ddim_sample_fn = ddim_sample
        self.diffusion_args = SimpleNamespace(
            condition_dim=int(arch["condition_dim"]),
            time_embed_dim=int(arch["time_embed_dim"]),
            base_filters=int(arch["base_filters"]),
            dropout=float(arch["dropout"]),
            norm=str(arch.get("norm", "batch")),
            condition_mode=str(arch.get("condition_mode", "avg")),
            diffusion_steps=int(payload["diffusion_steps"]),
            ddim_steps=int(ddim_steps),
        )
        self.obs_cols = list(payload["obs_cols"])
        self.static_feature_cols = list(payload["static_feature_cols"])
        self.gain_cols = list(payload.get("gain_cols", ["kp", "ki", "kd"]))
        self.gain_bounds = {
            str(key): (float(value[0]), float(value[1]))
            for key, value in payload.get(
                "gain_bounds",
                {"kp": (0.55, 1.45), "ki": (0.70, 2.50), "kd": (0.0, 0.12)},
            ).items()
        }
        self.obs_steps = int(payload["obs_steps"])
        self.trained_horizon_steps = int(payload["horizon_steps"])
        self.seq_scaler = payload["seq_scaler"]
        self.static_scaler = payload["static_scaler"]
        self.sample_count = int(sample_count)
        self.cost_surrogate_model_path = (
            str(cost_surrogate_model_path) if cost_surrogate_model_path else ""
        )
        self.cost_selection_metric = str(cost_selection_metric)
        self.cost_payload = None
        self.cost_model = None
        self.response_surrogate_model_path = (
            str(response_surrogate_model_path) if response_surrogate_model_path else ""
        )
        self.response_payload = None
        self.response_model = None
        self.response_score_iae_weight = float(response_score_iae_weight)
        self.response_score_pwm_weight = float(response_score_pwm_weight)
        self.response_score_pwm_variation_weight = float(response_score_pwm_variation_weight)
        self.response_score_saturation_weight = float(response_score_saturation_weight)
        self.response_score_mode = str(response_score_mode)
        self.response_score_settling_weight = float(response_score_settling_weight)
        self.response_score_overshoot_weight = float(response_score_overshoot_weight)
        self.response_score_final_error_weight = float(response_score_final_error_weight)
        self.response_score_oscillation_weight = float(response_score_oscillation_weight)
        self.diffusion_candidate_mode = str(diffusion_candidate_mode)
        self.diffusion_deterministic_seed = int(diffusion_deterministic_seed)
        self.two_phase_boost_scale = float(two_phase_boost_scale)
        self.two_phase_ki_decay_scale = float(two_phase_ki_decay_scale)
        self.two_phase_kd_boost_scale = float(two_phase_kd_boost_scale)
        self.model_path = str(model_path)
        self.weights_path = str(payload["weights_path"])
        self.backend_name = backend_name
        self.fallback_generator = fallback_generator
        self.constants = build_diffusion_constants(tf, int(payload["diffusion_steps"]))
        self.model = build_unet(
            tf,
            obs_steps=self.obs_steps,
            obs_dim=len(self.obs_cols),
            static_dim=len(self.static_feature_cols),
            horizon_steps=self.trained_horizon_steps,
            args=self.diffusion_args,
        )
        self.model.load_weights(self.weights_path)
        if self.cost_surrogate_model_path:
            self.cost_payload = joblib.load(self.cost_surrogate_model_path)
            keras_model_path = self.cost_payload.get("keras_model_path")
            if not keras_model_path:
                raise ValueError(
                    "Missing keras_model_path in gain chunk cost surrogate payload: "
                    f"{self.cost_surrogate_model_path}"
                )
            self.cost_model = tf.keras.models.load_model(keras_model_path)
            target_cols = list(self.cost_payload.get("target_cols", []))
            if self.cost_selection_metric not in target_cols:
                raise ValueError(
                    f"cost_selection_metric={self.cost_selection_metric!r} not in "
                    f"cost surrogate target_cols={target_cols}"
                )
        if self.response_surrogate_model_path:
            self.response_payload = joblib.load(self.response_surrogate_model_path)
            weights_path = self.response_payload.get("weights_path")
            keras_model_path = self.response_payload.get("keras_model_path")
            if weights_path:
                try:
                    from train_response_surrogate import build_response_model
                except ImportError as exc:
                    raise ImportError(
                        "Could not import response surrogate builder. "
                        "Run from the project root or ensure src is on PYTHONPATH."
                    ) from exc
                self.response_model = build_response_model(
                    tf,
                    past_steps=int(self.response_payload["past_steps"]),
                    past_dim=len(self.response_payload["past_cols"]),
                    future_steps=int(self.response_payload["future_steps"]),
                    control_dim=len(self.response_payload["future_control_cols"]),
                    response_dim=len(self.response_payload["response_cols"]),
                    dropout=float(self.response_payload.get("dropout", 0.05)),
                )
                self.response_model.load_weights(weights_path)
            elif keras_model_path:
                self.response_model = tf.keras.models.load_model(
                    keras_model_path,
                    compile=False,
                )
            else:
                raise ValueError(
                    "Missing weights_path/keras_model_path in response surrogate payload: "
                    f"{self.response_surrogate_model_path}"
                )

    def generate(
        self,
        state: Dict[str, Any],
        schedule_start_time: float,
        dt: float,
        horizon_steps: int,
    ) -> Dict[str, Any]:
        selected_index = None
        predicted_cost_metrics = None
        predicted_response_metrics = None
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
            if self.diffusion_deterministic_seed > 0:
                self.tf.random.set_seed(self.diffusion_deterministic_seed)
            sample_diff = self.ddim_sample_fn(
                self.tf,
                self.model,
                x_seq,
                x_static,
                self.diffusion_args,
                self.constants,
                sample_count=self.sample_count,
            )
            if (
                self.response_model is not None
                and self.diffusion_candidate_mode == "two_phase"
            ):
                base_diff = np.mean(sample_diff[0], axis=0)
                base_norm = np.clip((base_diff + 1.0) / 2.0, 0.0, 1.0)
                candidate_norm, candidate_names = self._build_two_phase_candidates(
                    base_norm=base_norm,
                    state=state,
                )
                selected_index, predicted_response_metrics = (
                    self._select_candidate_by_response_surrogate(
                        state=state,
                        history=history,
                        candidate_norm=candidate_norm,
                    )
                )
                pred_norm = candidate_norm[selected_index]
                predicted_response_metrics["candidate_mode"] = "two_phase"
                predicted_response_metrics["selected_candidate_name"] = candidate_names[
                    selected_index
                ]
                pred_diff = pred_norm * 2.0 - 1.0
            elif self.response_model is not None and sample_diff.shape[1] > 1:
                candidate_norm = np.clip((sample_diff[0] + 1.0) / 2.0, 0.0, 1.0)
                selected_index, predicted_response_metrics = (
                    self._select_candidate_by_response_surrogate(
                        state=state,
                        history=history,
                        candidate_norm=candidate_norm,
                    )
                )
                pred_diff = sample_diff[0, selected_index]
            elif self.cost_model is not None and sample_diff.shape[1] > 1:
                candidate_norm = np.clip((sample_diff[0] + 1.0) / 2.0, 0.0, 1.0)
                selected_index, predicted_cost_metrics = self._select_candidate_by_cost(
                    state=state,
                    history=history,
                    candidate_norm=candidate_norm,
                )
                pred_diff = sample_diff[0, selected_index]
            else:
                # Default behavior remains sample-mean for backward compatibility.
                pred_diff = np.mean(sample_diff[0], axis=0)
            pred_norm = np.clip((pred_diff + 1.0) / 2.0, 0.0, 1.0)
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

        items = []
        target = float(state["target"])
        for step_index, gain in enumerate(gain_sequence):
            items.append(
                make_schedule_item(
                    step_index=step_index,
                    control_time=float(schedule_start_time) + step_index * float(dt),
                    target=target,
                    kp=float(gain["kp"]),
                    ki=float(gain["ki"]),
                    kd=float(gain["kd"]),
                )
            )

        first_gain = gain_sequence[0]
        last_gain = gain_sequence[-1]
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
            reason="diffusion_unet_ddim_gain_chunk_prediction",
            metadata={
                "backend": self.backend_name,
                "model_path": self.model_path,
                "weights_path": self.weights_path,
                "target": target,
                "sequence_length": int(len(gain_sequence)),
                "trained_horizon_steps": self.trained_horizon_steps,
                "ddim_steps": int(self.diffusion_args.ddim_steps),
                "sample_count": int(self.sample_count),
                "cost_surrogate_model_path": self.cost_surrogate_model_path,
                "cost_selection_metric": self.cost_selection_metric,
                "response_surrogate_model_path": self.response_surrogate_model_path,
                "response_score_weights": {
                    "mode": self.response_score_mode,
                    "iae": self.response_score_iae_weight,
                    "settling": self.response_score_settling_weight,
                    "overshoot": self.response_score_overshoot_weight,
                    "final_error": self.response_score_final_error_weight,
                    "oscillation": self.response_score_oscillation_weight,
                    "pwm": self.response_score_pwm_weight,
                    "pwm_variation": self.response_score_pwm_variation_weight,
                    "saturation": self.response_score_saturation_weight,
                },
                "diffusion_candidate_mode": self.diffusion_candidate_mode,
                "diffusion_deterministic_seed": self.diffusion_deterministic_seed,
                "two_phase_params": {
                    "boost_scale": self.two_phase_boost_scale,
                    "ki_decay_scale": self.two_phase_ki_decay_scale,
                    "kd_boost_scale": self.two_phase_kd_boost_scale,
                },
                "selected_sample_index": (
                    None if selected_index is None else int(selected_index)
                ),
                "predicted_cost_metrics": predicted_cost_metrics,
                "predicted_response_metrics": predicted_response_metrics,
                "first_gain": dict(first_gain),
                "last_gain": dict(last_gain),
            },
        )

    def _norm_to_actual_gain_array(self, gain_norm: np.ndarray) -> np.ndarray:
        gain_norm = np.asarray(gain_norm, dtype=np.float32)
        actual = np.zeros_like(gain_norm, dtype=np.float32)
        for idx, col in enumerate(self.gain_cols):
            lo, hi = self.gain_bounds.get(col, (0.0, 1.0))
            actual[..., idx] = lo + gain_norm[..., idx] * (hi - lo)
        return actual

    def _actual_to_norm_gain_array(self, gain_actual: np.ndarray) -> np.ndarray:
        gain_actual = np.asarray(gain_actual, dtype=np.float32)
        norm = np.zeros_like(gain_actual, dtype=np.float32)
        for idx, col in enumerate(self.gain_cols):
            lo, hi = self.gain_bounds.get(col, (0.0, 1.0))
            denom = max(hi - lo, 1e-12)
            norm[..., idx] = (gain_actual[..., idx] - lo) / denom
        return np.clip(norm, 0.0, 1.0)

    def _clip_gain_array(self, gain_actual: np.ndarray) -> np.ndarray:
        clipped = np.asarray(gain_actual, dtype=np.float32).copy()
        for idx, col in enumerate(self.gain_cols):
            lo, hi = self.gain_bounds.get(col, (0.0, 1.0))
            clipped[..., idx] = np.clip(clipped[..., idx], lo, hi)
        return clipped

    def _smooth_gain_array(self, gain_actual: np.ndarray, passes: int = 1) -> np.ndarray:
        smoothed = np.asarray(gain_actual, dtype=np.float32).copy()
        if smoothed.shape[0] < 3:
            return smoothed
        for _ in range(max(0, int(passes))):
            prev = smoothed.copy()
            smoothed[1:-1] = 0.25 * prev[:-2] + 0.5 * prev[1:-1] + 0.25 * prev[2:]
        return smoothed

    def _build_two_phase_candidates(
        self,
        base_norm: np.ndarray,
        state: Dict[str, Any],
    ) -> Tuple[np.ndarray, List[str]]:
        base = self._norm_to_actual_gain_array(base_norm)
        horizon = int(base.shape[0])
        if horizon <= 0:
            return base_norm[None, :, :], ["base"]

        half = max(1, horizon // 2)
        phase = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
        late = np.clip((phase - 0.5) / 0.5, 0.0, 1.0)
        early_mask = (np.arange(horizon) < half).astype(np.float32)

        current = float(state.get("current", state.get("rpm", 0.0)))
        target = float(state.get("target", current))
        error_mag = abs(target - current)
        # Large transitions benefit from a slightly stronger approach phase,
        # while small corrections stay close to the learned diffusion chunk.
        transition_scale = float(np.clip(error_mag / 25.0, 0.35, 1.0))

        boost_scale = max(0.0, self.two_phase_boost_scale)
        ki_decay_scale = max(0.0, self.two_phase_ki_decay_scale)
        kd_boost_scale = max(0.0, self.two_phase_kd_boost_scale)
        specs = [
            ("base", 1.00, 1.00, 1.00, 0.00, 1),
            ("boost10_ki_decay", 1.10, 1.12, 1.08, 0.22, 1),
            ("strong_boost10_strong_ki_decay", 1.18, 1.22, 1.15, 0.35, 1),
            ("boost8_kd_brake", 1.14, 1.10, 1.28, 0.30, 2),
            ("smooth_sigmoid_boost_decay", 1.12, 1.16, 1.18, 0.28, 3),
            ("conservative_settle", 1.04, 1.06, 1.22, 0.40, 2),
        ]
        candidates = []
        names = []
        for name, kp_boost, ki_boost, kd_late_boost, ki_decay, smooth_passes in specs:
            cand = base.copy()
            early_boost = early_mask * transition_scale
            if name == "smooth_sigmoid_boost_decay":
                early_boost = (1.0 - late) * transition_scale

            kp_factor = 1.0 + (kp_boost - 1.0) * early_boost * boost_scale
            ki_factor = 1.0 + (ki_boost - 1.0) * early_boost * boost_scale
            ki_factor *= 1.0 - np.clip(ki_decay * ki_decay_scale, 0.0, 0.85) * late
            kd_factor = 1.0 + (kd_late_boost - 1.0) * late * kd_boost_scale

            if "kp" in self.gain_cols:
                cand[:, self.gain_cols.index("kp")] *= kp_factor
            if "ki" in self.gain_cols:
                cand[:, self.gain_cols.index("ki")] *= ki_factor
            if "kd" in self.gain_cols:
                cand[:, self.gain_cols.index("kd")] *= kd_factor

            cand = self._smooth_gain_array(cand, passes=smooth_passes)
            cand = self._clip_gain_array(cand)
            candidates.append(cand)
            names.append(name)

        return self._actual_to_norm_gain_array(np.stack(candidates, axis=0)), names

    def _select_candidate_by_cost(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        candidate_norm: np.ndarray,
    ) -> Tuple[int, Dict[str, float]]:
        cost_obs_cols = list(self.cost_payload["obs_cols"])
        cost_static_cols = list(self.cost_payload["static_feature_cols"])
        cost_obs_steps = int(self.cost_payload["obs_steps"])
        raw_seq, raw_static = self._build_raw_inputs_for_columns(
            state=state,
            history=history,
            obs_cols=cost_obs_cols,
            static_feature_cols=cost_static_cols,
            obs_steps=cost_obs_steps,
        )
        x_seq, x_static = self._scale_raw_inputs(
            obs=raw_seq,
            static=raw_static,
            seq_scaler=self.cost_payload["seq_scaler"],
            static_scaler=self.cost_payload["static_scaler"],
        )
        sample_count = int(candidate_norm.shape[0])
        x_seq = np.repeat(x_seq, sample_count, axis=0)
        x_static = np.repeat(x_static, sample_count, axis=0)

        gain_flat = candidate_norm.reshape((sample_count, -1))
        gain_scaled = self.cost_payload["gain_scaler"].transform(gain_flat)
        gain_scaled = gain_scaled.reshape(candidate_norm.shape).astype(np.float32)

        pred_scaled = self.cost_model(
            [x_seq.astype(np.float32), x_static.astype(np.float32), gain_scaled],
            training=False,
        )
        pred = self.cost_payload["target_scaler"].inverse_transform(np.asarray(pred_scaled))
        target_cols = list(self.cost_payload["target_cols"])
        metric_idx = target_cols.index(self.cost_selection_metric)
        selected_index = int(np.argmin(pred[:, metric_idx]))
        selected_metrics = {
            col: float(pred[selected_index, idx])
            for idx, col in enumerate(target_cols)
        }
        selected_metrics["candidate_count"] = float(sample_count)
        return selected_index, selected_metrics

    def _build_response_past_raw(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> np.ndarray:
        past_cols = list(self.response_payload["past_cols"])
        past_steps = int(self.response_payload["past_steps"])
        records = list(history)[-max(past_steps - 1, 0) :] + [state]
        if not records:
            records = [state]
        while len(records) < past_steps:
            records.insert(0, records[0])
        records = records[-past_steps:]
        x = np.zeros((1, past_steps, len(past_cols)), dtype=np.float32)
        for row_idx, item in enumerate(records):
            features = self._runtime_features(item, history)
            for col_idx, col in enumerate(past_cols):
                if col == "schedule_chunk_index":
                    value = item.get("schedule_chunk_index", -1)
                elif col == "schedule_fallback_used":
                    value = item.get("schedule_fallback_used", 0.0)
                elif col == "current":
                    value = item.get("current", item.get("rpm", features.get("current", 0.0)))
                else:
                    value = features.get(col, item.get(col, 0.0))
                try:
                    x[0, row_idx, col_idx] = float(value)
                except (TypeError, ValueError):
                    x[0, row_idx, col_idx] = 0.0
        return x

    def _select_candidate_by_response_surrogate(
        self,
        state: Dict[str, Any],
        history: List[Dict[str, Any]],
        candidate_norm: np.ndarray,
    ) -> Tuple[int, Dict[str, float]]:
        sample_count = int(candidate_norm.shape[0])
        horizon_steps = int(self.response_payload["future_steps"])
        future_control_cols = list(self.response_payload["future_control_cols"])
        response_cols = list(self.response_payload["response_cols"])

        candidate_gains = self._norm_to_actual_gain_array(candidate_norm)
        if candidate_gains.shape[1] < horizon_steps:
            pad = np.repeat(candidate_gains[:, -1:, :], horizon_steps - candidate_gains.shape[1], axis=1)
            candidate_gains = np.concatenate([candidate_gains, pad], axis=1)
        candidate_gains = candidate_gains[:, :horizon_steps, :]

        future_control = np.zeros(
            (sample_count, horizon_steps, len(future_control_cols)),
            dtype=np.float32,
        )
        target = float(state["target"])
        for col_idx, col in enumerate(future_control_cols):
            if col == "target":
                future_control[:, :, col_idx] = target
            elif col in self.gain_cols:
                gain_idx = self.gain_cols.index(col)
                future_control[:, :, col_idx] = candidate_gains[:, :, gain_idx]

        past_raw = self._build_response_past_raw(state, history)
        past_raw = np.repeat(past_raw, sample_count, axis=0)
        past_scaled = self.response_payload["past_scaler"].transform(
            past_raw.reshape((sample_count, -1))
        ).reshape(past_raw.shape)
        control_scaled = self.response_payload["control_scaler"].transform(
            future_control.reshape((sample_count, -1))
        ).reshape(future_control.shape)

        pred_scaled = self.response_model(
            [past_scaled.astype(np.float32), control_scaled.astype(np.float32)],
            training=False,
        )
        response_scaler = self.response_payload["response_scaler"]
        pred_response = response_scaler.inverse_transform(
            np.asarray(pred_scaled).reshape((sample_count, -1))
        ).reshape((sample_count, horizon_steps, len(response_cols)))

        rpm_idx = response_cols.index("current") if "current" in response_cols else 0
        pwm_idx = response_cols.index("pwm") if "pwm" in response_cols else 1
        rpm = pred_response[:, :, rpm_idx]
        pwm = pred_response[:, :, pwm_idx]
        target_seq = future_control[:, :, future_control_cols.index("target")]
        dt = 0.1
        iae = np.sum(np.abs(target_seq - rpm), axis=1) * dt
        abs_error = np.abs(target_seq - rpm)
        final_error = abs_error[:, -1]
        current = float(state.get("current", state.get("rpm", 0.0)))
        direction = np.sign(float(state.get("target", current)) - current)
        if abs(direction) < 1e-9:
            direction = 1.0
        directional_overshoot = np.maximum(direction * (rpm - target_seq), 0.0)
        overshoot = np.max(directional_overshoot, axis=1)
        split_idx = max(1, horizon_steps // 2)
        early_error_area = np.sum(abs_error[:, :split_idx], axis=1) * dt
        late_error_area = np.sum(abs_error[:, split_idx:], axis=1) * dt
        late_overshoot = np.max(directional_overshoot[:, split_idx:], axis=1)
        tolerance = max(abs(float(state.get("target", 0.0))) * 0.02, 1.0)
        settling_time = np.full(sample_count, (horizon_steps + 1) * dt, dtype=np.float32)
        for row_idx in range(sample_count):
            for step_idx in range(horizon_steps):
                if np.all(abs_error[row_idx, step_idx:] <= tolerance):
                    settling_time[row_idx] = (step_idx + 1) * dt
                    break
        rpm_oscillation = np.sum(np.abs(np.diff(rpm, axis=1)), axis=1)
        late_rpm_oscillation = np.sum(np.abs(np.diff(rpm[:, split_idx:], axis=1)), axis=1)
        pwm_mean = np.mean(np.abs(pwm), axis=1)
        pwm_variation = np.sum(np.abs(np.diff(pwm, axis=1)), axis=1)
        saturation = np.mean(np.maximum(0.0, pwm - 140.0), axis=1)
        if self.response_score_mode == "settling_overshoot":
            score = (
                self.response_score_settling_weight * settling_time
                + self.response_score_overshoot_weight * late_overshoot
                + self.response_score_final_error_weight * final_error
                + self.response_score_iae_weight
                * (0.65 * early_error_area + 1.35 * late_error_area)
                + self.response_score_oscillation_weight * late_rpm_oscillation
                + self.response_score_pwm_weight * pwm_mean
                + self.response_score_pwm_variation_weight * pwm_variation
                + self.response_score_saturation_weight * saturation
            )
        else:
            score = (
                self.response_score_iae_weight * iae
                + self.response_score_pwm_weight * pwm_mean
                + self.response_score_pwm_variation_weight * pwm_variation
                + self.response_score_saturation_weight * saturation
            )
        selected_index = int(np.argmin(score))
        return selected_index, {
            "candidate_count": float(sample_count),
            "score_mode": self.response_score_mode,
            "selected_score": float(score[selected_index]),
            "selected_predicted_iae": float(iae[selected_index]),
            "selected_predicted_settling_time": float(settling_time[selected_index]),
            "selected_predicted_overshoot": float(overshoot[selected_index]),
            "selected_predicted_late_overshoot": float(late_overshoot[selected_index]),
            "selected_predicted_final_error": float(final_error[selected_index]),
            "selected_predicted_rpm_oscillation": float(rpm_oscillation[selected_index]),
            "selected_predicted_late_rpm_oscillation": float(
                late_rpm_oscillation[selected_index]
            ),
            "selected_predicted_early_error_area": float(early_error_area[selected_index]),
            "selected_predicted_late_error_area": float(late_error_area[selected_index]),
            "selected_predicted_pwm_mean": float(pwm_mean[selected_index]),
            "selected_predicted_pwm_variation": float(pwm_variation[selected_index]),
            "selected_predicted_saturation_penalty": float(saturation[selected_index]),
            "min_predicted_iae": float(np.min(iae)),
            "max_predicted_iae": float(np.max(iae)),
            "mean_predicted_iae": float(np.mean(iae)),
            "min_predicted_settling_time": float(np.min(settling_time)),
            "min_predicted_overshoot": float(np.min(overshoot)),
            "max_predicted_overshoot": float(np.max(overshoot)),
            "min_predicted_late_overshoot": float(np.min(late_overshoot)),
            "candidate_scores": [float(value) for value in score],
        }
