import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from schedule_schema import validate_schedule_chunk_message


@dataclass
class ScheduleLookupResult:
    found: bool
    item: Optional[Dict[str, Any]] = None
    chunk: Optional[Dict[str, Any]] = None
    reason: str = "not_found"
    chunk_index: Optional[int] = None
    discarded_steps: int = 0
    chunk_age_sec: Optional[float] = None
    accepted_at_wall_time: Optional[float] = None
    accepted_at_control_time: Optional[float] = None


class ScheduleBuffer:
    """
    Delay-aware schedule chunk buffer.

    The buffer is intentionally payload-agnostic. A chunk item may contain
    gains, PWM, feedforward PWM, or any later optimized control feature.
    """

    def __init__(
        self,
        run_id: str,
        device_id: str,
        max_chunks: int = 8,
        replace_same_chunk_id: bool = True,
    ):
        self.run_id = run_id
        self.device_id = device_id
        self.max_chunks = int(max_chunks)
        self.replace_same_chunk_id = bool(replace_same_chunk_id)
        self._chunks: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def clear(self):
        self._chunks.clear()

    def add_chunk(
        self,
        chunk: Dict[str, Any],
        now: Optional[float] = None,
        accepted_control_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        if now is None:
            now = time.time()

        valid, reason = validate_schedule_chunk_message(
            chunk,
            current_run_id=self.run_id,
            current_device_id=self.device_id,
            now=now,
        )

        if not valid:
            return False, reason

        chunk_id = str(chunk["chunk_id"])

        if self.replace_same_chunk_id:
            self._chunks = [
                existing
                for existing in self._chunks
                if str(existing.get("chunk_id")) != chunk_id
            ]

        stored_chunk = dict(chunk)
        stored_chunk["_accepted_at_wall_time"] = float(now)

        if accepted_control_time is not None:
            stored_chunk["_accepted_at_control_time"] = float(accepted_control_time)

        self._chunks.append(stored_chunk)
        self._sort_and_trim()
        return True, "accepted"

    def prune(self, control_time: Optional[float] = None, now: Optional[float] = None) -> int:
        if now is None:
            now = time.time()

        before_count = len(self._chunks)
        kept = []

        for chunk in self._chunks:
            if now > float(chunk["valid_until"]):
                continue

            if control_time is not None and self._chunk_end_time(chunk) <= float(control_time):
                continue

            kept.append(chunk)

        self._chunks = kept
        return before_count - len(self._chunks)

    def get_item(
        self,
        control_time: float,
        now: Optional[float] = None,
        payload_kind: Optional[str] = None,
    ) -> ScheduleLookupResult:
        if now is None:
            now = time.time()

        self.prune(control_time=control_time, now=now)

        candidates = []
        for chunk in self._chunks:
            if payload_kind is not None and chunk.get("payload_kind") != payload_kind:
                continue

            index = self._index_for_control_time(chunk, control_time)
            if index is None:
                continue

            candidates.append((chunk, index))

        if not candidates:
            return ScheduleLookupResult(found=False, reason="no_valid_chunk")

        chunk, index = self._select_best_candidate(candidates)
        item = dict(chunk["items"][index])

        discarded_steps = max(
            0,
            int(math.floor((float(control_time) - float(chunk["schedule_start_time"])) / float(chunk["dt"]))),
        )

        return ScheduleLookupResult(
            found=True,
            item=item,
            chunk=chunk,
            reason="valid",
            chunk_index=index,
            discarded_steps=discarded_steps,
            chunk_age_sec=now - float(chunk["generated_at"]),
            accepted_at_wall_time=chunk.get("_accepted_at_wall_time"),
            accepted_at_control_time=chunk.get("_accepted_at_control_time"),
        )

    def get_item_naive(
        self,
        control_time: float,
        now: Optional[float] = None,
        payload_kind: Optional[str] = None,
    ) -> ScheduleLookupResult:
        """
        Naive schedule application.

        This intentionally ignores the schedule_start_time embedded by the
        server. The newest chunk starts from index 0 when it arrives at the
        local controller. This is useful as a baseline against delay-aware
        time-indexed application.
        """

        if now is None:
            now = time.time()

        self.prune(control_time=None, now=now)

        candidates = []
        for chunk in self._chunks:
            if payload_kind is not None and chunk.get("payload_kind") != payload_kind:
                continue

            if "_accepted_at_control_time" not in chunk:
                continue

            accepted_control_time = float(chunk["_accepted_at_control_time"])
            if control_time < accepted_control_time:
                continue

            dt = float(chunk["dt"])
            horizon_steps = int(chunk["horizon_steps"])
            index = int(math.floor((float(control_time) - accepted_control_time) / dt))

            if 0 <= index < horizon_steps:
                candidates.append((chunk, index))

        if not candidates:
            return ScheduleLookupResult(found=False, reason="no_valid_chunk")

        chunk, index = self._select_best_candidate(candidates)
        item = dict(chunk["items"][index])

        delay_aware_index = self._index_for_control_time(chunk, control_time)
        if delay_aware_index is None:
            delay_aware_index = 0

        return ScheduleLookupResult(
            found=True,
            item=item,
            chunk=chunk,
            reason="valid_naive",
            chunk_index=index,
            discarded_steps=max(0, int(delay_aware_index)),
            chunk_age_sec=now - float(chunk["generated_at"]),
            accepted_at_wall_time=chunk.get("_accepted_at_wall_time"),
            accepted_at_control_time=chunk.get("_accepted_at_control_time"),
        )

    def snapshot(self) -> List[Dict[str, Any]]:
        return [dict(chunk) for chunk in self._chunks]

    def _sort_and_trim(self):
        self._chunks.sort(
            key=lambda chunk: (
                float(chunk.get("generated_at", 0.0)),
                int(chunk.get("source_seq", -1)),
            )
        )

        if len(self._chunks) > self.max_chunks:
            self._chunks = self._chunks[-self.max_chunks :]

    def _chunk_end_time(self, chunk: Dict[str, Any]) -> float:
        return float(chunk["schedule_start_time"]) + float(chunk["dt"]) * int(chunk["horizon_steps"])

    def _index_for_control_time(
        self,
        chunk: Dict[str, Any],
        control_time: float,
    ) -> Optional[int]:
        start_time = float(chunk["schedule_start_time"])
        dt = float(chunk["dt"])
        horizon_steps = int(chunk["horizon_steps"])

        if control_time < start_time:
            return None

        index = int(math.floor((float(control_time) - start_time) / dt))

        if index < 0 or index >= horizon_steps:
            return None

        return index

    def _select_best_candidate(
        self,
        candidates: List[Tuple[Dict[str, Any], int]],
    ) -> Tuple[Dict[str, Any], int]:
        # Prefer the newest generated schedule. If tied, prefer the newest source state.
        return max(
            candidates,
            key=lambda pair: (
                float(pair[0].get("generated_at", 0.0)),
                int(pair[0].get("source_seq", -1)),
            ),
        )
