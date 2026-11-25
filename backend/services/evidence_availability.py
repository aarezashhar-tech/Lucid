"""
Per-request, thread-local tracker for evidence-source failures.

A failed Wikipedia/GNews/FactCheck lookup must never be silently presented as
"no evidence found". Services mark failures here, and the verification pipeline
surfaces them as `evidenceUnavailable` in the API response.
"""
import contextvars
import threading
from typing import Dict, Tuple


_request_state: contextvars.ContextVar[Tuple[Dict[str, str], threading.Lock] | None] = contextvars.ContextVar(
    "lucid_evidence_tracker_state", default=None
)


class EvidenceAvailabilityTracker:
    def __init__(self):
        self._local = threading.local()

    def _state(self) -> Tuple[Dict[str, str], threading.Lock]:
        ctx_state = _request_state.get()
        if ctx_state is not None:
            return ctx_state
        if not hasattr(self._local, "state"):
            self._local.state = ({}, threading.Lock())
        return self._local.state

    def reset(self) -> None:
        fresh = ({}, threading.Lock())
        _request_state.set(fresh)
        self._local.state = fresh

    def mark_failure(self, source: str, error: object) -> None:
        failures, lock = self._state()
        with lock:
            failures[source] = str(error)

    def mark_success(self, source: str) -> None:
        failures, lock = self._state()
        with lock:
            failures.pop(source, None)

    def snapshot(self) -> Dict[str, str]:
        failures, lock = self._state()
        with lock:
            return dict(failures)

    def any_unavailable(self) -> bool:
        failures, lock = self._state()
        with lock:
            return bool(failures)


evidence_tracker = EvidenceAvailabilityTracker()
