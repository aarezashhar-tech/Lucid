"""
Per-request, thread-local tracker for evidence-source failures.

A failed Wikipedia/GNews/FactCheck lookup must never be silently presented as
"no evidence found". Services mark failures here, and the verification pipeline
surfaces them as `evidenceUnavailable` in the API response.
"""
import threading
from typing import Dict


class EvidenceAvailabilityTracker:
    def __init__(self):
        self._local = threading.local()

    def _failures(self) -> Dict[str, str]:
        if not hasattr(self._local, "failures"):
            self._local.failures = {}
        return self._local.failures

    def reset(self) -> None:
        self._local.failures = {}

    def mark_failure(self, source: str, error: object) -> None:
        self._failures()[source] = str(error)

    def mark_success(self, source: str) -> None:
        self._failures().pop(source, None)

    def snapshot(self) -> Dict[str, str]:
        return dict(self._failures())

    def any_unavailable(self) -> bool:
        return bool(self._failures())


evidence_tracker = EvidenceAvailabilityTracker()
