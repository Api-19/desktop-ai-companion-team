import threading
import time
from collections.abc import Callable, Iterator

import pytest

from companion.bodylink.pairing import PairingPaths, generate_pairing

BODY_ID = "body-1"


@pytest.fixture(scope="session")
def pairing(tmp_path_factory: pytest.TempPathFactory) -> PairingPaths:
    return generate_pairing(tmp_path_factory.mktemp("pair") / "main", BODY_ID)


@pytest.fixture(scope="session")
def foreign_pairing(tmp_path_factory: pytest.TempPathFactory) -> PairingPaths:
    """Same body_id, different CA."""
    return generate_pairing(tmp_path_factory.mktemp("pair") / "foreign", BODY_ID)


class Recorder:
    """Thread-safe sink for transport callbacks."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.started: list[tuple[str, float]] = []
        self.ended: list[tuple[str, str, float]] = []
        self.messages: list[object] = []
        self.frames: list[object] = []
        self.hellos: list[object] = []

    def _add(self, items: list, item: object) -> None:
        with self._cond:
            items.append(item)
            self._cond.notify_all()

    def on_session_started(self, session_id: str, hello: object = None) -> None:
        if hello is not None:
            self._add(self.hellos, hello)
        self._add(self.started, (session_id, time.monotonic()))

    def on_session_ended(self, session_id: str, reason: str) -> None:
        self._add(self.ended, (session_id, reason, time.monotonic()))

    def on_message(self, message: object) -> None:
        self._add(self.messages, message)

    def on_frame(self, frame: object) -> None:
        self._add(self.frames, frame)

    def wait_for(self, predicate: Callable[[], bool], timeout: float = 5.0) -> None:
        with self._cond:
            assert self._cond.wait_for(predicate, timeout), "timed out waiting for callback"

    def callbacks(self) -> dict[str, Callable]:
        return {
            "on_session_started": self.on_session_started,
            "on_session_ended": self.on_session_ended,
            "on_message": self.on_message,
            "on_frame": self.on_frame,
        }


@pytest.fixture
def recorder_factory() -> Callable[[], Recorder]:
    return Recorder


@pytest.fixture
def closer() -> Iterator[Callable]:
    """Register endpoints to close at teardown even if the test fails."""
    endpoints: list = []
    yield endpoints.append
    for endpoint in reversed(endpoints):
        endpoint.close()
