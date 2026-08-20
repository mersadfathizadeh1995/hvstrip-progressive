"""Serialized session-op execution off the GUI thread + a progress coalescer.

The house worker layer for HV Strip (copy-not-import from the invert_hvsr
reference, extended for the **dict** progress frames).

* :class:`OpQueue` — one live :class:`OpWorker` ``QThread`` at a time; the rest
  queue.  Ops are named zero-arg callables (bound to the session by
  :class:`AppState`) returning an api envelope dict.  Unlike the invert
  reference, this api can RAISE (e.g. ``_resolve_profile`` → ``KeyError``
  before any envelope wrapping), so :meth:`OpWorker.run` catches everything
  and synthesizes a failure envelope — if it didn't, ``finished_env`` would
  never fire, ``OpQueue._current`` would never clear, and the queue would be
  wedged busy until app restart.

* :class:`ProgressBridge` — the ``progress_cb`` adapter for
  :meth:`InversionSession.run_inversion`.  The session's subprocess-reader (a
  plain ``threading.Thread``, DC-004) calls it with a live ``dict`` frame at
  the engine's own emission rate; the bridge is a **latest-wins coalescer** —
  it hops to the GUI thread through a queued signal and emits **one** frame per
  event-loop wakeup no matter how fast frames arrive, so the live canvas paints
  the newest frame and never floods the UI thread.  The engine-side emission is
  never touched (frozen); this is the UI-side throttle (review finding 2).
"""

from __future__ import annotations

import threading
import traceback
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QThread, Signal

#: A bound, zero-argument session op returning an envelope dict.
OpFn = Callable[[], dict]


class ProgressBridge(QObject):
    """Thread-safe latest-**per-type** relay for ``dict`` progress frames.

    The session calls ``bridge(frame)`` from the reader thread; frames are
    emitted (coalesced) on the GUI thread — **at most one frame per distinct
    ``frame["type"]`` per event-loop wakeup**, latest-wins within each type.

    This is the UI-side throttle (review finding 2 / DC-002): the engine emits
    interleaved ``log`` / ``iteration`` / ``model`` lines at its own rate, so a
    *global* latest-wins would let the frequent ``log`` lines starve the
    ``model`` frames the live plot needs.  Coalescing per type keeps the
    live-plot ``setData`` fed **and** the convergence points flowing while still
    bounding the work to O(#types) per wakeup — never O(#frames).  Frames with
    no ``type`` key share one bucket (so a plain ``{iteration, misfit}`` stream
    still collapses to one latest frame per wakeup).
    """

    #: emitted on the GUI thread, one coalesced frame per (type × wakeup).
    frame = Signal(dict)
    #: internal worker→GUI hop; queued so it always defers to the event loop.
    _tick = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        # Ordered latest-per-type buffer (insertion order → deterministic flush).
        self._latest: "dict[Any, Dict[str, Any]]" = {}
        self._pending = False
        # QueuedConnection: even a same-thread emit defers to the event loop,
        # so N rapid frames collapse to ONE _flush (the coalescing guarantee,
        # and it stays correct across the reader-thread boundary).
        self._tick.connect(self._flush, Qt.QueuedConnection)

    def __call__(self, frame: Dict[str, Any]) -> None:
        """Called from the session's reader thread with a live frame."""
        f = dict(frame) if frame is not None else {}
        with self._lock:
            key = f.get("type")
            self._latest[key] = f  # latest-wins within this type
            already = self._pending
            self._pending = True
        if not already:
            self._tick.emit()  # wake the GUI thread exactly once per batch

    def _flush(self) -> None:
        with self._lock:
            frames = list(self._latest.values())
            self._latest = {}
            self._pending = False
        for frame in frames:
            self.frame.emit(frame)


class OpWorker(QThread):
    """Runs one envelope-returning op on its own thread."""

    finished_env = Signal(str, dict)

    def __init__(
        self, name: str, fn: OpFn, parent: Optional[QObject] = None
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._fn = fn

    def run(self) -> None:  # pragma: no cover - exercised via OpQueue
        try:
            envelope = self._fn()
        except Exception as exc:  # noqa: BLE001 — the queue must never wedge
            envelope = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        if not isinstance(envelope, dict):
            envelope = {
                "success": False,
                "error": (f"op '{self._name}' returned "
                          f"{type(envelope).__name__}, not an envelope dict"),
            }
        self.finished_env.emit(self._name, envelope)


class OpQueue(QObject):
    """Serialized op runner: one live worker, the rest wait in line."""

    op_started = Signal(str)
    op_finished = Signal(str, dict)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._pending: Deque[Tuple[str, OpFn]] = deque()
        self._current: Optional[OpWorker] = None

    # ------------------------------------------------------------------
    @property
    def is_busy(self) -> bool:
        return self._current is not None

    @property
    def current_op(self) -> Optional[str]:
        return self._current._name if self._current else None  # noqa: SLF001

    @property
    def n_pending(self) -> int:
        return len(self._pending)

    # ------------------------------------------------------------------
    def submit(self, name: str, fn: OpFn) -> None:
        """Run *fn* as soon as every earlier op has finished."""
        self._pending.append((name, fn))
        self._start_next()

    def _start_next(self) -> None:
        if self._current is not None or not self._pending:
            return
        name, fn = self._pending.popleft()
        worker = OpWorker(name, fn, parent=self)
        worker.finished_env.connect(self._on_finished)
        self._current = worker
        self.op_started.emit(name)
        worker.start()

    def _on_finished(self, name: str, envelope: dict) -> None:
        worker = self._current
        self._current = None
        if worker is not None:
            worker.finished_env.disconnect(self._on_finished)
            worker.wait()
            worker.deleteLater()
        self.op_finished.emit(name, envelope)
        self._start_next()

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Drop queued ops and wait for the in-flight one (window close)."""
        self._pending.clear()
        if self._current is not None:
            self._current.wait(30_000)


__all__ = ["OpFn", "OpQueue", "OpWorker", "ProgressBridge"]
