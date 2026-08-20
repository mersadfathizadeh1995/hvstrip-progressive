"""Progress streaming for the api's long operations — WITHOUT touching core.

``core.batch_workflow`` (frozen compute) already narrates its progress on
stdout (``[1/3] Layer Stripping``, ``[OK] … completed in 1.75s``,
``[3/8] Creating peak evolution analysis...``).  :func:`tee_progress` captures
that narration with a line-parsing stdout tee — active ONLY while a
``progress_cb`` is supplied — and converts each line into a small dict frame:

    {"type": "phase", "index": 1, "total": 3, "label": "Layer Stripping"}
    {"type": "log",   "text": "[OK] Layer stripping completed in 0.02s"}

The api layers add their own coarser frames (``{"type": "profile", ...}`` per
batch item, ``{"type": "study_phase", ...}`` for research) at their loop
boundaries.  Frames are STEP/PHASE-granular by construction — the engine's
adaptive-rescan inner calls never emit frames.

Concurrency: the GUI runs TWO OpQueues (main + research), so the tee cannot
assume it owns the process's only worker thread.  Two guards keep it safe:
the tee only PARSES lines written by the thread that installed it (another
thread's prints pass straight through — never cross-attributed, and the line
buffer stays single-threaded), and installation is single-flight (a second
concurrent ``tee_progress`` degrades to a no-op instead of corrupting the
``sys.stdout`` swap nesting).  Output always reaches the real stdout (a true
tee), so logs and the legacy behaviour are unchanged.
"""

from __future__ import annotations

import re
import sys
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

ProgressCallback = Callable[[Dict[str, Any]], None]

#: ``[i/n] Label`` — the workflow's numbered phase lines.
_PHASE_RE = re.compile(r"^\[(\d+)\s*/\s*(\d+)\]\s*(.+?)\s*$")
#: Lines worth forwarding verbatim as log frames.
_LOG_PREFIXES = ("[OK]", "[>]", "[*]", "[!]", "[WARN", "[ERROR")


def parse_line(line: str) -> Optional[Dict[str, Any]]:
    """One narration line → a progress frame (or ``None`` for noise)."""
    text = line.rstrip()
    if not text:
        return None
    m = _PHASE_RE.match(text)
    if m:
        return {
            "type": "phase",
            "index": int(m.group(1)),
            "total": int(m.group(2)),
            "label": m.group(3),
        }
    if text.startswith(_LOG_PREFIXES):
        return {"type": "log", "text": text}
    return None


class _TeeWriter:
    """A file-like stdout tee: passes everything through to the real stream
    and feeds complete lines to the frame parser."""

    def __init__(self, cb: ProgressCallback, passthrough) -> None:
        self._cb = cb
        self._out = passthrough
        self._buf = ""
        self._owner = threading.get_ident()

    # -- file-like surface -------------------------------------------------
    def write(self, s: str) -> int:
        try:
            self._out.write(s)
        except Exception:  # noqa: BLE001 — never let the real stream kill an op
            pass
        if threading.get_ident() != self._owner:
            # Another thread's output (the other OpQueue, a pool worker…) —
            # not this op's narration: pass through unparsed, and keep the
            # line buffer single-threaded.
            return len(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self) -> None:
        try:
            self._out.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:  # matplotlib/rich probes
        return False

    # -- internals ----------------------------------------------------------
    def _emit(self, line: str) -> None:
        frame = parse_line(line)
        if frame is None:
            return
        try:
            self._cb(frame)
        except Exception:  # noqa: BLE001 — a GUI-side error must not kill compute
            pass

    def close_buffer(self) -> None:
        if self._buf:
            self._emit(self._buf)
            self._buf = ""


#: Single-flight guard for the process-global ``sys.stdout`` swap.
_tee_active = threading.Lock()


@contextmanager
def tee_progress(progress_cb: Optional[ProgressCallback]) -> Iterator[None]:
    """Context manager: tee stdout into *progress_cb* frames.

    A no-op when *progress_cb* is ``None`` — the byte-identical legacy path.
    Also a no-op when another op's tee is already installed: two concurrent
    installs would corrupt the swap nesting (the first uninstall could strand
    the second tee as ``sys.stdout`` forever), so the later op just runs
    without line-parsed frames (its api-level frames still flow).
    """
    if progress_cb is None:
        yield
        return
    if not _tee_active.acquire(blocking=False):
        yield
        return
    try:
        real = sys.stdout
        tee = _TeeWriter(progress_cb, real)
        sys.stdout = tee
        try:
            yield
        finally:
            sys.stdout = real
            tee.close_buffer()
    finally:
        _tee_active.release()


def emit(progress_cb: Optional[ProgressCallback], **frame: Any) -> None:
    """Best-effort emit of an api-level frame (no-op when cb is None)."""
    if progress_cb is None:
        return
    try:
        progress_cb(dict(frame))
    except Exception:  # noqa: BLE001
        pass


__all__ = ["ProgressCallback", "tee_progress", "parse_line", "emit"]
