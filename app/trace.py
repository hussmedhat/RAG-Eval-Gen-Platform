"""
Lightweight per-request trace capture.

Agents already log their step-by-step progress via the standard
`logging` module (query rewrites, retrieval counts, analyst loop
decisions, evidence requests, etc.) — this hooks into that existing
logging instead of threading a trace list through every function
signature. A handler attached to the 'app' logger namespace captures
INFO+ records into a thread-local list for the duration of one
request, so /ask can return the same step-by-step trace visible in
the terminal, without modifying any agent code.
"""
import logging
import threading

_trace_local = threading.local()


class _TraceHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        entries = getattr(_trace_local, "entries", None)
        if entries is not None:
            entries.append({
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
            })


_trace_handler = _TraceHandler()
_trace_handler.setLevel(logging.INFO)

_installed = False


def install_trace_handler() -> None:
    """Attaches the trace handler to the 'app' logger namespace once.
    Attaching to 'app' rather than the root logger means only your own
    modules' logs are captured — not httpx/langchain/uvicorn noise,
    since none of those live under the 'app.' package name."""
    global _installed
    if _installed:
        return
    logging.getLogger("app").addHandler(_trace_handler)
    _installed = True


def start_trace() -> None:
    """Call at the start of a request to begin capturing this thread's
    log output into a fresh trace list."""
    _trace_local.entries = []


def get_trace() -> list[dict]:
    """Returns whatever was captured on this thread since the last
    start_trace() call, then clears it."""
    entries = getattr(_trace_local, "entries", [])
    _trace_local.entries = None
    return entries
