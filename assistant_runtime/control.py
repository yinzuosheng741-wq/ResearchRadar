"""Cooperative cancellation and content-free progress across specialist calls."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class RunCancelled(RuntimeError):
    pass


_current = ContextVar("assistant_run", default=None)


@contextmanager
def run_control(cancel: Event, emit):
    token = _current.set((cancel, emit))
    try:
        yield
    finally:
        _current.reset(token)


def checkpoint(stage="working", agent="coordinator"):
    current = _current.get()
    if current:
        cancel, emit = current
        if cancel.is_set():
            raise RunCancelled("run_cancelled")
        emit({"stage": stage, "agent": agent})
