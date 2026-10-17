"""One process-wide ADK Runner.

The FastAPI lifespan, the WhatsApp webhook and the local smoke scripts must all
share the same runner so they share the same in-memory sessions. Creating a
second Runner would silently give the webhook a different conversation history
from the ADK playground.
"""

from __future__ import annotations

import threading

from google.adk.runners import Runner

from app.app_utils import services

_runner: Runner | None = None
_lock = threading.Lock()


def get_runner() -> Runner:
    global _runner
    with _lock:
        if _runner is None:
            from app.agent import app as adk_app

            _runner = Runner(
                app=adk_app,
                session_service=services.get_session_service(),
                artifact_service=services.get_artifact_service(),
                auto_create_session=True,
            )
        return _runner


def set_runner(runner: Runner | None) -> None:
    """Swap the process-wide runner. Tests use this to inject a fake."""
    global _runner
    with _lock:
        _runner = runner
