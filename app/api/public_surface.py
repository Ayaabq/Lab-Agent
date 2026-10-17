"""Keeps the ADK development surface off the public tunnel.

The serving app is the ADK-generated FastAPI app with our webhook mounted on
it. That app also carries the ADK dev UI and its management routes — listing
apps, reading and deleting sessions, running the agent directly, browsing
artifacts. Those are fine on a laptop. Published through a tunnel they are an
unauthenticated console for the agent: anyone with the URL could run it,
replay conversations, or read a session that holds a patient verification.

GREEN API only needs one route, and a health probe is harmless, so public
requests get exactly those two and a 404 for everything else — 404 rather than
403 so the public surface does not advertise what it is hiding.

"Public" means "arrived through the Cloudflare tunnel", detected by the
headers Cloudflare adds at its edge. Those headers can be forged by anyone
talking to this port directly, which is the right way round: forging them can
only ever *reduce* what a caller may reach. Requests without them — the
operator on localhost — keep the full surface.

On a hosted deployment there is no tunnel and no trustworthy header to tell a
public caller from the operator, so every request is treated as public. That
mode is switched on by ``PUBLIC_SURFACE_ONLY=true``, and automatically on
Render, which sets ``RENDER`` in every service's environment. It can only be
switched on: there is deliberately no way to opt a hosted process out.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from app.logging_utils import get_logger

logger = get_logger(__name__)

#: Everything a tunnelled caller legitimately needs.
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/webhooks/green-api",
        "/healthz",
    }
)

#: Set by Cloudflare at its edge on every request it forwards.
_EDGE_HEADERS = ("cf-ray", "cf-connecting-ip", "cf-visitor")


def hosted_publicly() -> bool:
    """True when this process is reachable from the internet with no tunnel.

    Read per request rather than at import so a test can flip it; it is two
    dictionary lookups.
    """
    flag = (os.getenv("PUBLIC_SURFACE_ONLY") or "").strip().lower()
    return flag in {"1", "true", "yes", "on"} or bool(os.getenv("RENDER"))


def arrived_from_the_public_edge(request: Request) -> bool:
    if hosted_publicly():
        return True
    return any(h in request.headers for h in _EDGE_HEADERS)


async def restrict_public_surface(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Allow only :data:`PUBLIC_PATHS` to requests coming in off the tunnel."""
    if arrived_from_the_public_edge(request):
        path = request.url.path.rstrip("/") or "/"
        if path not in PUBLIC_PATHS:
            logger.warning(
                "blocked a non-public path arriving from the public edge",
                extra={"context": {"path": path, "method": request.method}},
            )
            # 404, not 403: a public caller learns nothing about what exists.
            return JSONResponse({"detail": "Not Found"}, status_code=404)
    return await call_next(request)
