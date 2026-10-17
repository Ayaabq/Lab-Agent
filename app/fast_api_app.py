# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from collections.abc import AsyncIterator

from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

from app.api.public_surface import hosted_publicly, restrict_public_surface
from app.api.routes import router as khatib_router
from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.app_utils.runner import get_runner
from app.config import get_settings
from app.logging_utils import configure_logging, get_logger

load_dotenv()
configure_logging(get_settings())
logger = get_logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)
# Cloud Trace export needs Application Default Credentials. This project
# authenticates with an AI Studio API key and has no GCP project, so exporting
# would abort startup with DefaultCredentialsError. Enabled only if someone
# deliberately sets GOOGLE_CLOUD_PROJECT later.
otel_to_cloud = bool(os.getenv("GOOGLE_CLOUD_PROJECT"))

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import app as adk_app
    from app.agent import root_agent

    # A missing variable degrades a feature rather than stopping the server,
    # and /health/integrations is not reachable on a hosted deployment, so the
    # startup log is where an operator finds out. Names only, never values.
    settings = get_settings()
    for problem in settings.problems():
        logger.warning("configuration problem", extra={"context": {"problem": problem}})
    logger.info(
        "server starting",
        extra={
            "context": {
                "model": settings.gemini.model,
                "live_sending_enabled": str(settings.green_api.live_enabled),
                "webhook_token_required": str(settings.green_api.webhook_token.is_set),
                "public_surface_only": str(hosted_publicly()),
            }
        },
    )

    # Shared with the WhatsApp webhook so both surfaces see the same sessions.
    runner = get_runner()
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    try:
        yield
    finally:
        from app.patients.client import get_khatib_client
        from app.rag.weaviate_store import close_store
        from app.whatsapp.green_api import get_green_api_client
        from app.whatsapp.service import get_whatsapp_service

        await get_whatsapp_service().drain()
        await get_green_api_client().aclose()
        # Also closes the tenant authenticator's own connection pool.
        await get_khatib_client().aclose()
        close_store()


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=otel_to_cloud,
    lifespan=lifespan,
)
app.title = "khatib-lab-support"
app.description = "API for interacting with the Agent khatib-lab-support"

# WhatsApp webhook + health endpoints.
app.include_router(khatib_router)

# Must be added after the routers so it wraps them: anything arriving through
# the Cloudflare tunnel sees only the webhook and the liveness probe, never the
# ADK dev UI or its session-management routes.
app.middleware("http")(restrict_public_surface)


# Main execution
if __name__ == "__main__":
    import uvicorn

    # Hosting platforms (Render included) say which port to bind via $PORT.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
