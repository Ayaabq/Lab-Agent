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

from google.adk.agents import Agent
from google.adk.apps import App

from app.config import get_settings
from app.llm import build_model
from app.logging_utils import configure_logging, get_logger
from app.prompts import SYSTEM_INSTRUCTION
from app.tools import (
    get_my_latest_results,
    request_human_agent,
    search_lab_knowledge,
    verify_patient,
)

_settings = get_settings()
configure_logging(_settings)
logger = get_logger(__name__)

# Authentication is an AI Studio API key (GOOGLE_API_KEY). Vertex AI, a GCP
# project, and service accounts are deliberately out of scope for this project.
if _settings.gemini.use_vertexai:
    logger.warning(
        "GOOGLE_GENAI_USE_VERTEXAI is true; this project expects an AI Studio key"
    )

# LLM_PROVIDER selects Gemini (the default) or OpenRouter; see app/llm.py.
MODEL = build_model(_settings)


root_agent = Agent(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name. Renaming the agent only here makes the two disagree,
    # and anything selecting traces by name stops finding this agent's.
    name="khatib_lab_support",
    model=MODEL,
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        search_lab_knowledge,
        request_human_agent,
        # Patient-specific. verify_patient must succeed before
        # get_my_latest_results can read anything; the gate is enforced in
        # app/tools.py, not by tool ordering or by the prompt.
        verify_patient,
        get_my_latest_results,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
