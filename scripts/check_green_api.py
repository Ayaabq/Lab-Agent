"""Read-only GREEN API instance check.

Calls only ``getStateInstance`` and ``getSettings``. Both are non-mutating:
nothing is sent, no notification is consumed or deleted, and no console
setting is changed.

    uv run python scripts/check_green_api.py
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()

from app.config import get_settings  # noqa: E402
from app.whatsapp.green_api import GreenApiError, get_green_api_client  # noqa: E402

#: Settings worth showing an operator. Anything that could carry a credential
#: is excluded by omission rather than by filtering.
_INTERESTING_SETTINGS = (
    "webhookUrl",
    "incomingWebhook",
    "outgoingWebhook",
    "outgoingMessageWebhook",
    "stateWebhook",
    "markIncomingMessagesReaded",
)


async def main() -> int:
    settings = get_settings()
    cfg = settings.green_api
    print(f"live sending enabled: {cfg.live_enabled}")
    if not cfg.is_configured:
        print("GREEN API is not fully configured — nothing to check.")
        return 1

    client = get_green_api_client()
    exit_code = 0
    try:
        try:
            state = await client.get_state_instance()
            print(f"stateInstance: {state.get('stateInstance', '<unknown>')}")
        except GreenApiError as exc:
            print(f"getStateInstance failed: {exc}")
            exit_code = 2

        try:
            snapshot = await client.get_settings_snapshot()
        except GreenApiError as exc:
            print(f"getSettings failed: {exc}")
            return exit_code or 2

        configured_hook = str(snapshot.get("webhookUrl") or "")
        print(f"webhook configured: {'yes' if configured_hook else 'no'}")
        for key in _INTERESTING_SETTINGS:
            if key in snapshot and key != "webhookUrl":
                print(f"  {key}: {snapshot[key]}")
        return exit_code
    finally:
        await client.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
