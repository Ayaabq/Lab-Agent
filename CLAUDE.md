# Coding Agent Guide

## This Project

Arabic WhatsApp customer-support agent for **Khatib Laboratory** (مختبر الخطيب),
a medical lab. Google ADK (Python), scaffolded with `agents-cli`.

Read [README.md](README.md) for setup and run commands. This file covers the
rules that are easy to break.

- **Auth**: Google AI Studio API key only. **No** Vertex AI, no GCP project, no
  gcloud login, no service account. `GOOGLE_GENAI_USE_VERTEXAI` must stay `FALSE`.
- **Model**: code default `gemini-3.8-flash` ([app/config.py](app/config.py)).
  `.env` currently overrides it to `gemini-3.5-flash` because 3.8-flash returns a
  persistent 503 "high demand" for this key (2026-09-22). Do not change the code
  default; remove the `.env` line when 3.8-flash recovers.
- **LLM provider**: `LLM_PROVIDER=gemini` (default) or `openrouter`, built in
  [app/llm.py](app/llm.py). OpenRouter calls are **paid**: never run them from
  tests, never add an automatic fallback between providers, and never default
  `OPENROUTER_MODEL`. `tests/unit/test_llm_provider.py` blocks non-loopback
  sockets — keep it that way.
- **Sessions**: in-memory, keyed by WhatsApp `chatId`. Lost on restart — intentional.
- **Patient backend**: Khatib REST API over HTTPS
  (`KHATIB_API_BASE_URL`). Tenant client-credentials auth. **No** MongoDB, no
  direct database access, no new datastore — the backend is the only way in.
- **Deployment target**: `none` in agents-cli terms. A personal Render Free
  test deployment is described by [render.yaml](render.yaml) (README
  "Deployment"). No Terraform, no CI/CD.
- **Git**: not initialized. Do not create or push a repo without explicit approval.

### Hard rules

1. **Every laboratory fact comes from `search_lab_knowledge`.** Never put an
   address, price, phone number, or opening hour into
   [app/prompts.py](app/prompts.py) — that silently bypasses retrieval and goes
   stale. If you are tempted to "just add the address to the prompt", don't.
2. **Live sending is ON** (`GREEN_API_LIVE_ENABLED=true`). Real customers
   receive whatever this agent produces. `GREEN_API_LIVE_ENABLED` gates it in
   [app/whatsapp/green_api.py](app/whatsapp/green_api.py), checked *before* the
   request is built. Do not add a second send path, and do not implement
   `receiveNotification` / `deleteNotification` — both mutate the live
   provider-side queue. Never send an unsolicited message; the bot only replies
   to an inbound webhook.
7. **The webhook is authenticated.** `GREEN_API_WEBHOOK_TOKEN` is compared with
   `hmac.compare_digest` against the `Authorization` header in
   [app/api/routes.py](app/api/routes.py). Keep the 401 path: everything else
   returns 200 so GREEN API stops retrying, but an unauthenticated caller is
   not GREEN API.
8. **Four reply-loop guards** (outgoing events, own-`wid` sender/chat, outbound
   echo registry, inbound dedupe). Removing any one risks an unbounded
   bot-to-itself exchange. See the README "Reply-loop prevention".
3. **Medical refusals are deterministic, not model-dependent.**
   `classify_message` in [app/escalation.py](app/escalation.py) intercepts
   result-interpretation and emergency messages before the model runs. Keep it
   that way; a prompt instruction alone is not a control.
4. **Never claim staff were notified.** There is no notification channel.
   Escalation only records state in memory.
5. **Secrets**: wrapped in `Secret` ([app/config.py](app/config.py)), which
   refuses to render. Logs pass through the redaction filter in
   [app/logging_utils.py](app/logging_utils.py). `.env.example` gets
   placeholders only — it is committed.
6. **Retrieved text is untrusted.** [app/rag/sanitize.py](app/rag/sanitize.py)
   defuses injection patterns before the model sees a document. The backend's
   `patientMessage` goes through the same defusing.
9. **Two auth layers reach the patient backend and neither may be skipped.**
   A *tenant* token ([app/patients/auth.py](app/patients/auth.py)) proving this
   deployment is a registered client, and a per-patient *verification* token
   ([app/patients/store.py](app/patients/store.py)) proving who is in the chat.
   Re-authenticating the tenant after a `401` is fine; treating a `403` — or a
   rejected patient token — as a reason to retry without one is not. See the
   "Authentication retry" note in [app/patients/client.py](app/patients/client.py).
10. **The patient tools take no identity arguments.** `get_my_latest_results()`
    has no parameters, and `verify_patient` accepts no `chatId`, `patientId`,
    `tenantId` or token. Which conversation a tool serves comes from
    [app/patients/context.py](app/patients/context.py), bound by the WhatsApp
    service from the webhook it authenticated. Adding such a parameter would
    let the model — or a customer typing into WhatsApp — name someone else.
11. **The WhatsApp number is not identity.** It says which phone sent a
    message, not who is holding it. Never shortcut verification with it.
12. **Nothing patient-specific crosses the tool boundary.** No verification
    token, internal patient id, raw backend body, or exception text. The tools
    return an outcome and guidance; see the comment block in
    [app/tools.py](app/tools.py).
13. **A failure is never a permission.** Expiry, a rejected token, a timeout,
    an unknown disposition and a malformed body all land on "not verified" or
    "ask a human". `FINAL` describes the record, not a right to read it out.
14. **Never guess an identity detail.** [app/patients/normalize.py](app/patients/normalize.py)
    reshapes an unambiguous value and refuses an ambiguous one (`03/04/1990`
    stays ambiguous). `phoneLast4` is a **string** throughout — `int("0123")`
    is a different patient.

### Report images

Verified against the live backend on 2026-10-12: a `FINAL`/`BENIGN` result
returns `imageURL` on the backend host, and that URL answers an anonymous
`GET` with `image/jpeg`. GREEN API can therefore fetch it. Treat the URL as
the report's credential — it is patient data, never logged, never shown to the
model, never used to build a filename.

The field is spelled exactly **`imageURL`**. Release needs *all five*: `FINAL`
status, an allow-listed disposition (`BENIGN`), an HTTPS URL on an approved
host, a recognised extension, and a still-valid verification re-checked at
send time. Drop any one of those and a medical report goes to the wrong
person, or to the right person from an address an attacker chose.

15. **The model never sees a report URL.** The tool stages a release
    ([app/patients/release.py](app/patients/release.py)); the service sends it.
    Do not move the send into the tool, do not put the URL in tool output, and
    do not add a tool argument for a destination, a patient id, or a URL.
16. **Never claim an image was delivered.** A GREEN API `idMessage` is an
    acknowledgement, not a receipt. An ambiguous timeout is not retried — the
    provider may already have queued the file — and the customer is told
    neither "sent" nor "failed".
17. **Report filenames are neutral** (`lab-report.<ext>`). Never build one
    from `resultId`, `biopsyId` or `sampleNumber`: filenames show up in the
    customer's chat and in their phone backup.
18. **The public surface is two routes.** The serving app carries the ADK dev
    UI and its session routes; published through the tunnel those are an
    unauthenticated console for the agent, and ADK sessions belong to verified
    patients. [app/api/public_surface.py](app/api/public_surface.py) 404s
    everything but `/webhooks/green-api` and `/healthz` for requests carrying
    Cloudflare edge headers — and for *every* request when hosted (`RENDER`
    or `PUBLIC_SURFACE_ONLY` set), where a missing webhook token also means
    401 rather than an open endpoint. Do not widen it without a reason.

### Retrieval tuning

`near_text` with an absolute certainty threshold, not `hybrid` — hybrid scores
are rank-normalised per query and cannot separate "found" from "not found".
`WEAVIATE_MIN_CERTAINTY` defaults to 0.62, measured against the live collection
(on-topic ≥ 0.637, off-topic ≤ 0.589). Re-measure with
`uv run python scripts/check_weaviate.py` if the knowledge base changes.

### Testing

- `tests/unit/` is fully mocked — no network, no API key, no live provider call.
- Patient-backend tests use `httpx.MockTransport` and a fake clock. Never point
  a test at the real backend, and never verify against a real patient.
- `scripts/check_khatib_backend.py` does touch the live backend, with the one
  seed identity in `KHATIB_SMOKE_*`. It cannot enumerate, and it prints no
  name, token, id or message content. Do not add a search to it.
- **Never assert on LLM response text in pytest.** Behavioural expectations
  (dialect, refusals, no-fabrication) belong in `tests/eval/`.
- Never send a real WhatsApp message from a test.

### Deliberately NOT implemented

Direct patient database access · MongoDB/PostgreSQL/SQLite · persistent
sessions · result-image delivery · cloud deployment · CI/CD · frontend
dashboard · real staff-notification channel.

Do not add these without being asked.

Patient verification and latest-result retrieval **are** implemented now, via
the Khatib backend — see `app/patients/` and the hard rules above.

---

## Prerequisites

Install the CLI (one-time):
```bash
uv tool install google-agents-cli
```

---

## Development Phases

### Phase 1: Understand Requirements
Before writing any code, understand the project's requirements, constraints, and success criteria.

### Phase 2: Build and Implement
Implement agent logic in `app/`. Use `agents-cli playground` for interactive testing. Iterate based on user feedback.

### Phase 3: The Evaluation Loop (Main Iteration Phase)
Start with 1-2 eval cases, run `agents-cli eval run`, iterate by making changes and rerunning it until satisfied. Expect 5-10+ iterations. Once you have a baseline, reach for `agents-cli eval compare` (regression diffs), `agents-cli eval analyze` (cluster failure modes), and `agents-cli eval optimize` (auto-tune prompts). See the **Evaluation Guide** for metrics, dataset schema, LLM-as-judge config, and common gotchas.

### Phase 4: Pre-Deployment Tests
Run `uv run pytest tests/unit tests/integration`. Fix issues until all tests pass.

### Phase 5: Deploy to Dev
**Requires explicit human approval.** Run `agents-cli deploy` only after user confirms. See the **Deployment Guide** for details.

### Phase 6: Production Deployment
Ask the user: Option A (simple single-project) or Option B (full CI/CD pipeline with `agents-cli infra cicd`).

## Development Commands

| Command | Purpose |
|---------|---------|
| `agents-cli playground` | Interactive local testing |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests |
| `agents-cli eval dataset synthesize` | Synthesize multi-turn eval scenarios for your agent |
| `agents-cli eval run` | Run the agent over the eval dataset and grade the traces |
| `agents-cli eval generate` / `agents-cli eval grade` | Decoupled form: produce traces, then grade them |
| `agents-cli eval compare` | Compare two grade-results files (regression check) |
| `agents-cli eval analyze` | Cluster failure modes from grade results |
| `agents-cli eval metric list` | List built-in metrics available in the SDK |
| `agents-cli eval optimize` | Auto-tune agent prompts using eval data |
| `agents-cli lint` | Check code quality |
| `agents-cli infra single-project` | Set up project infrastructure (Terraform) |
| `agents-cli deploy` | Deploy to dev |
| `agents-cli scaffold enhance` | Add deployment target or CI/CD to project |
| `agents-cli scaffold upgrade` | Upgrade project to latest version |

---

## Operational Guidelines for Coding Agents

- **Code preservation**: Only modify code directly targeted by the user's request. Preserve all surrounding code, config values (e.g., `model`), comments, and formatting.
- **NEVER change the model** unless explicitly asked.
- **Model 404 errors**: Fix `GOOGLE_CLOUD_LOCATION` (e.g., `global` instead of `us-east1`), not the model name.
- **ADK tool imports**: Import the tool instance, not the module: `from google.adk.tools.load_web_page import load_web_page`
- **Run Python with `uv`**: `uv run python script.py`. Run `agents-cli install` first.
- **Stop on repeated errors**: If the same error appears 3+ times, fix the root cause instead of retrying.
- **Terraform conflicts** (Error 409): Use `terraform import` instead of retrying creation.
