# khatib-lab-support

Arabic WhatsApp customer-support agent for **Khatib Laboratory** (مختبر الخطيب).

Built on the [Google ADK](https://adk.dev/), scaffolded with `agents-cli` 1.7.0.
Answers customer questions in spoken Levantine Arabic, grounding **every**
factual claim in a Weaviate Cloud knowledge base, and hands off to a human when
it cannot answer safely.

> **⚠ LIVE.** `GREEN_API_LIVE_ENABLED=true` in `.env`. Real WhatsApp replies are
> sent to anyone who messages the instance. To stop immediately, set it back to
> `false` and restart — no code change needed.
>
> The webhook is authenticated: `GREEN_API_WEBHOOK_TOKEN` must match the
> `webhookUrlToken` configured in the GREEN API console, or requests get a 401.

---

## What it does

| Capability | Where |
|---|---|
| Arabic (Levantine) support persona, WhatsApp-length replies | [app/prompts.py](app/prompts.py) |
| ADK `root_agent` and tool wiring | [app/agent.py](app/agent.py) |
| Retrieval over the lab knowledge base | [app/rag/weaviate_store.py](app/rag/weaviate_store.py) |
| Prompt-injection defusing of retrieved text | [app/rag/sanitize.py](app/rag/sanitize.py) |
| Agent tools (knowledge, escalation, verification, results) | [app/tools.py](app/tools.py) |
| Human-escalation triage and state | [app/escalation.py](app/escalation.py) |
| Tenant authentication for the patient backend | [app/patients/auth.py](app/patients/auth.py) |
| Patient verification + latest-result HTTP client | [app/patients/client.py](app/patients/client.py) |
| Verification state, scoped and expiring | [app/patients/store.py](app/patients/store.py) |
| Trusted conversation identity for the patient tools | [app/patients/context.py](app/patients/context.py) |
| Arabic-digit / date / phone-suffix parsing | [app/patients/normalize.py](app/patients/normalize.py) |
| Result disposition handling and image gating | [app/patients/results.py](app/patients/results.py) |
| Staging a report image for delivery | [app/patients/release.py](app/patients/release.py) |
| Keeping ADK dev routes off the public tunnel | [app/api/public_surface.py](app/api/public_surface.py) |
| GREEN API client (dry-run guarded) | [app/whatsapp/green_api.py](app/whatsapp/green_api.py) |
| Webhook parsing and filtering | [app/whatsapp/models.py](app/whatsapp/models.py) |
| Duplicate-delivery suppression | [app/whatsapp/dedupe.py](app/whatsapp/dedupe.py) |
| Webhook orchestration, per-chat sessions | [app/whatsapp/service.py](app/whatsapp/service.py) |
| HTTP routes and health endpoints | [app/api/routes.py](app/api/routes.py) |
| Configuration and secret handling | [app/config.py](app/config.py) |
| Structured logging with redaction | [app/logging_utils.py](app/logging_utils.py) |

### Not implemented (deliberately out of scope)

Direct patient database access · MongoDB/PostgreSQL/SQLite · persistent
sessions · result-image delivery · cloud deployment · CI/CD · frontend
dashboard · real staff-notification channel.

Patient identity verification and latest-result retrieval **are** implemented,
through the laboratory's own REST backend. This project holds no patient data
of its own and opens no database connection.

---

## Project structure

```
khatib-lab-support/
├── app/
│   ├── agent.py                   # ADK root_agent + App
│   ├── prompts.py                 # Arabic system instruction (no lab facts)
│   ├── config.py                  # Typed settings; secrets wrapped in Secret
│   ├── logging_utils.py           # JSON logs + secret/PII redaction
│   ├── escalation.py              # Triage + in-memory awaiting-human state
│   ├── tools.py                   # The four agent-facing tools
│   ├── fast_api_app.py            # FastAPI entry point (ADK + our router)
│   ├── api/
│   │   ├── routes.py              # /webhooks/green-api, /healthz, /health/integrations
│   │   └── public_surface.py      # Tunnel sees only the webhook + /healthz
│   ├── patients/
│   │   ├── auth.py                # Cached tenant login (one at a time)
│   │   ├── client.py              # Bounded HTTP client for the backend
│   │   ├── context.py             # Which conversation a tool is serving
│   │   ├── normalize.py           # Arabic digits, dates, phone suffixes
│   │   ├── release.py             # Staged report images awaiting delivery
│   │   ├── results.py             # Disposition handling; report-image gating
│   │   └── store.py               # Verification state, scoped and expiring
│   ├── rag/
│   │   ├── weaviate_store.py      # Read-only Weaviate Cloud retrieval
│   │   └── sanitize.py            # Untrusted-content defusing
│   ├── whatsapp/
│   │   ├── green_api.py           # Provider client; live sending guarded
│   │   ├── models.py              # Typed webhook payloads + skip rules
│   │   ├── dedupe.py              # Bounded TTL cache keyed by idMessage
│   │   └── service.py             # Orchestration, per-chat serialisation
│   └── app_utils/                 # Scaffold-provided services, A2A, runner
├── scripts/
│   ├── check_weaviate.py          # Read-only connectivity check
│   ├── check_green_api.py         # Read-only instance check (non-mutating)
│   ├── check_khatib_backend.py    # Login -> verify -> latest result (seed identity)
│   └── agent_smoke.py             # Arabic questions through the agent
├── tests/
│   ├── unit/                      # Fully mocked; no network
│   ├── integration/               # Live model calls
│   └── eval/                      # Behavioural eval dataset
├── .env                           # Real secrets. Git-ignored. Never commit.
├── .env.example                   # Placeholders only
└── CLAUDE.md                      # Guidance for AI-assisted development
```

---

## Setup

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) and
`agents-cli` (`uv tool install google-agents-cli`).

**No Google Cloud account, gcloud login, Vertex AI, or service account is
needed.** Authentication is a single Google AI Studio API key.

```bash
cp .env.example .env     # then fill in real values in .env
agents-cli install
```

Required in `.env`:

| Variable | Purpose |
|---|---|
| `GOOGLE_GENAI_USE_VERTEXAI` | Must be `FALSE` |
| `GOOGLE_API_KEY` | Google AI Studio key |
| `GEMINI_MODEL` | Code default is `gemini-3.8-flash`; see the note below |
| `WEAVIATE_URL`, `WEAVIATE_API_KEY` | Weaviate Cloud cluster |
| `WEAVIATE_COLLECTION` | Optional; auto-discovered when the cluster has exactly one |
| `GREEN_API_API_URL`, `GREEN_API_ID_INSTANCE`, `GREEN_API_TOKEN_INSTANCE` | WhatsApp provider |
| `GREEN_API_LIVE_ENABLED` | **Keep `false`** until you are ready to message real customers |
| `KHATIB_API_BASE_URL` | Patient/results backend. No default; unset disables the patient tools |
| `KHATIB_CLIENT_ID`, `KHATIB_CLIENT_SECRET` | Tenant credentials for that backend |
| `KHATIB_IMAGE_ALLOWED_HOSTS` | Optional; defaults to the backend host only |

Without `KHATIB_API_BASE_URL` and the two `KHATIB_*` credentials the agent still answers general
questions; patient verification and results report a temporary technical issue
and offer a human handoff. `/health/integrations` names the missing variable.

### Note on the model

The code default in [app/config.py](app/config.py) is `gemini-3.8-flash`, the
model the scaffold chose. As of 2026-09-22 that model returns a persistent
`503 UNAVAILABLE — "This model is currently experiencing high demand"` for this
API key, so `.env` pins `GEMINI_MODEL=gemini-3.5-flash`, which is verified
working end to end. Delete that line from `.env` to return to the code default
once 3.8-flash has capacity again.

**Free-tier quota.** This key is on the Gemini API free tier: **20 requests per
day, per model**. One customer question costs 2+ requests (one to choose the
tool call, one to answer), so a single `scripts/agent_smoke.py` run can exhaust
a day's allowance for one model. A `429 RESOURCE_EXHAUSTED` means the daily
quota is gone — point `GEMINI_MODEL` at a different model (each has its own
quota), wait for the reset, or move the key to a paid plan.

---

## Running locally

```bash
# ADK playground — chat with the agent in a browser
agents-cli playground

# One-off prompt
agents-cli run "شو أوقات الدوام عندكم؟"

# FastAPI server (webhook + health + ADK web UI) on :8000
uv run uvicorn app.fast_api_app:app --host 0.0.0.0 --port 8000

# Representative Arabic questions, end to end (no WhatsApp involved)
uv run python scripts/agent_smoke.py
```

### Health endpoints

```bash
curl localhost:8000/healthz
curl localhost:8000/health/integrations           # config only, no external calls
curl 'localhost:8000/health/integrations?probe=true'  # + read-only Weaviate probe
```

### Testing the webhook without WhatsApp

> **⚠ Only while `GREEN_API_LIVE_ENABLED=false`.** With live sending on, this
> payload makes the bot send a **real WhatsApp message** to whatever `chatId`
> it names — an unsolicited message to a real person if that number is real.
> Check the flag before running it, or point `chatId` at a number you control.

With the server running, post a realistic GREEN API payload. While
`GREEN_API_LIVE_ENABLED=false` nothing is sent back to WhatsApp — watch the
logs for the `DRY-RUN` line instead.

```bash
curl -X POST localhost:8000/webhooks/green-api \
  -H 'Content-Type: application/json' \
  -d '{
    "typeWebhook": "incomingMessageReceived",
    "instanceData": {"idInstance": 0, "wid": "963900000000@c.us"},
    "timestamp": 1730000000,
    "idMessage": "TEST-1",
    "senderData": {"chatId": "963991234567@c.us", "sender": "963991234567@c.us", "senderName": "زبون"},
    "messageData": {"typeMessage": "textMessage", "textMessageData": {"textMessage": "وين موقع المخبر؟"}}
  }'
```

Re-posting the same `idMessage` returns `{"status":"ignored","reason":"duplicate"}`.

### Safe connectivity checks

Both are read-only. `check_green_api.py` calls only `getStateInstance` and
`getSettings` — it never sends, never consumes the notification queue, and never
changes console settings.

```bash
uv run python scripts/check_weaviate.py
uv run python scripts/check_green_api.py
```

### Patient backend smoke test

`check_khatib_backend.py` runs the three calls a real conversation makes —
tenant login, patient verification, latest result — against **one** designated
seed identity read from `KHATIB_SMOKE_FULL_NAME` plus `KHATIB_SMOKE_DATE_OF_BIRTH`
or `KHATIB_SMOKE_PHONE_LAST4`. It cannot search or enumerate, and it prints no
name, birth date, phone suffix, token, patient id, or message content.

```bash
uv run python scripts/check_khatib_backend.py
```

With no seed identity configured it stops after the login step. Never point
those variables at a real patient.

---

## Quality checks

```bash
agents-cli lint                       # ruff check + format + codespell + ty
uv run pytest tests/unit -q           # fully mocked, no network, no API key
uv run pytest tests/integration       # calls the real model; needs GOOGLE_API_KEY
agents-cli eval run                   # behavioural evaluation (see tests/eval/)
```

Behavioural expectations (dialect, refusals, no-fabrication) live in
`tests/eval/`, not in pytest — LLM output is non-deterministic, so asserting on
response text in unit tests produces flaky suites.

---

## How grounding works

`search_lab_knowledge` is the only source of laboratory facts. The system
instruction contains **no** lab data on purpose: adding an address or a price
there would bypass retrieval and start producing stale answers.

**Retrieval tuning.** Queries use `near_text`, not `hybrid`. Hybrid scores are
rank-normalised per query — an entirely off-topic question still scores ~1.0 on
its best match — so they cannot distinguish "found" from "not found".
`near_text` certainty is absolute. Measured against the live collection,
on-topic questions scored ≥ 0.637 and off-topic ones ≤ 0.589, so
`WEAVIATE_MIN_CERTAINTY` defaults to **0.62**. Retune it if the knowledge base
changes shape:

```bash
uv run python scripts/check_weaviate.py "سؤال جديد"
```

Below the threshold the tool returns `found: false`, and the agent says the
information is unavailable and offers a human handoff rather than guessing.

**Retrieved text is untrusted.** Anything matching an instruction-injection
pattern is replaced with a visible marker before the model sees it, and the
tool's own guidance tells the model to treat results as reference data only.

---

## Patient verification and laboratory results

The agent can tell a customer the **status** of their own latest result. It
cannot read out a value, interpret anything, or send a report.

### Two authentication layers

Both are required for every result read, and neither can be skipped.

1. **Tenant** — client-credentials login to
   `POST /api/v1.0/auth/tenant/login`, proving this deployment is a registered
   client. It is infrastructure: the customer never supplies it, the model
   never sees it, and no agent-facing surface can trigger a login directly.
   The token is cached per tenant, renewed early (default 60s before expiry),
   and a burst of inbound messages produces exactly one login, not one each.
   The response carries a `refreshToken`, but **no refresh endpoint has been
   specified**, so it is not stored — renewal is a fresh login.
2. **Patient** — `POST /api/v1.0/patients/verify` returns a verification token
   that is sent as `x-patient-verification-token` alongside the tenant bearer
   on `GET /api/v1.0/results/{patientId}/latest`.

A `401` re-authenticates the tenant once. A `403` does not: on the results
route it can equally mean the *patient* token was refused, and re-authenticating
the tenant must never be mistaken for clearing a patient check.

### What the customer has to give

Full name, plus **one** of: date of birth, or the last four digits of the phone
number registered with the lab. The agent collects whatever is missing across
messages and verifies once it has enough.

- Arabic-Indic and Persian digits are accepted everywhere (`١٩٩٠-٠٣-٠٤`).
- `25/12/1990` is unambiguous. `03/04/1990` is **not**, and the agent asks
  rather than choosing a reading — a guessed birth date is a guessed identity.
- The phone suffix stays a four-character **string**: `"0123"` never becomes
  `123`, which would be a different patient.
- The WhatsApp number is not identity. It says which phone sent a message, not
  who is holding it.

### Where the verification lives

Server-side only, in memory, keyed by **tenant + channel + chatId + ADK
session**. The patient id and verification token never reach the model, the
customer, a log line, or ADK session state.

It is cleared when it expires, when the customer starts verifying as someone
else, when a check fails, and when the backend rejects the token. A new ADK
session does not inherit it, and no chat can see another chat's. Failed checks
are counted per chat; past the limit (`KHATIB_MAX_VERIFICATION_ATTEMPTS`,
default 5) the conversation goes to a human instead. A backend outage does not
count against that budget — it was not the customer's doing.

A failed check never says whether a patient with those details exists. Wrong
details and no such patient are one outcome.

### Result handling

| Backend says | The agent does |
|---|---|
| `FINAL` / `BENIGN` + a usable `imageURL` | Relays `patientMessage` unchanged **as the report image's caption** and sends the image to the same chat |
| `FINAL` / `BENIGN` with no usable image | Relays `patientMessage` and says plainly that the report copy cannot be sent here. No image, no promise of one |
| `DRAFT` / `PENDING` | Relays the backend's `patientMessage`. Never "ready", never a report image, never a collection time of its own |
| `DRAFT` / `PENDING`, `expectedReadyAt` already past | Hands off to a human. Does not repeat the lapsed time, does not invent a new one |
| `FINAL` / `CLINICIAN_REVIEW_REQUIRED` | Relays `patientMessage` **and** hands off. `FINAL` describes the record, not a right to discuss it |
| Anything else — unknown disposition, missing message, empty body | Hands off to a human |
| Timeout, outage, bad tenant credential | One "temporary technical issue" reply. The model is not told which |

The backend's `patientMessage` is the source of truth for anything
patient-specific, and goes through the same injection-defusing as retrieved
knowledge-base text. General factual questions still come from Weaviate.

As everywhere else in this project, no reply claims a staff member was
notified: there is still no notification channel.

### Report images

The backend returns the report image URL in a field named exactly **`imageURL`**
(configurable as `KHATIB_RESULT_IMAGE_FIELD`). A URL on its own authorises
nothing. Five independent facts must all hold before a report leaves:

1. `status` is `FINAL` — on its own this authorises nothing, but its absence forbids release.
2. `disposition` is allow-listed by `KHATIB_IMAGE_RELEASE_DISPOSITIONS` (default `BENIGN`).
3. The URL is **HTTPS** on a host in `KHATIB_IMAGE_ALLOWED_HOSTS` (default: the backend host only).
4. The path ends in a recognised extension — `.jpg`, `.jpeg`, `.png`, `.webp`, `.pdf`.
5. The conversation still holds the **same** unexpired verification, re-checked
   at the moment of sending rather than when the backend answered.

Where the sending happens matters. The tool *stages* a release
([app/patients/release.py](app/patients/release.py)); the WhatsApp service
sends it, after the text reply and inside the per-chat lock. That keeps the
image URL out of the model's reach entirely — it is never in tool output, a
prompt, or a log line — and puts the send where the live-sending switch, the
de-duplication and the authenticated chat id already live.

The file is sent with GREEN API `sendFileByUrl` as `lab-report.<ext>`. The
filename is deliberately neutral: a report arriving as `biopsy-99134.pdf`
would put a record identifier in the customer's chat and every backup of it.
`patientMessage` rides along as the caption when it fits in WhatsApp's 1024
characters, and is sent as its own complete message when it does not — the
laboratory's wording about a result is relayed whole or not at all.

**Delivery is never claimed.** GREEN API returning an `idMessage` means it
accepted the request, not that anyone received a file, so no reply says the
report was delivered. If the send fails the customer gets an Arabic message
saying it could not be sent; if it times out *ambiguously* they get one that
says neither — and the report is not pushed again, because the provider may
already have queued it.

> **Access contract: resolved (2026-10-12).** GREEN API fetches `imageURL`
> from its own infrastructure with none of our credentials, so the URL has to
> be retrievable anonymously. Verified against the live backend with the
> designated seed identity: an unauthenticated `GET` returns `200`,
> `image/jpeg`. No signed-URL work is needed for delivery to function.
>
> **Privacy consequence worth a decision.** Because that URL is retrievable by
> anyone who has it, the URL *is* the credential for a patient's report. That
> is unavoidable given GREEN API's fetch model, but it means the path must be
> unguessable, and the capability should ideally expire. Two questions for the
> backend owner: are report paths high-entropy rather than sequential, and can
> they be given a short TTL? This project already treats the URL as patient
> data — never logged, never shown to the model, never put in a filename.

---

## Human escalation

Escalation is recorded **in memory only** and is lost on restart. There is no
staff-notification channel yet, so no message ever claims that an employee was
contacted or will call back.

Two of the triggers are handled deterministically *before* the model runs —
requests to interpret results or prescribe treatment, and messages suggesting a
medical emergency. Those refusals must not depend on the model choosing to
refuse.

Three more reasons come from the patient backend: a result awaiting clinician
review, a pending result whose expected time has lapsed, and identity checks
that have failed too often in one chat.

---

## Live operation

`GREEN_API_LIVE_ENABLED=true`. The application answers every incoming private
WhatsApp message on the instance — the number is dedicated to the bot and there
is no chat allowlist.

**Done already:** instance authorised, live flag on, webhook token generated in
`.env`, server verified answering on `:8000`.

**Still required — the provider cannot reach this machine yet:**

1. **Expose `:8000` over HTTPS.** GREEN API only calls public HTTPS URLs.
   `ngrok http 8000` for a trial, or a reverse proxy for anything lasting.
2. **In the GREEN API console**, set:
   - `webhookUrl` → `https://<your-host>/webhooks/green-api`
   - `webhookUrlToken` → the value of `GREEN_API_WEBHOOK_TOKEN` in `.env`
   - `incomingWebhook` → `yes`
   Leave the outgoing webhooks off; they are ignored anyway and only add load.
   No script here modifies console settings — do this by hand.
3. **Send a message from a number you control** and watch the logs.

Verify with `uv run python scripts/check_green_api.py` — `webhook configured`
should flip to `yes`.

### Stopping

Set `GREEN_API_LIVE_ENABLED=false` and restart: replies become redacted dry-run
log lines and no request leaves the process. Clearing `webhookUrl` in the
console stops deliveries at the source.

### Reply-loop prevention

Four independent guards, because one bad loop means unbounded message spend:

1. Outgoing and outgoing-API webhooks are ignored.
2. A message whose `sender` or `chatId` equals the instance's own `wid` is
   ignored — that covers the bot's self-chat.
3. Every `idMessage` returned by our own `sendMessage` is remembered; if it
   comes back as an incoming notification it is dropped.
4. Inbound `idMessage` values are de-duplicated for 15 minutes.

---

## Security notes

- `.env` is git-ignored; `.env.example` holds placeholders only. Never put a
  real key in `.env.example`.
- Secrets are wrapped in `Secret`, which refuses to render its value in logs,
  `repr()`, or tracebacks.
- Logs are JSON and pass through a redaction filter: known secret values, GREEN
  API credential paths, bearer tokens, chat ids, and phone numbers are masked.
- Customer message bodies are never logged in full — only a length and a short
  prefix.
- Error responses expose exception *type names* only, never messages that could
  carry a cluster host or request URL.
- Nothing logs a tenant token, a patient verification token, a name, a birth
  date, a phone suffix, an internal patient id, a request body, or a report
  URL. The redaction filter additionally masks the verification header and the
  `/results/{id}/latest` path, but that is a backstop — call sites do not pass
  identity data in the first place, because a name has no regular shape for a
  pattern to catch.
- The tenant scope recorded in log lines is a SHA-256 digest of the backend URL
  and client id, never the client id itself.

---

## Deployment

### Render (personal test deployment, Free plan)

[render.yaml](render.yaml) is a Render Blueprint: Python 3.12 runtime,
`pip install -r requirements.txt`, `uvicorn app.fast_api_app:app --host 0.0.0.0
--port $PORT`, health check on `/healthz`. Secrets are `sync: false` and are
entered in the Render dashboard, never committed.

On Render (which sets `RENDER=true`) or with `PUBLIC_SURFACE_ONLY=true`:

- Only `/healthz` and `/webhooks/green-api` answer. The ADK dev UI, session
  routes, A2A and `/health/integrations` return 404 to everyone, with or
  without Cloudflare headers.
- `GREEN_API_WEBHOOK_TOKEN` is mandatory: without it every webhook gets a 401.
- Configuration problems are logged by variable name at startup.

`GREEN_API_LIVE_ENABLED` starts as `false` in the Blueprint. Flip it in the
dashboard only after the service is healthy and the GREEN API `webhookUrl`
points at `https://<service>.onrender.com/webhooks/green-api`.

Regenerate `requirements.txt` after changing dependencies:

```bash
uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt -o requirements.txt
```

**Free-plan behaviour.** The instance sleeps after ~15 minutes without traffic
and takes up to a minute to wake. Every sleep, deploy or restart wipes all
in-memory state: ADK sessions (conversation history), patient verifications,
staged report releases, escalation records and the de-duplication cache. A
customer mid-conversation starts over, and a verified patient must verify
again. GREEN API redelivers what it could not deliver during the wake-up;
`GREEN_API_MAX_MESSAGE_AGE_SECONDS` drops anything older than 10 minutes.

### Other targets

This project was scaffolded with `--deployment-target none` and
`--prototype`. To add another deployment target:

```bash
agents-cli scaffold enhance . --deployment-target <agent_runtime|cloud_run|gke>
```

Note that the current session storage is in-memory and will not survive more
than one instance.
