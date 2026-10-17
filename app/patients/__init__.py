"""Patient identity verification and laboratory-result retrieval.

Everything in this package is **server-side infrastructure**. The model sees
only the two tool wrappers in :mod:`app.tools`, and those return a safe
outcome — never a tenant token, a patient verification token, an internal
patient id, a raw backend response, or an exception.

Module map:

* :mod:`app.patients.context` — the trusted conversation identity a tool runs
  under. Never supplied by the model.
* :mod:`app.patients.normalize` — Arabic-digit, date and phone-suffix parsing.
* :mod:`app.patients.auth` — cached tenant login, scoped to one tenant.
* :mod:`app.patients.store` — verification state, scoped to tenant + chat +
  ADK session, plus the per-chat failed-attempt limit.
* :mod:`app.patients.results` — disposition handling and the (inert) future
  result-image extension point.
* :mod:`app.patients.client` — the bounded HTTP client for the backend.
"""
