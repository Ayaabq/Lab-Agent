"""State shared across serving instances.

On one long-lived process (local, Railway) every store keeps its state in
memory. On a serverless host successive webhooks may reach different
instances, so the same stores are backed by Upstash Redis instead, reached
over HTTPS with ``httpx`` — see :mod:`app.state.backend` for which one a
process gets.

Two rules hold for everything written to Redis:

* **Values are encrypted** (Fernet). A verification token, a conversation
  transcript or an escalation record is unreadable without
  ``STATE_ENCRYPTION_KEY``, which never lives in Redis.
* **Key names are keyed hashes.** A chat id is a phone number; it becomes an
  HMAC before it becomes part of a key, so the keyspace itself discloses
  nothing.

And one for every caller: :class:`~app.state.upstash.StateUnavailable` means
"could not confirm", never "proceed". Each call site decides what the safe
outcome is — usually "not verified", "do not send the report" or the fallback
reply.
"""
