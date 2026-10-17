"""Local smoke test for the Khatib patient/results backend.

Three calls, in the order a real conversation makes them:

    tenant login -> patient verification -> latest result

It reads **one** designated seed/test identity from the environment and never
searches, lists or iterates. There is no way to make this script enumerate
patients, and that is deliberate: a connectivity check must not become a way
to find out who is in the database.

Output is sanitised. No token, no patient id, no name, no birth date, no phone
suffix and no report URL is printed — only whether each step worked and which
case the result landed in.

    uv run python scripts/check_khatib_backend.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from app.config import get_settings  # noqa: E402
from app.patients.auth import KhatibAuthError, TenantAuthenticator  # noqa: E402
from app.patients.client import KhatibApiClient, VerifyStatus  # noqa: E402
from app.patients.normalize import (  # noqa: E402
    FieldStatus,
    normalize_date_of_birth,
    normalize_full_name,
    normalize_phone_last4,
)
from app.patients.results import (  # noqa: E402
    ImageRejection,
    image_rejection,
    parse_latest_result,
    report_file_name,
)

SEED_NAME_VAR = "KHATIB_SMOKE_FULL_NAME"
SEED_DOB_VAR = "KHATIB_SMOKE_DATE_OF_BIRTH"
SEED_LAST4_VAR = "KHATIB_SMOKE_PHONE_LAST4"


def _missing_configuration(khatib) -> list[str]:
    missing = []
    if not khatib.base_url:
        missing.append("KHATIB_API_BASE_URL")
    if not khatib.client_id.is_set:
        missing.append("KHATIB_CLIENT_ID")
    if not khatib.client_secret.is_set:
        missing.append("KHATIB_CLIENT_SECRET")
    return missing


async def main() -> int:
    khatib = get_settings().khatib

    missing = _missing_configuration(khatib)
    if missing:
        print("backend not configured. Missing: " + ", ".join(missing))
        return 1

    print(f"backend: {khatib.base_url}")
    print(f"tenant scope: {khatib.tenant_key}")

    authenticator = TenantAuthenticator(khatib)
    client = KhatibApiClient(khatib, authenticator=authenticator)
    try:
        # 1. Tenant login.
        try:
            await authenticator.get_token()
        except KhatibAuthError as exc:
            # The message carries a status or an exception type, never a value.
            print(f"[1/3] tenant login FAILED: {exc}")
            return 2
        print("[1/3] tenant login: ok")

        # 2. Patient verification, for the one configured identity.
        identity = _seed_identity()
        if identity is None:
            print(
                f"[2/3] skipped — no seed identity configured.\n"
                f"      Set {SEED_NAME_VAR} plus {SEED_DOB_VAR} or "
                f"{SEED_LAST4_VAR} to a designated test patient.\n"
                f"      Never point these at a real patient."
            )
            return 0
        name, dob, last4 = identity
        method = "dateOfBirth" if dob else "phoneLast4"
        print(f"[2/3] verifying the seed identity by {method}…")

        outcome = await client.verify_patient(
            full_name=name, date_of_birth=dob, phone_last4=last4
        )
        if not outcome.ok:
            print(f"[2/3] verification did not succeed: {outcome.status.value}")
            if outcome.status is VerifyStatus.NOT_VERIFIED:
                print(
                    "      The backend did not accept this identity. Check the "
                    "seed record with the backend owner — do not try other "
                    "names."
                )
            return 3
        print(
            "[2/3] verification: ok "
            f"(method={outcome.method or 'unspecified'}, "
            f"valid for {outcome.expires_in_seconds}s)"
        )

        # 3. Latest result, using only what step 2 returned.
        result_outcome = await client.get_latest_result(
            patient_id=outcome.patient_id,
            verification_token=outcome.verification_token,
        )
        if not result_outcome.ok:
            print(f"[3/3] latest result: {result_outcome.status.value}")
            return 0 if result_outcome.status.value == "no_result" else 4

        result = parse_latest_result(result_outcome.payload, settings=khatib)
        print(
            "[3/3] latest result: ok — "
            f"status={result.status.value}, "
            f"disposition={result.raw_disposition or 'none'}, "
            f"case={result.case.value}"
        )
        # The message itself is patient-specific text; only its presence and
        # size are reported.
        print(
            "      patientMessage: "
            + (
                f"present ({len(result.patient_message)} chars)"
                if result.patient_message
                else "absent"
            )
        )
        # Never prints the URL itself — only whether it would be released
        # and, if not, which check stopped it.
        if result.image is None:
            print(f"      report image: no '{khatib.result_image_field}' field")
        else:
            reason = image_rejection(
                status=result.status,
                raw_disposition=result.raw_disposition,
                image=result.image,
                settings=khatib,
            )
            if reason is ImageRejection.NONE:
                print(
                    "      report image: RELEASABLE as "
                    f"{report_file_name(result.image.url)}"
                )
                print(
                    "      NOTE: GREEN API fetches this URL from its own "
                    "servers. If the backend requires our bearer token to "
                    "serve it, delivery will fail — that needs a signed URL "
                    "from the backend, not a public bucket."
                )
            else:
                print(f"      report image: withheld ({reason.value})")
        return 0
    finally:
        await client.aclose()


def _seed_identity() -> tuple[str, str, str] | None:
    """Read and validate the one designated test identity, or return None."""
    name = normalize_full_name(os.getenv(SEED_NAME_VAR))
    if not name.ok:
        return None

    dob = normalize_date_of_birth(os.getenv(SEED_DOB_VAR))
    last4 = normalize_phone_last4(os.getenv(SEED_LAST4_VAR))
    for label, field in ((SEED_DOB_VAR, dob), (SEED_LAST4_VAR, last4)):
        if field.status in {FieldStatus.INVALID, FieldStatus.AMBIGUOUS}:
            print(f"      ({label} is set but not usable: {field.status.value})")

    if not dob.ok and not last4.ok:
        return None
    return name.value, dob.value, last4.value


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
