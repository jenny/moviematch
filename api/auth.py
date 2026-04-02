"""
Admin session cookie authentication.

Cookies are signed with HMAC-SHA256 keyed on ADMIN_SECRET_KEY. The token
payload is the Unix timestamp (hex) at which the session was issued. On every
authenticated request the cookie is reissued with the current timestamp,
implementing a 15-minute sliding inactivity window.

Token format: "<timestamp_hex>.<hmac_sha256_hex>"
"""

import hashlib
import hmac
import os
import secrets
import time
from typing import Optional

from fastapi import HTTPException, Request, Response

from config import ADMIN_SECRET_KEY

SESSION_COOKIE = "admin_session"
SESSION_TTL = 15 * 60  # seconds


def _sign(timestamp_hex: str) -> str:
    return hmac.new(
        ADMIN_SECRET_KEY.encode(),
        timestamp_hex.encode(),
        digestmod=hashlib.sha256,
    ).hexdigest()


def make_session_cookie() -> str:
    """Return a fresh signed session token encoding the current time."""
    ts = format(int(time.time()), "x")
    return f"{ts}.{_sign(ts)}"


def verify_session_cookie(token: Optional[str]) -> bool:
    """
    Return True iff the token has a valid signature and is within SESSION_TTL.
    Both checks always run (constant-time) to prevent timing oracle attacks.
    """
    if not token or not ADMIN_SECRET_KEY:
        return False
    parts = token.split(".", 1)
    if len(parts) != 2:
        return False
    timestamp_hex, provided_sig = parts
    try:
        timestamp = int(timestamp_hex, 16)
    except ValueError:
        return False
    expected_sig = _sign(timestamp_hex)
    sig_ok = secrets.compare_digest(provided_sig, expected_sig)
    age_ok = time.time() - timestamp <= SESSION_TTL
    return sig_ok and age_ok


def _cookie_kwargs() -> dict:
    """Security flags for the session cookie. Secure=True only on Railway (HTTPS)."""
    return dict(
        httponly=True,
        samesite="strict",
        secure=bool(os.getenv("RAILWAY_ENVIRONMENT")),
        max_age=SESSION_TTL,
    )


def set_session_cookie(response: Response) -> None:
    """Write a fresh session cookie onto response (resets the inactivity timer)."""
    response.set_cookie(SESSION_COOKIE, make_session_cookie(), **_cookie_kwargs())


def require_admin(request: Request, response: Response) -> None:
    """
    FastAPI dependency for admin API endpoints.
    Raises HTTP 401 if the session cookie is absent or invalid.
    Refreshes the cookie on success to implement a sliding inactivity window.
    """
    if not verify_session_cookie(request.cookies.get(SESSION_COOKIE)):
        raise HTTPException(status_code=401, detail="Unauthorized")
    set_session_cookie(response)
