import time
from unittest.mock import patch

import pytest

import api.auth as auth_module
from api.auth import SESSION_TTL, make_session_cookie, verify_session_cookie


@pytest.fixture(autouse=True)
def set_secret(monkeypatch):
    monkeypatch.setattr(auth_module, "ADMIN_SECRET_KEY", "testsecret123")


class TestVerifySessionCookie:
    def test_fresh_valid_token(self):
        assert verify_session_cookie(make_session_cookie()) is True

    def test_none_returns_false(self):
        assert verify_session_cookie(None) is False

    def test_empty_string_returns_false(self):
        assert verify_session_cookie("") is False

    def test_malformed_no_dot_returns_false(self):
        assert verify_session_cookie("notavalidtoken") is False

    def test_non_hex_timestamp_returns_false(self):
        assert verify_session_cookie("gggg.abcd1234") is False

    def test_tampered_signature_returns_false(self):
        token = make_session_cookie()
        ts, _ = token.split(".", 1)
        assert verify_session_cookie(f"{ts}.{'a' * 64}") is False

    def test_expired_token_returns_false(self):
        past = int(time.time()) - SESSION_TTL - 1
        ts = format(past, "x")
        sig = auth_module._sign(ts)
        assert verify_session_cookie(f"{ts}.{sig}") is False

    def test_token_at_boundary_is_valid(self):
        # One second before expiry should still be valid
        past = int(time.time()) - SESSION_TTL + 1
        ts = format(past, "x")
        sig = auth_module._sign(ts)
        assert verify_session_cookie(f"{ts}.{sig}") is True

    def test_no_secret_key_returns_false(self, monkeypatch):
        monkeypatch.setattr(auth_module, "ADMIN_SECRET_KEY", "")
        assert verify_session_cookie(make_session_cookie()) is False
