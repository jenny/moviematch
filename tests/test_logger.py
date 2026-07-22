from unittest.mock import patch

from logger import redact


class TestRedact:
    def test_replaces_configured_secret(self):
        with patch("config.WATCHMODE_API_KEY", "wm_s3cret_key_value"):
            out = redact("failed for url: https://api.watchmode.com/?apiKey=wm_s3cret_key_value")
        assert "wm_s3cret_key_value" not in out
        assert "***" in out

    def test_replaces_every_occurrence(self):
        with patch("config.OMDB_API_KEY", "omdb_s3cret_value"):
            out = redact("omdb_s3cret_value and again omdb_s3cret_value")
        assert "omdb_s3cret_value" not in out
        assert out.count("***") == 2

    def test_scrubs_multiple_distinct_secrets(self):
        with patch("config.WATCHMODE_API_KEY", "wm_s3cret_key_value"), \
             patch("config.OMDB_API_KEY", "omdb_s3cret_value"):
            out = redact("wm_s3cret_key_value / omdb_s3cret_value")
        assert "s3cret" not in out

    def test_accepts_non_string_values(self):
        """Call sites pass exception objects, not strings."""
        with patch("config.OMDB_API_KEY", "omdb_s3cret_value"):
            out = redact(ValueError("boom omdb_s3cret_value"))
        assert out == "boom ***"

    def test_empty_secret_does_not_mangle_message(self):
        """str.replace('', x) interleaves x between every character — must be guarded."""
        with patch("config.OMDB_API_KEY", ""), patch("config.WATCHMODE_API_KEY", None):
            assert redact("a normal message") == "a normal message"

    def test_short_secret_is_not_blind_replaced(self):
        """A too-short value would riddle unrelated text with ***."""
        with patch("config.OMDB_API_KEY", "abc"):
            assert redact("abcdefg is fine") == "abcdefg is fine"

    def test_unset_secrets_are_skipped(self):
        with patch("config.OMDB_API_KEY", None), patch("config.WATCHMODE_API_KEY", None):
            assert redact("nothing to scrub") == "nothing to scrub"
