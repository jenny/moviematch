from slowapi import Limiter
from slowapi.util import get_remote_address

from api.geo import get_client_ip


def _client_key(request) -> str:
    """Rate-limit key: the real client IP.

    slowapi's default `get_remote_address` reads `request.client.host`, which
    behind a reverse proxy (Railway) is the *proxy's* IP — identical for every
    visitor, so all clients would share a single rate-limit bucket. We reuse
    `get_client_ip` (the same X-Forwarded-For parsing used for geo/logging) so
    the limit is applied per real client, falling back to the direct peer when
    no forwarding header is present.
    """
    return get_client_ip(request) or get_remote_address(request)


limiter = Limiter(key_func=_client_key)
