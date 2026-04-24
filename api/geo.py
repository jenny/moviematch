import ipaddress
import logging

import httpx
from fastapi import Request

logger = logging.getLogger(__name__)

# Process-lifetime cache — each unique public IP is resolved at most once.
_country_cache: dict[str, str] = {}


def get_client_ip(request: Request) -> str | None:
    """Extract the real client IP from the request.

    Railway (and most reverse proxies) set X-Forwarded-For; fall back to the
    direct connection address for local development.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def resolve_country(ip: str) -> str:
    """Return the ISO 3166-1 alpha-2 country code for a public IP via ipinfo.io.

    Returns "US" for private/loopback addresses or on any lookup failure so that
    streaming lookups always have a valid region — never blocks a response.
    Uses ipinfo.io free tier (no key required, 50k requests/month).
    """
    if ip in _country_cache:
        logger.debug("[geo] resolved %s -> %s (cache_hit=True)", ip, _country_cache[ip])
        return _country_cache[ip]

    # Skip private/loopback addresses without a network call.
    try:
        addr = ipaddress.ip_address(ip)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            logger.debug("[geo] skipping private/loopback ip %s", ip)
            _country_cache[ip] = "US"
            return "US"
    except ValueError:
        _country_cache[ip] = "US"
        return "US"

    try:
        resp = httpx.get(f"https://ipinfo.io/{ip}/json", timeout=2.0)
        if resp.status_code == 200:
            country = resp.json().get("country") or "US"
            _country_cache[ip] = country
            logger.debug("[geo] resolved %s -> %s (cache_hit=False)", ip, country)
            return country
    except Exception as exc:
        logger.warning("[geo] ipinfo lookup failed for %s: %s", ip, exc)

    return "US"
