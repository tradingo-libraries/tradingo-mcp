"""News and economic calendar tools backed by Miniflux."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def _client() -> Any:
    import miniflux

    base_url = os.environ.get("MINIFLUX_BASE_URL", "http://miniflux:8084")
    api_key = os.environ.get("MINIFLUX_API_KEY")
    username = os.environ.get("MINIFLUX_USERNAME")
    password = os.environ.get("MINIFLUX_PASSWORD")

    if api_key:
        return miniflux.Client(base_url, api_key=api_key)
    if username and password:
        return miniflux.Client(base_url, username=username, password=password)
    raise RuntimeError(
        "Set MINIFLUX_API_KEY or MINIFLUX_USERNAME/MINIFLUX_PASSWORD to use news tools."
    )


def fetch(query: str, lookback_days: int = 7, limit: int = 20) -> list[dict[str, Any]]:
    """Search Miniflux entries matching query, published within lookback_days."""
    client = _client()
    after_ts = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).timestamp()
    )
    resp = client.get_entries(
        search=query,
        limit=limit,
        direction="desc",
        order="published_at",
        published_after=after_ts,
    )
    entries = resp.get("entries") or []
    return [
        {
            "source": e.get("feed", {}).get("title", ""),
            "title": e.get("title", ""),
            "summary": (e.get("content") or "")[:500],
            "link": e.get("url", ""),
            "published": e.get("published_at", ""),
        }
        for e in entries
    ]


def list_feeds() -> list[dict[str, Any]]:
    """List all RSS feeds configured in Miniflux."""
    client = _client()
    feeds = client.get_feeds()
    return [
        {
            "id": f.get("id"),
            "title": f.get("title", ""),
            "site_url": f.get("site_url", ""),
            "feed_url": f.get("feed_url", ""),
            "category": f.get("category", {}).get("title", ""),
            "last_checked": f.get("checked_at", ""),
            "error": f.get("parsing_error_message", ""),
        }
        for f in (feeds or [])
    ]


def add_feed(feed_url: str, category_id: int | None = None) -> dict[str, Any]:
    """Add a new RSS feed to Miniflux."""
    client = _client()
    kwargs: dict[str, Any] = {"feed_url": feed_url}
    if category_id is not None:
        kwargs["category_id"] = category_id
    feed = client.create_feed(**kwargs)
    return {"id": feed, "feed_url": feed_url}


def remove_feed(feed_id: int) -> dict[str, Any]:
    """Remove a feed from Miniflux by ID."""
    client = _client()
    client.delete_feed(feed_id)
    return {"deleted": feed_id}


def refresh_feeds(feed_id: int | None = None) -> dict[str, Any]:
    """Trigger a feed refresh. Refreshes one feed if feed_id given, all otherwise."""
    client = _client()
    if feed_id is not None:
        client.refresh_feed(feed_id)
        return {"refreshed": feed_id}
    client.refresh_all_feeds()
    return {"refreshed": "all"}


def calendar(
    lookback_days: int = 3,
    lookahead_days: int = 7,
    currencies: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch economic calendar events."""
    try:
        with httpx.Client(timeout=10) as http:
            resp = http.get(_CALENDAR_URL)
            resp.raise_for_status()
            events = resp.json()
    except Exception as e:
        return [{"error": str(e)}]

    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=lookback_days)
    end = now + timedelta(days=lookahead_days)

    result = []
    for evt in events:
        try:
            dt = datetime.fromisoformat(evt.get("date", "").replace("Z", "+00:00"))
        except Exception:
            continue
        if not (start <= dt <= end):
            continue
        if currencies and evt.get("country", "").upper() not in [
            c.upper() for c in currencies
        ]:
            continue
        result.append(
            {
                "date": dt.isoformat(),
                "country": evt.get("country", ""),
                "event": evt.get("title", ""),
                "impact": evt.get("impact", ""),
                "forecast": evt.get("forecast", ""),
                "previous": evt.get("previous", ""),
            }
        )

    return result
