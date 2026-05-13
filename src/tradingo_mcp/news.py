"""RSS + economic calendar news tools."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

_RSS_FEEDS = [
    ("Reuters Markets", "https://feeds.reuters.com/reuters/businessNews"),
    ("FT Markets", "https://www.ft.com/rss/home/uk"),
    ("Investing.com News", "https://www.investing.com/rss/news.rss"),
]

_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def fetch(query: str, lookback_days: int = 7, limit: int = 20) -> list[dict[str, Any]]:
    """Fetch news from RSS feeds matching query."""
    try:
        import feedparser
    except ImportError:
        return [{"error": "feedparser not installed"}]

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
    results: list[dict[str, Any]] = []
    query_lower = query.lower()

    for source_name, url in _RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                if (
                    query_lower not in title.lower()
                    and query_lower not in summary.lower()
                ):
                    continue
                published = entry.get("published_parsed")
                if published:
                    pub_dt = datetime(  # noqa: DTZ001
                        published[0],
                        published[1],
                        published[2],
                        published[3],
                        published[4],
                        published[5],
                    ).replace(tzinfo=timezone.utc)
                    if pub_dt < cutoff:
                        continue
                results.append(
                    {
                        "source": source_name,
                        "title": title,
                        "summary": summary[:500],
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                    }
                )
                if len(results) >= limit:
                    return results
        except Exception:
            continue

    return results


def calendar(
    lookback_days: int = 3,
    lookahead_days: int = 7,
    currencies: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch economic calendar events."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(_CALENDAR_URL)
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
