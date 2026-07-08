"""News ingestion from the official RSS feeds of reliable outlets.

Feeds are fetched concurrently, parsed with feedparser, and cached. A feed
that errors simply contributes nothing for that refresh cycle, so one dead
endpoint never takes the app down.
"""

import calendar
import html
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import feedparser
import requests
import yfinance as yf

from ..config import CACHE_TTL, MACRO_FEEDS
from .cache import cached
from .market import UA

_TAG_RE = re.compile(r"<[^>]+>")
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|corp|corporation|company|co|plc|ltd|limited|holdings?|"
    r"group|technologies|technology|international|sa|nv|ag|se)\.?$",
    re.I,
)


def _clean_company(name):
    """'Apple Inc.' -> 'Apple', so headline matching works on the common name."""
    name = re.sub(r"[,.()]", " ", name or "").strip()
    prev = None
    while prev != name:
        prev = name
        name = _COMPANY_SUFFIX_RE.sub("", name).strip()
    return name


def _fetch_feed(url, source, category):
    def fetch():
        try:
            resp = requests.get(url, headers={"User-Agent": UA}, timeout=10)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
        except Exception:
            return []
        items = []
        for entry in parsed.entries[:40]:
            title = html.unescape(getattr(entry, "title", "") or "").strip()
            if not title:
                continue
            published = None
            for key in ("published_parsed", "updated_parsed"):
                struct = entry.get(key)
                if struct:
                    published = calendar.timegm(struct)  # feedparser structs are UTC
                    break
            summary = html.unescape(_TAG_RE.sub(" ", getattr(entry, "summary", "") or ""))
            items.append({
                "title": title,
                "summary": re.sub(r"\s+", " ", summary).strip()[:300],
                "link": getattr(entry, "link", "") or "",
                "source": source,
                "category": category,
                "published": published,
            })
        return items

    return cached(("feed", url), CACHE_TTL["feed"], fetch) or []


def get_macro_news():
    """All business/politics/world items from the configured outlets, newest first."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = list(pool.map(
            lambda f: _fetch_feed(f["url"], f["source"], f["category"]),
            MACRO_FEEDS,
        ))
    items = [item for batch in batches for item in batch]
    items.sort(key=lambda a: a["published"] or 0, reverse=True)
    return items


def _yahoo_ticker_news(ticker):
    """Per-ticker headlines from Yahoo Finance's news aggregation (Reuters, AP,
    Bloomberg, Barron's etc.), via yfinance. Handles both the nested `content`
    schema (yfinance >= 0.2.50) and the older flat one."""
    def fetch():
        try:
            raw = yf.Ticker(ticker).news or []
        except Exception:
            return []
        items = []
        for entry in raw[:20]:
            content = entry.get("content") or entry  # nested vs. flat schema
            title = (content.get("title") or "").strip()
            if not title:
                continue
            published = None
            pub = content.get("pubDate") or content.get("displayTime")
            if pub:
                try:
                    published = datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
                except (ValueError, TypeError):
                    published = None
            elif content.get("providerPublishTime"):
                published = float(content["providerPublishTime"])
            provider = (content.get("provider") or {}).get("displayName") or entry.get("publisher") or "Yahoo Finance"
            link = ((content.get("canonicalUrl") or {}).get("url")
                    or (content.get("clickThroughUrl") or {}).get("url")
                    or entry.get("link") or "")
            items.append({
                "title": html.unescape(title),
                "summary": html.unescape((content.get("summary") or content.get("description") or "").strip())[:300],
                "link": link,
                "source": provider,
                "category": "company",
                "published": published,
            })
        return items

    return cached(("ynews", ticker), CACHE_TTL["feed"], fetch) or []


def get_ticker_news(ticker, company_name):
    """Company-specific headlines: Yahoo Finance's per-ticker news, plus any
    mention of the company in the general reliable-outlet feeds."""
    ticker = ticker.upper()
    items = list(_yahoo_ticker_news(ticker))

    # Match on the company's common name, case-sensitively and on word
    # boundaries, in the headline only — "Apple" must not match "candy apples".
    name = _clean_company(company_name)
    name_re = re.compile(rf"\b{re.escape(name)}\b") if len(name) >= 3 else None
    ticker_re = re.compile(rf"\b{re.escape(ticker)}\b") if len(ticker) >= 2 else None

    for article in get_macro_news():
        title = article["title"]
        if (name_re and name_re.search(title)) or (ticker_re and ticker_re.search(title)):
            items.append({**article, "category": "company"})

    seen, unique = set(), []
    for article in items:
        key = article["title"].lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
    unique.sort(key=lambda a: a["published"] or 0, reverse=True)
    return unique[:30]
