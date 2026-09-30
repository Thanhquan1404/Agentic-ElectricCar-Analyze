# tools/vnexpress_detail.py
from __future__ import annotations
import re, json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag, NavigableString
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; VinFast-RAG-Agent/1.0)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Referer": "https://vnexpress.net/",
}

# ---------- ST1: fetch HTML ----------
def _session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )))
    s.headers.update(DEFAULT_HEADERS)
    return s

_SESSION = _session()

def fetch_html(url: str, timeout: int = 15) -> dict:
    """ST1 — Fetch raw HTML."""
    if not url or not url.startswith("http"):
        logger.warning("Fetch HTML failed: Invalid URL '%s'", url)
        return {"ok": False, "error": "invalid_url"}
    try:
        r = _SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        # VNExpress uses utf-8
        r.encoding = r.apparent_encoding or "utf-8"
        return {"ok": True, "html": r.text, "status": r.status_code}
    except requests.exceptions.RequestException as e:
        logger.error("HTTP error while fetching URL '%s': %s", url, e)
        return {"ok": False, "error": f"http_error: {e}"}


# ---------- ST2: parse DOM ----------
def parse_dom(html: str) -> BeautifulSoup:
    """ST2 — Parse HTML into a BeautifulSoup object. Use lxml if available, fallback to html.parser."""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception as e:
        logger.debug("Fallback to html.parser due to: %s", e)
        return BeautifulSoup(html, "html.parser")


# ---------- ST3: extract metadata ----------
def _meta(soup: BeautifulSoup, prop: str) -> Optional[str]:
    tag = soup.find("meta", attrs={"property": prop}) or \
          soup.find("meta", attrs={"name": prop})
    return tag.get("content") if tag and tag.get("content") else None

def extract_metadata(soup: BeautifulSoup) -> dict:
    """ST3 — Extract metadata from <meta>, <h1>, .description, tags, etc."""
    meta = {
        "title": _meta(soup, "og:title"),
        "lead": _meta(soup, "og:description"),
        "og_image": _meta(soup, "og:image"),
        "publish_time": _meta(soup, "article:published_time"),
        "author": _meta(soup, "article:author"),
        "section": _meta(soup, "article:section"),
    }

    # Fallback H1 (title-detail)
    if not meta["title"]:
        h1 = soup.find("h1", class_=re.compile(r"title-detail|title"))
        if h1:
            meta["title"] = h1.get_text(strip=True)

    # Fallback description (lead/sapo)
    if not meta["lead"]:
        p = soup.find("p", class_=re.compile(r"description"))
        if p:
            meta["lead"] = p.get_text(strip=True)

    # Author fallback: <span class="author_mail"> or <p class="author_mail">
    if not meta["author"]:
        a = soup.find(class_=re.compile(r"author_mail"))
        if a:
            meta["author"] = a.get_text(strip=True)

    # Tags: <h4 class="tag"> or <a class="tag"> or <div class="tags">
    tags = []
    tag_box = soup.find(class_=re.compile(r"\btags?\b"))
    if tag_box:
        for a in tag_box.find_all("a", href=True):
            t = a.get_text(strip=True)
            if t and len(t) < 60:
                tags.append(t)
    if not tags:
        for a in soup.select("a.tag, h4.tag a"):
            t = a.get_text(strip=True)
            if t:
                tags.append(t)
    meta["tags"] = list(dict.fromkeys(tags))[:20]  # deduplicate, max 20 tags

    return meta


# ---------- ST4: extract body ----------
# VNExpress: <article class="fck_detail"> containing <p class="Normal">, <figure>, <table>
BODY_SELECTORS = [
    "article.fck_detail",
    "div.fck_detail",
    "article[itemprop='articleBody']",
    "div[itemprop='articleBody']",
    "div#main-detail-body",
]

def extract_body(soup: BeautifulSoup) -> dict:
    """ST4 — Extract article body into paragraphs, figures, and tables."""
    node = None
    for sel in BODY_SELECTORS:
        node = soup.select_one(sel)
        if node:
            break

    if not node:
        logger.warning("Article body selector not found in HTML")
        return {"paragraphs": [], "figures": [], "tables": [], "found": False}

    paragraphs, figures, tables = [], [], []

    for p in node.find_all("p", recursive=True):
        # Exclude paragraphs inside figure (captions) or table
        if p.find_parent("figure") or p.find_parent("table"):
            continue
        txt = p.get_text(" ", strip=True)
        if txt and len(txt) > 10:  # ignore empty or extremely short paragraphs
            paragraphs.append(p)

    for fig in node.find_all("figure", recursive=True):
        img = fig.find("img")
        cap = fig.find("figcaption")
        if img:
            figures.append({
                "src": img.get("data-src") or img.get("src"),
                "alt": img.get("alt", ""),
                "caption": cap.get_text(" ", strip=True) if cap else "",
            })

    for tb in node.find_all("table", recursive=True):
        tables.append(tb.get_text(" | ", strip=True))

    return {
        "paragraphs": paragraphs,   # Keep Tag objects for ST5 cleaning
        "figures": figures,
        "tables": tables,
        "found": True,
    }


# ---------- ST5: clean HTML tags ----------
JUNK_PATTERNS = [
    r"^See also.*", r"^Related news.*", r"^Read more.*", r"^Xem thêm.*", r"^Tin liên quan.*", r"^Đọc tiếp.*",
    r"^Advertisement.*", r"^Quảng cáo.*", r"^Video.*", r"^\s*$",
]

def _clean_tag(el: Tag) -> str:
    """
    Convert Tag to plain text:
    - Remove script/style/iframe/ins/button/form/svg elements
    - Keep <a> as text, append link marker for external links
    - Normalize whitespace
    """
    for bad in el.find_all(["script", "style", "iframe", "ins", "button",
                            "form", "svg", "noscript", "aside"]):
        bad.decompose()

    # Process <a> tags: preserve text; mark non-VNExpress external links with [↗]
    for a in el.find_all("a"):
        txt = a.get_text(strip=True)
        href = a.get("href", "")
        if href and href.startswith("http") and "vnexpress.net" not in href:
            a.replace_with(f"{txt} [↗]")
        else:
            a.replace_with(txt)

    # Convert <br> to whitespace
    for br in el.find_all("br"):
        br.replace_with(" ")

    text = el.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_html_tags(text: str) -> str:
    """ST5 — Clean leftover HTML tags from any raw string."""
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


# ---------- ST6: junk filter ----------
def is_junk(text: str) -> bool:
    """ST6 — Identify junk lines and promotional sentences."""
    if not text or len(text) < 15:
        return True
    for pat in JUNK_PATTERNS:
        if re.match(pat, text, flags=re.IGNORECASE):
            return True
    # Placeholder warnings
    if ("See also" in text or "Xem thêm" in text) and len(text) < 40:
        return True
    return False


# ---------- ST7: assemble ----------
@dataclass
class ArticleDetail:
    article_id: str
    share_url: str
    title: str
    lead: str
    author: Optional[str]
    publish_time: Optional[str]
    tags: list
    content_text: str           # Clean full text joined with "\n\n"
    paragraphs: list            # Clean list[str]
    figures: list               # list[dict]
    word_count: int
    image_count: int
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def vnexpress_fetch_article_detail(
    share_url: str = "",
    article_id: str = "",
    timeout: int = 15,
    verbose: bool = True,
) -> dict:
    """
    Main tool: fetch, parse, and clean full VNExpress article content.

    Returns:
      {
        "ok": bool,
        "article": ArticleDetail dict | None,
        "error": str | None,
        "stages": {"fetch":.., "parse":.., "meta":.., "body":.., "clean":..}
      }
    """
    stages: dict = {}

    # ST1
    if not share_url:
        logger.error("Fetch article detail failed: share_url is required")
        return {"ok": False, "error": "share_url_required (article_id-only not supported)"}
    fetch = fetch_html(share_url, timeout=timeout)
    stages["fetch"] = fetch.get("ok", False)
    if not fetch["ok"]:
        return {"ok": False, "error": fetch["error"], "stages": stages}

    # ST2
    soup = parse_dom(fetch["html"])
    stages["parse"] = True

    # ST3
    meta = extract_metadata(soup)
    stages["meta"] = sum(1 for v in meta.values() if v) >= 3

    # ST4
    body = extract_body(soup)
    stages["body"] = body["found"] and len(body["paragraphs"]) >= 1
    if not stages["body"]:
        logger.warning("Could not locate article body for URL: %s", share_url)
        return {"ok": False, "error": "body_not_found", "stages": stages,
                "meta": meta}

    # ST5 + ST6
    clean_paras: list[str] = []
    for p in body["paragraphs"]:
        txt = _clean_tag(p)
        if txt and not is_junk(txt):
            clean_paras.append(txt)

    # If body extracted but cleaned list is empty, flag stage for debugging selectors
    stages["clean"] = len(clean_paras) > 0

    article = ArticleDetail(
        article_id=article_id or _id_from_url(share_url),
        share_url=share_url,
        title=clean_html_tags(meta.get("title") or ""),
        lead=clean_html_tags(meta.get("lead") or ""),
        author=clean_html_tags(meta.get("author") or "") or None,
        publish_time=meta.get("publish_time"),
        tags=meta.get("tags", []),
        content_text="\n\n".join(clean_paras),
        paragraphs=clean_paras,
        figures=body["figures"],
        word_count=sum(len(t.split()) for t in clean_paras),
        image_count=len(body["figures"]),
    )

    result = {
        "ok": True,
        "article": article.to_dict(),
        "error": None,
        "stages": stages,
    }

    if verbose:
        _pretty_print(result["article"])
    return result


# ---------- helpers ----------
def _id_from_url(url: str) -> str:
    """Extract article_id from URL formatted like ...-5125632.html"""
    m = re.search(r"-(\d{6,})\.html?$", urlparse(url).path)
    return m.group(1) if m else ""


def _pretty_print(a: dict) -> None:
    print("=" * 78)
    print(f"Title: {a['title']}")
    print(f"Share URL: {a['share_url']}")
    print(f"Author: {a.get('author') or '—'}  |  Publish Time: {a.get('publish_time') or '—'}")
    print(f"Tags: {', '.join(a['tags']) if a['tags'] else '—'}")
    print("-" * 78)
    print(f"LEAD: {a['lead']}")
    print("-" * 78)
    for i, p in enumerate(a["paragraphs"], 1):
        print(f"[{i:02d}] {p}")
    if a["figures"]:
        print("-" * 78)
        print(f"Images: {a['image_count']} images:")
        for f in a["figures"][:5]:
            print(f"   • {f['caption'] or f['alt'] or f['src']}")
    print("-" * 78)
    print(f"Statistic {len(a['paragraphs'])} paragraphs | {a['word_count']} words | "
          f"{a['image_count']} images | fetched_at={a['fetched_at']}")
    print("=" * 78)