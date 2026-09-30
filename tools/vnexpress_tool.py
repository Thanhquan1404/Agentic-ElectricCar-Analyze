# tools/vnexpress_tool.py

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ---------- Configuration ----------

VNEXPRESS_TAX_ENDPOINT = "https://gw.vnexpress.net/tax"

VINFAST_TAX_ID = 1798

DEFAULT_DATA_SELECT = (
    "article_id,article_type,title,share_url,"
    "thumbnail_url,lead,publish_time,update_time"
)

DEFAULT_HEADERS = {
    "User-Agent": "VinFast-RAG-Agent/1.0 (+contact: your-email@example.com)",
    "Accept": "application/json",
    "Referer": "https://vnexpress.net/",
}


# ---------- Data model ----------

@dataclass
class Article:
    """Normalized VNExpress article representation."""

    article_id: str
    title: str
    lead: str
    share_url: str
    thumbnail_url: Optional[str] = None
    article_type: Optional[str] = None
    publish_time: Optional[str] = None
    update_time: Optional[str] = None
    source: str = "vnexpress"
    tax_name: str = "VinFast"
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    content_hash: str = ""

    def __post_init__(self) -> None:
        """Generate a deterministic content hash for the normalized article."""

        raw = f"{self.article_id}|{self.title}|{self.lead}"
        self.content_hash = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict:
        """Convert the article dataclass into a dictionary."""

        return asdict(self)


# ---------- HTTP session with retry/backoff ----------

def _build_session() -> requests.Session:
    """
    Build a reusable HTTP session with retry and exponential backoff.

    Retries are enabled for rate limiting and common transient server errors.
    """

    logger.debug("Building VNExpress HTTP session")

    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(max_retries=retries),
    )
    session.headers.update(DEFAULT_HEADERS)

    logger.debug(
        "HTTP session configured: retries=%d backoff_factor=%.1f "
        "status_forcelist=%s",
        retries.total,
        retries.backoff_factor,
        retries.status_forcelist,
    )

    return session


_SESSION = _build_session()


# ---------- Tool 1: Fetch article list ----------

def vnexpress_fetch_vinfast_articles(
    limit: int = 20,
    data_select: str = DEFAULT_DATA_SELECT,
    thumb_size: str = "280x168",
    thumb_quality: int = 100,
    timeout: int = 10,
) -> dict:
    """
    Fetch the latest VinFast-related articles from the VNExpress API.

    Returns:
        {
            "ok": bool,
            "count": int,
            "articles": [Article dict],
            "raw": <original response>,
            "error": str | None
        }
    """

    logger.info(
        "Starting VNExpress article fetch: tax_id=%d limit=%d "
        "timeout=%ds thumb_size=%s thumb_quality=%d",
        VINFAST_TAX_ID,
        limit,
        timeout,
        thumb_size,
        thumb_quality,
    )

    # Validate request parameters before building the API request.
    if not (1 <= limit <= 50):
        logger.warning(
            "Invalid article limit: limit=%d expected_range=[1,50]",
            limit,
        )

        return {
            "ok": False,
            "error": f"limit must be in [1,50], received {limit}",
            "articles": [],
        }

    params = {
        "tax_id": VINFAST_TAX_ID,
        "limit": limit,
        "data_select": data_select,
        "thumb_size": thumb_size,
        "thumb_quality": thumb_quality,
        "thumb_dpr": "1,2",
        "thumb_fit": "crop",
    }

    url = f"{VNEXPRESS_TAX_ENDPOINT}?{urlencode(params)}"

    logger.debug(
        "VNExpress request prepared: endpoint=%s tax_id=%d "
        "data_select=%s",
        VNEXPRESS_TAX_ENDPOINT,
        VINFAST_TAX_ID,
        data_select,
    )

    start_time = time.perf_counter()

    try:
        logger.debug("Sending GET request to VNExpress")

        response = _SESSION.get(
            url,
            timeout=timeout,
        )

        latency_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        logger.debug(
            "VNExpress response received: status=%d latency=%dms "
            "content_length=%d",
            response.status_code,
            latency_ms,
            len(response.content),
        )

        response.raise_for_status()

        payload = response.json()

        logger.debug(
            "VNExpress JSON payload parsed successfully: payload_type=%s",
            type(payload).__name__,
        )

    except requests.exceptions.RequestException as exc:
        latency_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        logger.error(
            "VNExpress request failed: latency=%dms error=%s",
            latency_ms,
            exc,
        )

        return {
            "ok": False,
            "error": f"http_error: {exc}",
            "articles": [],
        }

    except ValueError as exc:
        latency_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        logger.error(
            "VNExpress response contains invalid JSON: "
            "latency=%dms error=%s",
            latency_ms,
            exc,
        )

        return {
            "ok": False,
            "error": f"invalid_json: {exc}",
            "articles": [],
        }

    # Extract articles while supporting multiple VNExpress response schemas.
    articles_raw = _extract_articles(payload)

    if articles_raw is None:
        logger.error(
            "VNExpress response schema mismatch: "
            "unable to locate article collection for tax_id=%d",
            VINFAST_TAX_ID,
        )

        return {
            "ok": False,
            "error": "schema_mismatch: data[tax_id].articles.data not found",
            "raw": payload,
            "articles": [],
        }

    logger.debug(
        "Raw article collection extracted: count=%d",
        len(articles_raw),
    )

    normalization_result = vinfast_normalize_articles(
        {"articles": articles_raw}
    )

    normalized = normalization_result["articles"]

    latency_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    logger.info(
        "VNExpress article fetch completed successfully: "
        "raw_count=%d normalized_count=%d latency=%dms",
        len(articles_raw),
        len(normalized),
        latency_ms,
    )

    return {
        "ok": True,
        "count": len(normalized),
        "articles": normalized,
        "latency_ms": latency_ms,
        "raw": payload,
        "error": None,
    }


# ---------- Helper: Flexible article extraction ----------

def _extract_articles(payload: Any) -> Optional[list]:
    """
    Extract the article list from supported VNExpress response schemas.

    Supported structures include:
        data["1798"]["articles"]["data"]
        data["1798"]["articles"]
        data["articles"]["data"]
    """

    logger.debug(
        "Attempting to extract articles from VNExpress payload: "
        "payload_type=%s",
        type(payload).__name__,
    )

    if not isinstance(payload, dict):
        logger.warning(
            "Unable to extract articles: payload is not a dictionary"
        )
        return None

    data = payload.get("data")

    if not isinstance(data, dict):
        logger.warning(
            "Unable to extract articles: payload['data'] is not a dictionary"
        )
        return None

    # Variant 1: data["1798"].articles.data
    node = data.get(str(VINFAST_TAX_ID)) or data.get(VINFAST_TAX_ID)

    if isinstance(node, dict):
        articles = node.get("articles")

        if isinstance(articles, dict):
            items = articles.get("data")

            if isinstance(items, list):
                logger.debug(
                    "Article collection extracted using schema variant 1: "
                    "data[%d].articles.data count=%d",
                    VINFAST_TAX_ID,
                    len(items),
                )
                return items

        if isinstance(articles, list):
            logger.debug(
                "Article collection extracted using schema variant 1: "
                "data[%d].articles count=%d",
                VINFAST_TAX_ID,
                len(articles),
            )
            return articles

    # Variant 2: data.articles.data
    articles = data.get("articles")

    if isinstance(articles, dict) and isinstance(
        articles.get("data"),
        list,
    ):
        items = articles["data"]

        logger.debug(
            "Article collection extracted using fallback schema: "
            "data.articles.data count=%d",
            len(items),
        )

        return items

    logger.warning(
        "No supported article collection found in VNExpress payload"
    )

    return None


# ---------- Tool 2: Normalize articles ----------

def vinfast_normalize_articles(raw_response: dict) -> dict:
    """
    Normalize raw VNExpress article records into Article dictionaries.

    Supported input formats:
        {"articles": [{...}, {...}]}
        {"raw": {...}}
    """

    logger.info("Starting VNExpress article normalization")

    items = raw_response.get("articles")

    if items is None and "raw" in raw_response:
        logger.debug(
            "No direct article list provided; extracting articles "
            "from raw response"
        )

        items = _extract_articles(
            raw_response["raw"]
        ) or []

    if items is None:
        items = []

    logger.debug(
        "Normalization input received: item_count=%d",
        len(items),
    )

    normalized: list[dict] = []
    seen_ids: set[str] = set()

    skipped_invalid = 0
    skipped_duplicate = 0

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            skipped_invalid += 1

            logger.debug(
                "Skipping invalid article item: index=%d type=%s",
                index,
                type(item).__name__,
            )

            continue

        article_id = str(
            item.get("article_id")
            or item.get("id")
            or ""
        ).strip()

        if not article_id:
            skipped_invalid += 1

            logger.debug(
                "Skipping article without article_id: index=%d",
                index,
            )

            continue

        if article_id in seen_ids:
            skipped_duplicate += 1

            logger.debug(
                "Skipping duplicate article: article_id=%s",
                article_id,
            )

            continue

        seen_ids.add(article_id)

        thumbnail_url = item.get("thumbnail_url")

        if not thumbnail_url and isinstance(
            item.get("thumb_list"),
            dict,
        ):
            thumb_list = item["thumb_list"]

            thumbnail_url = (
                thumb_list.get(
                    "thumb_280_168_100_1_crop"
                )
                or next(
                    iter(thumb_list.values()),
                    None,
                )
            )

            logger.debug(
                "Thumbnail fallback applied: article_id=%s "
                "thumbnail_found=%s",
                article_id,
                bool(thumbnail_url),
            )

        article = Article(
            article_id=article_id,
            title=(item.get("title") or "").strip(),
            lead=(
                item.get("lead")
                or item.get("description")
                or ""
            ).strip(),
            share_url=(
                item.get("share_url")
                or item.get("url")
                or ""
            ),
            thumbnail_url=thumbnail_url,
            article_type=item.get("article_type"),
            publish_time=(
                item.get("publish_time")
                or item.get("published_at")
            ),
            update_time=item.get("update_time"),
        )

        normalized.append(article.to_dict())

        logger.debug(
            "Article normalized: article_id=%s title_length=%d "
            "content_hash=%s",
            article_id,
            len(article.title),
            article.content_hash[:12],
        )

    logger.info(
        "VNExpress article normalization completed: "
        "input=%d normalized=%d skipped_invalid=%d "
        "skipped_duplicate=%d",
        len(items),
        len(normalized),
        skipped_invalid,
        skipped_duplicate,
    )

    return {
        "ok": True,
        "count": len(normalized),
        "articles": normalized,
    }
