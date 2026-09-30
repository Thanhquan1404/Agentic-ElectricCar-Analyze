from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from groq import Groq, GroqError

from utils.logger import setup_logger


logger = setup_logger(__name__)

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_CLASSIFY_MODEL = os.getenv("LLM_CLASSIFY_MODEL", "")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))

PROMPT_VERSION = "cp3-classify-v1"


# ---------------------------------------------------------------------------
# Article taxonomy
# ---------------------------------------------------------------------------

ARTICLE_TYPES = [
    "sales_ranking",
    "product_launch",
    "pricing_promotion",
    "policy_regulation",
    "recall_service",
    "event_launch",
    "finance_market",
    "partnership",
    "competition",
    "technology",
    "consumer_opinion",
    "macro_market",
    "other",
]


# Keyword anchors used as a lightweight prior to stabilize LLM classification
# and provide a deterministic fallback when the LLM request fails.
TYPE_KEYWORDS: dict[str, list[str]] = {
    "sales_ranking": [
        "doanh số",
        "bán chạy",
        "xếp hạng",
        "thị phần",
        "top",
    ],
    "product_launch": [
        "ra mắt",
        "giới thiệu",
        "phiên bản",
        "mẫu mới",
        "VF ",
    ],
    "pricing_promotion": [
        "giá",
        "ưu đãi",
        "giảm",
        "khuyến mại",
        "trợ giá",
        "lăn bánh",
    ],
    "policy_regulation": [
        "nghị định",
        "thông tư",
        "chính sách",
        "lệ phí",
        "thuế",
        "quy định",
        "Bộ ",
        "Chính phủ",
    ],
    "recall_service": [
        "triệu hồi",
        "lỗi",
        "bảo hành",
        "sửa chữa",
        "khiếu nại",
        "tai nạn",
        "cháy",
    ],
    "event_launch": [
        "khai trương",
        "showroom",
        "triển lãm",
        "sự kiện",
        "đại lý",
    ],
    "finance_market": [
        "cổ phiếu",
        "lỗ",
        "lợi nhuận",
        "IPO",
        "vốn",
        "tài chính",
    ],
    "partnership": [
        "hợp tác",
        "ký kết",
        "đối tác",
        "liên doanh",
        "M&A",
    ],
    "competition": [
        "đối thủ",
        "cạnh tranh",
        "Toyota",
        "Hyundai",
        "BYD",
    ],
    "technology": [
        "pin",
        "sạc",
        "ADAS",
        "công nghệ",
        "R&D",
        "phần mềm",
    ],
    "consumer_opinion": [
        "người dùng",
        "khách hàng nói",
        "đánh giá",
        "review",
        "khảo sát",
    ],
    "macro_market": [
        "thị trường",
        "xu hướng",
        "ASEAN",
        "Đông Nam Á",
        "toàn cầu",
    ],
}


# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""
You are a classifier for VNExpress articles related to VinFast.

Read the TITLE, LEAD, and the first 8 paragraphs.
Select EXACTLY ONE primary article type and optionally ONE secondary type.

TAXONOMY:

{chr(10).join(f"- {article_type}" for article_type in ARTICLE_TYPES)}

RULES:

- If the article mentions MULTIPLE competitors together with sales or
  ranking figures, classify it as sales_ranking.
- If the article focuses on a VEHICLE PRODUCT, model, or version,
  classify it as product_launch.
- If the article focuses on POLICIES or REGULATIONS, classify it as
  policy_regulation rather than macro_market.
- If the article focuses on RECALLS or TECHNICAL ISSUES, classify it as
  recall_service rather than consumer_opinion.
- Select other only when the article does not meaningfully match any
  available category.
- If confidence < 0.6, secondary_type MUST be provided.
- If the article remains ambiguous after selecting a secondary type,
  return other.

Return ONLY valid JSON.
Do not include prose outside the JSON.

{{
  "article_type": "<one taxonomy value>",
  "confidence": 0.0,
  "secondary_type": "<one taxonomy value>|null",
  "reasoning_1line": "<20 words or fewer>"
}}
"""


# ---------------------------------------------------------------------------
# Classification data model
# ---------------------------------------------------------------------------

@dataclass
class Classification:
    """Structured result produced by the article classifier."""

    article_type: str
    confidence: float
    secondary_type: Optional[str]
    reasoning_1line: str
    model: str = LLM_CLASSIFY_MODEL
    prompt_version: str = PROMPT_VERSION
    classified_at: str = ""

    def __post_init__(self) -> None:
        """Populate the classification timestamp when it is not provided."""

        if not self.classified_at:
            self.classified_at = datetime.now(
                timezone.utc
            ).isoformat()

    def to_dict(self) -> dict:
        """Convert the classification object into a serializable dictionary."""

        return asdict(self)


# ---------------------------------------------------------------------------
# LLM request
# ---------------------------------------------------------------------------

def _call_llm(
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> dict:
    """
    Execute the classification request through the Groq API.

    Returns:
        A dictionary containing request status, generated content,
        token usage, and error information.
    """

    if not LLM_CLASSIFY_MODEL or not GROQ_API_KEY:
        logger.error(
            "Classifier configuration is incomplete | "
            "model=%s | api_key_configured=%s",
            LLM_CLASSIFY_MODEL or "missing",
            bool(GROQ_API_KEY),
        )

        return {
            "ok": False,
            "error": "missing_llm_env",
            "content": None,
            "usage": None,
        }

    started_at = time.perf_counter()

    logger.debug(
        "Classifier LLM request started | model=%s | "
        "temperature=%.2f | max_tokens=%d | timeout=%ss",
        LLM_CLASSIFY_MODEL,
        temperature,
        max_tokens,
        LLM_TIMEOUT,
    )

    try:
        client = Groq(
            api_key=GROQ_API_KEY,
        )

        completion = client.chat.completions.create(
            model=LLM_CLASSIFY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_object",
            },
            timeout=LLM_TIMEOUT,
        )

        content = completion.choices[0].message.content

        usage = (
            completion.usage.model_dump()
            if completion.usage
            else None
        )

        latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        logger.info(
            "Classifier LLM request completed | "
            "model=%s | latency=%dms | "
            "prompt_tokens=%s | completion_tokens=%s | total_tokens=%s",
            LLM_CLASSIFY_MODEL,
            latency_ms,
            usage.get("prompt_tokens") if usage else "N/A",
            usage.get("completion_tokens") if usage else "N/A",
            usage.get("total_tokens") if usage else "N/A",
        )

        return {
            "ok": True,
            "content": content,
            "usage": usage,
            "error": None,
        }

    except GroqError as exc:
        latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        logger.error(
            "Classifier Groq API request failed | "
            "latency=%dms | error=%s",
            latency_ms,
            exc,
            exc_info=True,
        )

        return {
            "ok": False,
            "error": f"llm_api_error: {exc}",
            "content": None,
            "usage": None,
        }

    except Exception as exc:
        latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        logger.exception(
            "Unexpected classifier LLM error | "
            "latency=%dms | error=%s",
            latency_ms,
            exc,
        )

        return {
            "ok": False,
            "error": f"llm_unexpected_error: {exc}",
            "content": None,
            "usage": None,
        }


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

def _keyword_scores(article: dict) -> dict[str, int]:
    """
    Calculate lightweight keyword scores for each article category.

    The score is used as a classification prior and as a deterministic
    fallback when the LLM classifier is unavailable.
    """

    text = " ".join(
        [
            article.get("title", ""),
            article.get("lead", ""),
            " ".join(
                article.get("paragraphs", [])[:8]
            ),
        ]
    ).lower()

    scores = {
        article_type: 0
        for article_type in ARTICLE_TYPES
        if article_type != "other"
    }

    for article_type, keywords in TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text:
                scores[article_type] += 1

    return scores


def _keyword_fallback(article: dict) -> Classification:
    """
    Produce a deterministic classification from keyword scores.

    The fallback intentionally uses a low confidence value because
    keyword matching does not provide the same semantic understanding
    as the LLM classifier.
    """

    scores = _keyword_scores(article)

    best_type = (
        max(scores, key=scores.get)
        if scores
        else "other"
    )

    best_score = scores.get(
        best_type,
        0,
    )

    if best_score == 0:
        best_type = "other"

    logger.warning(
        "Keyword fallback classification selected | "
        "article_id=%s | type=%s | score=%d | confidence=0.45",
        article.get("article_id", "unknown"),
        best_type,
        best_score,
    )

    return Classification(
        article_type=best_type,
        confidence=0.45,
        secondary_type=None,
        reasoning_1line=(
            f"keyword_fallback (score={best_score})"
        ),
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_user_prompt(article: dict) -> str:
    """Build the classification prompt from the article metadata and content."""

    lead = article.get("lead", "")
    title = article.get("title", "")

    paragraphs = "\n".join(
        f"[{index:02d}] {paragraph}"
        for index, paragraph in enumerate(
            article.get("paragraphs", [])[:8],
            1,
        )
    )

    keyword_scores = _keyword_scores(article)

    keyword_hint = ", ".join(
        f"{category}={score}"
        for category, score in sorted(
            keyword_scores.items(),
            key=lambda item: -item[1],
        )[:3]
        if score > 0
    )

    logger.debug(
        "Classification prompt built | "
        "article_id=%s | title_chars=%d | lead_chars=%d | "
        "paragraphs=%d | keyword_hints=%s",
        article.get("article_id", "unknown"),
        len(title),
        len(lead),
        len(article.get("paragraphs", [])[:8]),
        keyword_hint or "none",
    )

    return f"""
TITLE: {title}

LEAD: {lead}

FIRST 8 PARAGRAPHS:

{paragraphs}

KEYWORD HINTS
Reference only. They are not mandatory classification signals.

{keyword_hint or "No keyword matches detected."}

Classify this article.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_article_type(article: dict) -> dict:
    """
    Classify one article into a primary and optional secondary category.

    Returns:
        {
            "ok": bool,
            "classification": Classification dictionary or None,
            "error": str or None,
            "usage": dict or None,
            "latency_ms": int,
            "fallback_used": bool
        }
    """

    started_at = time.perf_counter()

    article_id = article.get(
        "article_id",
        "unknown",
    )

    logger.info(
        "Article classification started | "
        "article_id=%s | model=%s",
        article_id,
        LLM_CLASSIFY_MODEL or "not_configured",
    )

    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------

    if not article or not article.get("title"):
        latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        logger.warning(
            "Article classification skipped | "
            "reason=empty_article_or_missing_title | "
            "article_id=%s | latency=%dms",
            article_id,
            latency_ms,
        )

        return {
            "ok": False,
            "classification": None,
            "error": "empty_article",
            "usage": None,
            "fallback_used": False,
            "latency_ms": latency_ms,
        }

    logger.debug(
        "Classification input validated | "
        "article_id=%s | title=%s | paragraphs=%d",
        article_id,
        article.get("title", "")[:120],
        len(article.get("paragraphs", [])),
    )

    # -----------------------------------------------------------------------
    # LLM classification
    # -----------------------------------------------------------------------

    logger.info(
        "Classification stage started | "
        "article_id=%s | strategy=llm_with_keyword_fallback",
        article_id,
    )

    llm = _call_llm(
        SYSTEM_PROMPT,
        _build_user_prompt(article),
    )

    # -----------------------------------------------------------------------
    # LLM failure fallback
    # -----------------------------------------------------------------------

    if not llm["ok"]:
        logger.warning(
            "LLM classification unavailable | "
            "article_id=%s | error=%s | "
            "switching_to=keyword_fallback",
            article_id,
            llm["error"],
        )

        classification = _keyword_fallback(article)

        latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        logger.info(
            "Classification completed using fallback | "
            "article_id=%s | type=%s | confidence=%.2f | "
            "latency=%dms",
            article_id,
            classification.article_type,
            classification.confidence,
            latency_ms,
        )

        return {
            "ok": True,
            "classification": classification.to_dict(),
            "error": llm["error"],
            "usage": None,
            "fallback_used": True,
            "latency_ms": latency_ms,
        }

    # -----------------------------------------------------------------------
    # JSON parsing
    # -----------------------------------------------------------------------

    logger.debug(
        "Parsing classifier JSON response | "
        "article_id=%s",
        article_id,
    )

    raw = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        (llm.get("content") or "").strip(),
        flags=re.MULTILINE,
    ).strip()

    try:
        parsed = json.loads(raw)

    except json.JSONDecodeError as exc:
        logger.error(
            "Classifier JSON parsing failed | "
            "article_id=%s | error=%s | raw_length=%d",
            article_id,
            exc,
            len(raw),
            exc_info=True,
        )

        classification = _keyword_fallback(article)

        latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        logger.warning(
            "Classification fallback activated after JSON parsing failure | "
            "article_id=%s | type=%s | latency=%dms",
            article_id,
            classification.article_type,
            latency_ms,
        )

        return {
            "ok": True,
            "classification": classification.to_dict(),
            "error": f"invalid_json: {exc}",
            "usage": llm.get("usage"),
            "fallback_used": True,
            "latency_ms": latency_ms,
        }

    logger.debug(
        "Classifier JSON parsed successfully | "
        "article_id=%s | keys=%s",
        article_id,
        ", ".join(parsed.keys()),
    )

    # -----------------------------------------------------------------------
    # Result normalization
    # -----------------------------------------------------------------------

    article_type = parsed.get(
        "article_type",
        "other",
    )

    if article_type not in ARTICLE_TYPES:
        logger.warning(
            "Invalid primary article type returned | "
            "article_id=%s | value=%s | fallback=other",
            article_id,
            article_type,
        )

        article_type = "other"

    secondary_type = parsed.get(
        "secondary_type"
    )

    if secondary_type not in ARTICLE_TYPES:
        if secondary_type is not None:
            logger.warning(
                "Invalid secondary article type returned | "
                "article_id=%s | value=%s | fallback=null",
                article_id,
                secondary_type,
            )

        secondary_type = None

    try:
        confidence = float(
            parsed.get(
                "confidence",
                0.0,
            )
        )

    except (TypeError, ValueError):
        logger.warning(
            "Invalid classifier confidence | "
            "article_id=%s | value=%s | fallback=0.0",
            article_id,
            parsed.get("confidence"),
        )

        confidence = 0.0

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    classification = Classification(
        article_type=article_type,
        confidence=confidence,
        secondary_type=secondary_type,
        reasoning_1line=(
            parsed.get("reasoning_1line") or ""
        )[:200],
    )

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    latency_ms = int(
        (time.perf_counter() - started_at) * 1000
    )

    usage = llm.get("usage") or {}

    logger.info(
        "Article classification completed | "
        "article_id=%s | type=%s | confidence=%.3f | "
        "secondary=%s | latency=%dms | "
        "prompt_tokens=%s | completion_tokens=%s | "
        "total_tokens=%s | fallback=False",
        article_id,
        classification.article_type,
        classification.confidence,
        classification.secondary_type or "none",
        latency_ms,
        usage.get("prompt_tokens", "N/A"),
        usage.get("completion_tokens", "N/A"),
        usage.get("total_tokens", "N/A"),
    )

    return {
        "ok": True,
        "classification": classification.to_dict(),
        "error": None,
        "usage": llm.get("usage"),
        "fallback_used": False,
        "latency_ms": latency_ms,
    }
