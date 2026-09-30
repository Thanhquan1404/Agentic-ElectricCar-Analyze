from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from groq import Groq, GroqError

from tools.classify import ARTICLE_TYPES, classify_article_type
from tools.lenses import build_lens_block, build_schema_hint, get_lenses
from utils.logger import setup_logger


load_dotenv()

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_ANALYZE_MODEL = os.getenv("LLM_ANALYZE_MODEL", "gpt-4o-mini")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
PROMPT_VERSION = "cp3-analyze-v2"


# ---------------------------------------------------------------------------
# System context
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """
[USER CONTEXT]

You support an OFFICIAL VINFAST DISTRIBUTOR in Vietnam.

Primary objective: INCREASE VINFAST VEHICLE SALES AND MAINTAIN SMOOTH OPERATIONS.

[ARTICLE TYPE]

- article_type:
{article_type}

- confidence:
{confidence}

- secondary_type:
{secondary_type}

- applied lenses:
{lenses}

[MANDATORY FUZZY PRINCIPLES]

- DO NOT use binary classification. Always use a degree from 0.0 to 1.0
  together with a band.
- band must be one of:
  ["very_low", "low", "medium", "high", "very_high"].
- Ambiguous statements must be added to fuzzy_ambiguities with at least
  two possible interpretations and their respective strengths.
- If a field is not applicable, return null. DO NOT fabricate values
  to fill the schema.

[KEY SIGNALS]

- key_signals must contain ALL important signals:
  positive, negative, and neutral.
- Every signal MUST contain:
  polarity ∈ ["positive", "negative", "neutral"].
- For recall_service and finance_market articles, negative signals are
  especially important. DO NOT force them into marketing opportunities.
- action_hint must be SPECIFIC:
  responsible channel, responsible person/team, and deadline when available.

[NO FABRICATION]

- Do not add competitors or numerical data that do not appear in the article.
- Do not make political inferences.
- source_quote MUST contain an exact quotation from the source article.

[LENS-SPECIFIC INSTRUCTIONS]

{lens_block}

[OUTPUT]

Return ONLY valid JSON matching the schema provided in the user prompt.
Do not include prose outside the JSON.
"""


# ---------------------------------------------------------------------------
# JSON schema hint
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA_HINT = """
{
  "article_id": "string",
  "article_type": "string",
  "summary_1line": "string (≤25 words, distributor perspective)",

  "vinfast_stance": {
    "sentiment": 0.0,
    "confidence": 0.0,
    "band": "very_low|low|medium|high|very_high",
    "evidence": ["exact quotation"]
  },

  "relevance_to_distributor": {
    "score": 0.0,
    "band": "very_low|low|medium|high|very_high",
    "reason": "string"
  },

  "entities": {
    "brands": [],
    "models": [],
    "markets": [],
    "people": [],
    "numbers": [
      {
        "value": "string",
        "unit": "string",
        "context": "string"
      }
    ]
  },

  "key_signals": [
    {
      "signal": "string",
      "polarity": "positive|negative|neutral",
      "strength": 0.0,
      "band": "very_low|low|medium|high|very_high",
      "angle": "string",
      "action_hint": "string",
      "source_quote": "string"
    }
  ],

  "fuzzy_ambiguities": [
    {
      "sentence": "string",
      "interpretations": [
        {
          "reading": "string",
          "strength": 0.0
        },
        {
          "reading": "string",
          "strength": 0.0
        }
      ]
    }
  ],

  "meta": {
    "model": "string",
    "analyzed_at": "ISO8601",
    "prompt_version": "cp3-v2"
  }
}
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_enum_value(value: Any) -> str:
    """Extract a string value from an Enum instance or a plain string."""
    if hasattr(value, "value"):
        return str(value.value)

    return str(value) if value is not None else ""


def _safe_article_id(article: dict) -> str:
    """Return a safe article identifier for logging."""
    return str(article.get("article_id") or "unknown")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 2500,
) -> dict:
    """
    Execute the Groq LLM request.

    Returns:
        A dictionary containing request status, generated content,
        token usage, and error information.
    """

    if not LLM_ANALYZE_MODEL or not GROQ_API_KEY:
        logger.error(
            "LLM configuration is incomplete | model=%s | api_key_configured=%s",
            LLM_ANALYZE_MODEL or "missing",
            bool(GROQ_API_KEY),
        )

        return {
            "ok": False,
            "error": "missing_llm_env",
            "content": None,
            "usage": None,
        }

    request_started = time.perf_counter()

    logger.debug(
        "LLM request started | model=%s | temperature=%.2f | max_tokens=%d | timeout=%ss",
        LLM_ANALYZE_MODEL,
        temperature,
        max_tokens,
        LLM_TIMEOUT,
    )

    try:
        client = Groq(api_key=GROQ_API_KEY)

        completion = client.chat.completions.create(
            model=LLM_ANALYZE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=LLM_TIMEOUT,
        )

        content = completion.choices[0].message.content
        usage = completion.usage.model_dump() if completion.usage else None

        latency_ms = int(
            (time.perf_counter() - request_started) * 1000
        )

        logger.info(
            "LLM request completed | model=%s | latency=%dms | "
            "prompt_tokens=%s | completion_tokens=%s | total_tokens=%s",
            LLM_ANALYZE_MODEL,
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
            (time.perf_counter() - request_started) * 1000
        )

        logger.error(
            "Groq API request failed | latency=%dms | error=%s",
            latency_ms,
            exc,
            exc_info=True,
        )

        return {
            "ok": False,
            "error": f"groq_api_error: {exc}",
            "content": None,
            "usage": None,
        }

    except Exception as exc:
        latency_ms = int(
            (time.perf_counter() - request_started) * 1000
        )

        logger.exception(
            "Unexpected LLM error | latency=%dms | error=%s",
            latency_ms,
            exc,
        )

        return {
            "ok": False,
            "error": f"groq_unexpected_error: {exc}",
            "content": None,
            "usage": None,
        }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_user_prompt(
    article: dict,
    schema_hint: dict,
) -> str:
    """Build the user prompt containing article metadata and content."""

    article_id = _safe_article_id(article)
    tags = ", ".join(article.get("tags") or []) or "N/A"
    paragraphs = article.get("paragraphs") or []

    body = "\n".join(
        f"[{index:02d}] {paragraph}"
        for index, paragraph in enumerate(paragraphs, 1)
    )

    schema_json = json.dumps(
        schema_hint,
        ensure_ascii=False,
        indent=2,
    )

    logger.debug(
        "Building LLM prompt | article_id=%s | paragraphs=%d | tags=%d",
        article_id,
        len(paragraphs),
        len(article.get("tags") or []),
    )

    return f"""
=== METADATA ===

article_id: {article.get("article_id")}
title: {article.get("title")}
lead: {article.get("lead")}
author: {article.get("author") or "N/A"}
publish_time: {article.get("publish_time") or "N/A"}
tags: {tags}
share_url: {article.get("share_url")}

=== CLEANED CONTENT ===

{body}

=== REQUIREMENTS ===

1. Evaluate VinFast stance using fuzzy sentiment, confidence, and evidence.
2. Evaluate distributor relevance using score, band, and reason.
3. Extract entities:
   brands, models, markets, people, and numbers with context.
4. Extract at least two key signals with:
   polarity, strength, band, angle, action_hint, and exact source_quote.
5. Identify sentences with multiple interpretations and list all relevant
   interpretations with their respective strengths.
6. Fill lens-specific extensions according to the schema.
   Return null when a field is not applicable.

Return ONLY JSON according to the following schema:

{schema_json}
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

BASE_REQUIRED = [
    "article_id",
    "article_type",
    "summary_1line",
    "relevance_to_distributor",
    "vinfast_stance",
    "entities",
    "key_signals",
    "fuzzy_ambiguities",
    "meta",
]


def _validate(
    parsed: dict,
    lenses: list[str],
) -> list[str]:
    """Validate required base fields and dynamically selected lens fields."""

    missing = [
        key
        for key in BASE_REQUIRED
        if key not in parsed
    ]

    for lens in lenses:
        if lens not in parsed:
            missing.append(f"lens:{lens}")

    return missing


def _normalize_signals(parsed: dict) -> None:
    """Ensure every signal has a valid polarity value."""

    valid_polarities = {
        "positive",
        "negative",
        "neutral",
    }

    for signal in parsed.get("key_signals", []) or []:
        if not isinstance(signal, dict):
            continue

        polarity = _extract_enum_value(
            signal.get("polarity")
        ).lower()

        if polarity not in valid_polarities:
            logger.warning(
                "Invalid signal polarity detected | value=%s | "
                "defaulting_to=neutral",
                polarity,
            )
            signal["polarity"] = "neutral"
        else:
            signal["polarity"] = polarity


# ---------------------------------------------------------------------------
# Public analysis function
# ---------------------------------------------------------------------------

def information_analyze(
    article: dict,
    *,
    temperature: float = 0.2,
    use_classifier: bool = True,
) -> dict:
    """
    Analyze an article through classification, lens routing, and LLM analysis.

    The function exposes detailed execution metrics through the logger,
    including classification latency, LLM latency, total latency,
    selected article type, selected lenses, token usage, and validation status.
    """

    started_at = time.perf_counter()
    article_id = _safe_article_id(article)

    logger.info(
        "Analysis started | article_id=%s | classifier=%s | temperature=%.2f",
        article_id,
        "enabled" if use_classifier else "disabled",
        temperature,
    )

    # -----------------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------------

    if not article or not article.get("paragraphs"):
        latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        logger.warning(
            "Analysis skipped | reason=empty_article | latency=%dms",
            latency_ms,
        )

        return {
            "ok": False,
            "error": "empty_article",
            "analysis": None,
            "classification": None,
            "lenses": [],
            "missing_keys": [],
            "usage": None,
            "latency_ms": latency_ms,
            "classify_ms": 0,
            "fallback_used": False,
        }

    logger.debug(
        "Input validated | article_id=%s | paragraphs=%d",
        article_id,
        len(article.get("paragraphs") or []),
    )

    # -----------------------------------------------------------------------
    # Stage A: Article classification
    # -----------------------------------------------------------------------

    classification_started = time.perf_counter()

    if use_classifier:
        logger.info(
            "Stage A started | action=classify_article | article_id=%s",
            article_id,
        )

        clf_res = classify_article_type(article)

        clf = clf_res.get("classification") or {
            "article_type": "other",
            "confidence": 0.0,
            "secondary_type": None,
            "reasoning_1line": "no_classification",
        }

    else:
        logger.info(
            "Stage A skipped | reason=classifier_disabled | article_id=%s",
            article_id,
        )

        clf = {
            "article_type": "other",
            "confidence": 0.0,
            "secondary_type": None,
            "reasoning_1line": "disabled",
        }

        clf_res = {
            "fallback_used": True,
        }

    classify_ms = int(
        (time.perf_counter() - classification_started) * 1000
    )

    raw_article_type = clf.get("article_type")
    article_type = _extract_enum_value(raw_article_type)

    valid_article_types = {
        _extract_enum_value(article_type_item)
        for article_type_item in ARTICLE_TYPES
    }

    if article_type not in valid_article_types:
        logger.warning(
            "Unknown article type returned by classifier | "
            "value=%s | fallback=other",
            article_type,
        )
        article_type = "other"

    secondary_type = (
        _extract_enum_value(clf.get("secondary_type"))
        or None
    )

    if secondary_type and secondary_type not in valid_article_types:
        logger.warning(
            "Unknown secondary article type | value=%s | fallback=null",
            secondary_type,
        )
        secondary_type = None

    logger.info(
        "Stage A completed | article_id=%s | type=%s | confidence=%.3f | "
        "secondary=%s | latency=%dms | fallback=%s",
        article_id,
        article_type,
        clf.get("confidence", 0.0),
        secondary_type or "none",
        classify_ms,
        clf_res.get("fallback_used", False),
    )

    # -----------------------------------------------------------------------
    # Stage B: Lens routing
    # -----------------------------------------------------------------------

    logger.info(
        "Stage B started | action=resolve_lenses | article_id=%s | "
        "article_type=%s",
        article_id,
        article_type,
    )

    lenses = get_lenses(
        article_type,
        secondary_type,
    )

    logger.info(
        "Lens routing completed | article_id=%s | lens_count=%d | lenses=%s",
        article_id,
        len(lenses),
        ", ".join(lenses) if lenses else "none",
    )

    # -----------------------------------------------------------------------
    # Stage C: Prompt construction
    # -----------------------------------------------------------------------

    logger.debug(
        "Stage C started | action=build_prompt | article_id=%s",
        article_id,
    )

    schema_hint = build_schema_hint(
        article_type,
        lenses,
    )

    system = _SYSTEM_TEMPLATE.format(
        article_type=article_type,
        confidence=clf.get("confidence", 0.0),
        secondary_type=secondary_type or "null",
        lenses=", ".join(lenses),
        lens_block=build_lens_block(lenses),
    )

    user = _build_user_prompt(
        article,
        schema_hint,
    )

    logger.debug(
        "Prompt construction completed | article_id=%s | "
        "system_chars=%d | user_chars=%d",
        article_id,
        len(system),
        len(user),
    )

    # -----------------------------------------------------------------------
    # Stage D: LLM analysis
    # -----------------------------------------------------------------------

    logger.info(
        "Stage D started | action=llm_analysis | article_id=%s | model=%s",
        article_id,
        LLM_ANALYZE_MODEL,
    )

    llm = _call_llm(
        system,
        user,
        temperature=temperature,
    )

    if not llm["ok"]:
        total_latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        logger.error(
            "Stage D failed | article_id=%s | error=%s | "
            "classify_ms=%d | total_latency=%dms",
            article_id,
            llm["error"],
            classify_ms,
            total_latency_ms,
        )

        return {
            "ok": False,
            "error": llm["error"],
            "analysis": None,
            "classification": clf,
            "lenses": lenses,
            "missing_keys": [],
            "usage": None,
            "classify_ms": classify_ms,
            "latency_ms": total_latency_ms,
            "fallback_used": clf_res.get(
                "fallback_used",
                False,
            ),
        }

    logger.info(
        "Stage D completed | article_id=%s | model=%s",
        article_id,
        LLM_ANALYZE_MODEL,
    )

    # -----------------------------------------------------------------------
    # Stage E: JSON parsing
    # -----------------------------------------------------------------------

    logger.debug(
        "Stage E started | action=parse_json | article_id=%s",
        article_id,
    )

    raw_content = (
        llm.get("content") or ""
    ).strip()

    raw = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        raw_content,
        flags=re.MULTILINE,
    ).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        total_latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )

        logger.error(
            "JSON parsing failed | article_id=%s | error=%s | "
            "raw_length=%d | total_latency=%dms",
            article_id,
            exc,
            len(raw),
            total_latency_ms,
            exc_info=True,
        )

        return {
            "ok": False,
            "error": f"invalid_json: {exc}",
            "analysis": None,
            "classification": clf,
            "lenses": lenses,
            "missing_keys": [],
            "usage": llm.get("usage"),
            "classify_ms": classify_ms,
            "latency_ms": total_latency_ms,
            "fallback_used": clf_res.get(
                "fallback_used",
                False,
            ),
        }

    # FIX: Kiểm tra nếu parsed không phải kiểu dictionary (ví dụ: LLM trả về JSON null/string/list)
    if not isinstance(parsed, dict):
        total_latency_ms = int(
            (time.perf_counter() - started_at) * 1000
        )
        logger.error(
            "Parsed JSON is not a dictionary | article_id=%s | parsed_type=%s | total_latency=%dms",
            article_id,
            type(parsed).__name__,
            total_latency_ms,
        )
        return {
            "ok": False,
            "error": f"invalid_json_structure: expected dict, got {type(parsed).__name__}",
            "analysis": None,
            "classification": clf,
            "lenses": lenses,
            "missing_keys": [],
            "usage": llm.get("usage"),
            "classify_ms": classify_ms,
            "latency_ms": total_latency_ms,
            "fallback_used": clf_res.get(
                "fallback_used",
                False,
            ),
        }

    logger.debug(
        "JSON parsing completed | article_id=%s | top_level_keys=%d",
        article_id,
        len(parsed),
    )

    # -----------------------------------------------------------------------
    # Stage F: Post-processing
    # -----------------------------------------------------------------------

    parsed.setdefault("meta", {})

    parsed["meta"].update(
        {
            "model": LLM_ANALYZE_MODEL,
            "analyzed_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "prompt_version": PROMPT_VERSION,
            "classifier_version": "cp3-classify-v1",
        }
    )

    parsed["article_id"] = (
        parsed.get("article_id")
        or article.get("article_id", "")
    )

    parsed["article_type"] = article_type
    parsed["article_type_confidence"] = clf.get(
        "confidence",
        0.0,
    )
    parsed["secondary_type"] = secondary_type

    _normalize_signals(parsed)

    missing = _validate(
        parsed,
        lenses,
    )

    total_latency_ms = int(
        (time.perf_counter() - started_at) * 1000
    )

    analysis_ok = len(missing) == 0

    # -----------------------------------------------------------------------
    # Final execution summary
    # -----------------------------------------------------------------------

    usage = llm.get("usage") or {}

    logger.info(
        "Analysis completed | article_id=%s | status=%s | "
        "type=%s | lenses=%d | signals=%d | "
        "missing_keys=%d | classify_ms=%d | total_latency=%dms | "
        "total_tokens=%s",
        article_id,
        "SUCCESS" if analysis_ok else "VALIDATION_FAILED",
        article_type,
        len(lenses),
        len(parsed.get("key_signals") or []),
        len(missing),
        classify_ms,
        total_latency_ms,
        usage.get("total_tokens", "N/A"),
    )

    if missing:
        logger.warning(
            "Analysis validation failed | article_id=%s | missing=%s",
            article_id,
            ", ".join(missing),
        )

    return {
        "ok": analysis_ok,
        "analysis": parsed,
        "classification": clf,
        "lenses": lenses,
        "error": None if not missing else f"missing_keys: {missing}",
        "missing_keys": missing,
        "usage": llm.get("usage"),
        "classify_ms": classify_ms,
        "latency_ms": total_latency_ms,
        "fallback_used": clf_res.get(
            "fallback_used",
            False,
        ),
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_analysis(analysis: dict) -> None:
    """Print a human-readable representation of the analysis result."""

    if not analysis:
        print("Empty analysis")
        return

    lens_keys = [
        key
        for key in analysis.keys()
        if key in {
            "competitive_position",
            "growth_signal",
            "product_positioning",
            "sales_enablement",
            "margin_impact",
            "demand_lever",
            "compliance_risk",
            "reputation_risk",
            "service_ops",
            "channel_expansion",
            "brand_presence",
            "investor_signal",
            "cost_pressure",
            "ecosystem_play",
            "capability_gap",
            "threat_assessment",
            "counter_positioning",
            "product_differentiation",
            "training_need",
            "perception_signal",
            "objection_handling",
            "demand_context",
            "timing_signal",
            "general_signal",
        }
    ]

    print("=" * 90)

    print(
        f"ARTICLE TYPE: {analysis.get('article_type')} "
        f"(confidence={analysis.get('article_type_confidence', 0):.2f})"
    )

    print(
        f"LENSES: {', '.join(lens_keys) if lens_keys else 'none'}"
    )

    print(
        f"SUMMARY: {analysis.get('summary_1line', '')}"
    )

    stance = analysis.get("vinfast_stance", {}) or {}

    print(
        f"STANCE: {stance.get('sentiment')} "
        f"({stance.get('band')}) | "
        f"confidence={stance.get('confidence')}"
    )

    relevance = (
        analysis.get("relevance_to_distributor", {})
        or {}
    )

    print(
        f"DISTRIBUTOR RELEVANCE: {relevance.get('score')} "
        f"({relevance.get('band')})"
    )

    print("-" * 90)

    # Key signals grouped by polarity.
    signals = analysis.get("key_signals") or []

    for polarity, icon in [
        ("positive", "[+]"),
        ("negative", "[!]"),
        ("neutral", "[-]"),
    ]:
        group = [
            signal
            for signal in signals
            if isinstance(signal, dict)
            and signal.get("polarity") == polarity
        ]

        if not group:
            continue

        print(
            f"{icon} {polarity.upper()} SIGNALS ({len(group)})"
        )

        for signal in group:
            print(
                f"    [{signal.get('strength', 0):.2f} "
                f"{signal.get('band', '')}] "
                f"{signal.get('signal', '')}"
            )

            if signal.get("angle"):
                print(
                    f"      Angle: {signal['angle']}"
                )

            if signal.get("action_hint"):
                print(
                    f"      Action: {signal['action_hint']}"
                )

            if signal.get("source_quote"):
                print(
                    f"      Quote: \"{signal['source_quote']}\""
                )

    print("-" * 90)

    # Entities.
    entities = analysis.get("entities") or {}

    if entities.get("brands"):
        print(
            f"BRANDS: {', '.join(entities['brands'])}"
        )

    if entities.get("models"):
        print(
            f"MODELS: {', '.join(entities['models'])}"
        )

    if entities.get("markets"):
        print(
            f"MARKETS: {', '.join(entities['markets'])}"
        )

    if entities.get("people"):
        print(
            f"PEOPLE: {', '.join(entities['people'])}"
        )

    if entities.get("numbers"):
        print("NUMBERS:")

        for number in entities["numbers"][:6]:
            print(
                f"    {number.get('value', '')} "
                f"{number.get('unit', '')} — "
                f"{number.get('context', '')}"
            )

    # Lens-specific extensions.
    for lens in lens_keys:
        value = analysis.get(lens)

        if not value or (
            isinstance(value, dict)
            and not any(value.values())
        ):
            continue

        print("-" * 90)
        print(f"LENS: {lens}")
        _print_nested(value, indent=4)

    # Fuzzy ambiguities.
    ambiguities = analysis.get(
        "fuzzy_ambiguities"
    )

    if ambiguities:
        print("-" * 90)
        print("FUZZY AMBIGUITIES")

        for ambiguity in ambiguities:
            print(
                f"    \"{ambiguity.get('sentence', '')}\""
            )

            for interpretation in ambiguity.get(
                "interpretations",
                [],
            ):
                print(
                    f"      [{interpretation.get('strength', 0):.2f}] "
                    f"{interpretation.get('reading', '')}"
                )

    print("=" * 90)


def _print_nested(
    value: Any,
    indent: int = 4,
) -> None:
    """Recursively print nested dictionaries and lists."""

    padding = " " * indent

    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                print(f"{padding}{key}:")
                _print_nested(
                    item,
                    indent + 2,
                )
            else:
                print(
                    f"{padding}{key}: {item}"
                )

    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                _print_nested(
                    item,
                    indent + 2,
                )
            else:
                print(
                    f"{padding}{item}"
                )

    else:
        print(
            f"{padding}{value}"
        )
