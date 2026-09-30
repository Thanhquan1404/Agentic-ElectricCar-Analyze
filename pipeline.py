# pipeline.py
"""
Pipeline orchestration separated from the UI.
All functions here can be called from Streamlit, CLI, or API server.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from tools.vnexpress_tool import vnexpress_fetch_vinfast_articles
from tools.vnexpress_detail import vnexpress_fetch_article_detail
from tools.analyze import information_analyze
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------------------------------ #
# Dataclasses
# ------------------------------------------------------------------ #
@dataclass
class ListingResult:
    ok: bool
    articles: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: int = 0


@dataclass
class AnalysisResult:
    ok: bool
    article_id: str = ""
    detail: Optional[dict] = None          # ArticleDetail dict
    analysis: Optional[dict] = None        # Analysis dict
    classification: Optional[dict] = None
    lenses: list[str] = field(default_factory=list)
    error: Optional[str] = None
    detail_ms: int = 0
    analyze_ms: int = 0
    classify_ms: int = 0
    total_ms: int = 0


# ------------------------------------------------------------------ #
# Step 1 — Listing
# ------------------------------------------------------------------ #
def fetch_listing(limit: int = 20,
                  progress_cb: Optional[Callable[[str], None]] = None) -> ListingResult:
    """Fetch the latest list of articles about VinFast."""
    if progress_cb:
        progress_cb(f"📡 Fetching the latest {limit} articles from VnExpress...")
    t0 = time.perf_counter()
    res = vnexpress_fetch_vinfast_articles(limit=limit)
    ms = int((time.perf_counter() - t0) * 1000)
    if not res.get("ok"):
        return ListingResult(ok=False, error=res.get("error", "unknown"),
                             latency_ms=ms)
    return ListingResult(ok=True, articles=res.get("articles", []),
                         latency_ms=ms)


# ------------------------------------------------------------------ #
# Step 2 — Fetch detail + analyze (Agent flow)
# ------------------------------------------------------------------ #
def analyze_article(article: dict,
                    verbose: bool = False,
                    progress_cb: Optional[Callable[[str], None]] = None) -> AnalysisResult:
    """
    Receive 1 article from listing → fetch detail → analyze.
    This is the main function triggered when the user clicks "Analyze".
    """
    t_start = time.perf_counter()
    article_id = article.get("article_id", "")
    share_url = article.get("share_url", "")

    if not share_url:
        return AnalysisResult(ok=False, article_id=article_id,
                              error="missing_share_url")

    # ---- Step 2a: Fetch detail ----
    if progress_cb:
        progress_cb(f"📄 Fetching full text for article {article_id}...")
    t0 = time.perf_counter()
    detail = vnexpress_fetch_article_detail(
        share_url=share_url,
        article_id=article_id,
        verbose=verbose,
    )
    detail_ms = int((time.perf_counter() - t0) * 1000)

    if not detail.get("ok"):
        return AnalysisResult(ok=False, article_id=article_id,
                              error=f"detail: {detail.get('error')}",
                              detail_ms=detail_ms,
                              total_ms=int((time.perf_counter()-t_start)*1000))

    # ---- Step 2b: Analyze (includes classification internally) ----
    if progress_cb:
        progress_cb("🧠 Classifying and analyzing with LLM...")
    t0 = time.perf_counter()
    result = information_analyze(detail["article"])
    analyze_ms = int((time.perf_counter() - t0) * 1000)

    if not result.get("analysis"):
        return AnalysisResult(
            ok=False, article_id=article_id,
            detail=detail["article"],
            error=f"analyze: {result.get('error')}",
            detail_ms=detail_ms, analyze_ms=analyze_ms,
            total_ms=int((time.perf_counter()-t_start)*1000),
        )

    return AnalysisResult(
        ok=True,
        article_id=article_id,
        detail=detail["article"],
        analysis=result["analysis"],
        classification=result.get("classification"),
        lenses=result.get("lenses", []),
        detail_ms=detail_ms,
        classify_ms=result.get("classify_ms", 0),
        analyze_ms=analyze_ms,
        total_ms=int((time.perf_counter() - t_start) * 1000),
    )