# main.py
"""
CLI mode. Used when not running Streamlit.
Original bug: listing["articles"][2] → should be listing["articles"][:2] or [2].

python main.py --limit 10 --index 2
"""
import sys
import json
import logging
from pipeline import fetch_listing, analyze_article
from tools.analyze import print_analysis

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_cli(limit: int = 5, article_index: int = 0):
    """Run pipeline for 1 article at article_index."""
    listing = fetch_listing(limit=limit)
    if not listing.ok:
        print(f"Listing fail: {listing.error}")
        sys.exit(1)

    print(f"Fetched {len(listing.articles)} articles in {listing.latency_ms}ms\n")

    if not (0 <= article_index < len(listing.articles)):
        print(f"Index {article_index} out of range [0, {len(listing.articles)-1}]")
        sys.exit(1)

    art = listing.articles[article_index]
    print(f"Selected article [{article_index}]: {art['title']}\n")

    def cb(msg: str):
        print(msg)

    result = analyze_article(art, verbose=False, progress_cb=cb)

    if not result.ok:
        print(f"\nAnalyze fail: {result.error}")
        if result.detail:
            print("\nCleaned content (first 10 paragraphs):")
            for i, p in enumerate(result.detail.get("paragraphs", [])[:10], 1):
                print(f"  [{i:02d}] {p}")
        sys.exit(1)

    print()
    print_analysis(result.analysis)
    print(f"\ndetail={result.detail_ms}ms | "
          f"classify={result.classify_ms}ms | "
          f"analyze={result.analyze_ms}ms | "
          f"total={result.total_ms}ms")
    print(f"Lens: {', '.join(result.lenses)}")

    # Save to file
    out = f"analysis_{result.article_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result.analysis, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=5,
                   help="Number of articles to fetch (default 5)")
    p.add_argument("--index", type=int, default=0,
                   help="Index of article to analyze (0-based)")
    args = p.parse_args()
    run_cli(limit=args.limit, article_index=args.index)