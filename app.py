# app.py
"""
UI that allows the user to:
  1. View the latest VinFast news list
  2. Select an article → agent fetches detail + analyzes
  3. View results as visual cards (signals, entities, lenses)
  4. Download analysis JSON
"""
from __future__ import annotations
import json
import os
import streamlit as st
from datetime import datetime, timezone


from pipeline import fetch_listing, analyze_article


# ---------- Page config ----------
st.set_page_config(
    page_title="VinFast News Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Light CSS ----------
st.markdown("""
<style>
.signal-positive { border-left: 4px solid #16a34a; padding-left: 12px; margin: 8px 0; }
.signal-negative { border-left: 4px solid #dc2626; padding-left: 12px; margin: 8px 0; }
.signal-neutral  { border-left: 4px solid #6b7280; padding-left: 12px; margin: 8px 0; }
.type-badge {
  display: inline-block; padding: 2px 10px; border-radius: 12px;
  background: #1e40af; color: white; font-size: 12px; font-weight: 600;
}
.metric-box {
  background: #f3f4f6; border-radius: 8px; padding: 12px; margin: 4px 0;
}
.article-card {
  border: 1px solid #e5e7eb; border-radius: 8px;
  padding: 12px; margin-bottom: 12px; background: white;
}
</style>
""", unsafe_allow_html=True)



# ---------- Session state ----------
def _init_state():
    st.session_state.setdefault("listing", None)
    st.session_state.setdefault("selected_article", None)
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("last_fetch_at", None)


_init_state()

# ================== Helpers ==================
def _render_nested(val, indent: int = 0):
    """Render nested dict/list as markdown."""
    pad = "&nbsp;" * (indent * 4)
    if isinstance(val, dict):
        for k, v in val.items():
            if isinstance(v, (dict, list)):
                st.markdown(f"{pad}**{k}**")
                _render_nested(v, indent + 1)
            else:
                st.markdown(f"{pad}- **{k}**: {v}")
    elif isinstance(val, list):
        for item in val:
            if isinstance(item, (dict, list)):
                _render_nested(item, indent + 1)
                st.markdown(f"{pad}---")
            else:
                st.markdown(f"{pad}- {item}")
    else:
        st.markdown(f"{pad}{val}")



def _chip_list(label: str, items: list):
    if not items:
        return
    st.markdown(f"**{label}**")
    st.markdown(" ".join(f"`{x}`" for x in items))



# ---------- Sidebar ----------
with st.sidebar:
    st.title("VinFast News Agent")
    st.caption("Agent that fetches & analyzes VinFast news from VNExpress")


    st.divider()
    st.subheader("Settings")


    limit = st.slider("Number of articles to fetch", min_value=5, max_value=50,
                      value=5, step=5)


    groq_api_key = os.getenv("GROQ_API_KEY", "")


    model = os.getenv(
        "LLM_ANALYZE_MODEL",
        "llama-3.3-70b-versatile"
    )


    if groq_api_key:
        st.success(f"LLM: Groq / {model}")
        st.caption("API: OpenAI-compatible")
    else:
        st.error("Missing GROQ_API_KEY in .env")
        st.stop()


    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reload", use_container_width=True):
            st.session_state.listing = None
            st.session_state.selected_article = None
            st.session_state.analysis_result = None
            st.rerun()
    with col_b:
        if st.button("Fetch", type="primary", use_container_width=True):
            st.session_state.listing = None
            st.session_state.selected_article = None
            st.session_state.analysis_result = None
            st.rerun()


    if st.session_state.last_fetch_at:
        st.caption(f"Last fetch: {st.session_state.last_fetch_at}")


    st.divider()
    st.caption("**Pipeline**")
    st.caption("CP1 → listing")
    st.caption("CP2 → fetch detail + clean")
    st.caption("CP3 → classify + analyze")



# ---------- Header ----------
st.title("📰 Latest VinFast News")
st.caption("Select an article for the agent to fetch details and analyze")



# ---------- Step 1: Fetch listing ----------
if st.session_state.listing is None:
    with st.spinner("Fetching article list..."):
        res = fetch_listing(limit=limit)
    if not res.ok:
        st.error(f"Failed to fetch listing: {res.error}")
        st.stop()
    st.session_state.listing = res.articles
    st.session_state.last_fetch_at = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


articles = st.session_state.listing


col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("Total articles", len(articles))
col_info2.metric("Fetched at", st.session_state.last_fetch_at or "—")
col_info3.metric("Selected",
                 st.session_state.selected_article["article_id"]
                 if st.session_state.selected_article else "—")


st.divider()



# ---------- Step 2: Article list ----------
def render_article_card(art: dict, idx: int):
    with st.container():
        c1, c2 = st.columns([1, 4])
        with c1:
            thumb = art.get("thumbnail_url")
            if thumb:
                st.image(thumb, use_container_width=True)
        with c2:
            st.markdown(f"#### {idx}. {art.get('title','')}")
            st.caption(art.get("lead", "")[:280])
            meta = []
            if art.get("publish_time"):
                meta.append(f"{art['publish_time']}")
            if art.get("article_id"):
                meta.append(f"`{art['article_id']}`")
            if meta:
                st.caption(" · ".join(meta))


        btn_col1, btn_col2 = st.columns([1, 3])
        with btn_col1:
            if st.button("Analyze", key=f"btn_{art['article_id']}",
                         type="primary", use_container_width=True):
                st.session_state.selected_article = art
                st.session_state.analysis_result = None
                st.rerun()
        with btn_col2:
            st.markdown(f"[Open original]({art.get('share_url','')})")
        st.divider()



# If no article selected → show list
if not st.session_state.selected_article:
    st.subheader(f"List of {len(articles)} articles")
    for i, art in enumerate(articles, 1):
        render_article_card(art, i)
    st.stop()



# ---------- Step 3: Analysis view ----------
selected = st.session_state.selected_article


# Header of the selected article
top_l, top_r = st.columns([4, 1])
with top_l:
    st.subheader(f"{selected.get('title','')}")
    st.caption(selected.get("lead", ""))
with top_r:
    if st.button("Back to list", use_container_width=True):
        st.session_state.selected_article = None
        st.session_state.analysis_result = None
        st.rerun()


st.markdown(f"[Original article]({selected.get('share_url','')}) · "
            f"`{selected.get('article_id','')}`")
st.divider()



# Run pipeline if no result yet
if st.session_state.analysis_result is None:
    progress = st.progress(0, text="Starting...")
    status = st.empty()


    def cb(msg: str):
        status.info(msg)


    with st.spinner("Agent is processing..."):
        result = analyze_article(selected, progress_cb=cb)


    progress.progress(100, text="Done")
    status.empty()


    if not result.ok:
        st.error(f"Error: {result.error}")
        if result.detail:
            with st.expander("Cleaned content (debug)"):
                st.write(result.detail.get("paragraphs", [])[:10])
        st.stop()


    st.session_state.analysis_result = result
    st.rerun()


res = st.session_state.analysis_result
a = res.analysis



# ---------- Meta row ----------
st.markdown("### Analysis Summary")


m1, m2, m3, m4 = st.columns(4)
with m1:
    atype = a.get("article_type", "other")
    conf = a.get("article_type_confidence", 0)
    st.markdown(
        f"<span class='type-badge'>{atype}</span>",
        unsafe_allow_html=True,
    )
    st.caption(f"confidence: {conf:.2f}")
with m2:
    rel = a.get("relevance_to_distributor", {}) or {}
    st.metric("Relevance",
              f"{rel.get('score', 0):.2f}",
              rel.get("band", "—"))
with m3:
    vs = a.get("vinfast_stance", {}) or {}
    st.metric("Sentiment",
              f"{vs.get('sentiment', 0):+.2f}",
              vs.get("band", "—"))
with m4:
    st.metric("Total time", f"{res.total_ms} ms")


# Summary line
st.info(f"**{a.get('summary_1line','')}**")


# Latency breakdown
with st.expander("⏱Time & token details"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fetch detail", f"{res.detail_ms} ms")
    c2.metric("Classify", f"{res.classify_ms} ms")
    c3.metric("Analyze", f"{res.analyze_ms} ms")
    c4.metric("Total", f"{res.total_ms} ms")
    st.caption(f"Lens: {', '.join(res.lenses)}")



# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Signals", "Lens details", "Entities",
     "Ambiguities", "Original content"]
)



# ===== TAB 1: Key signals =====
with tab1:
    sigs = a.get("key_signals") or []
    if not sigs:
        st.info("No signals extracted.")
    else:
        groups = {
            "positive": ("POSITIVE", "signal-positive"),
            "negative": ("NEGATIVE", "signal-negative"),
            "neutral":  ("NEUTRAL", "signal-neutral"),
        }
        for pol, (label, css) in groups.items():
            items = [s for s in sigs if s.get("polarity") == pol]
            if not items:
                continue
            st.markdown(f"#### {label} ({len(items)})")
            for s in items:
                strength = s.get("strength", 0)
                band = s.get("band", "")
                st.markdown(
                    f"<div class='{css}'>"
                    f"<b>{s.get('signal','')}</b><br>"
                    f"<small>strength: <b>{strength:.2f}</b> · band: {band}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if s.get("angle"):
                    st.markdown(f"*Angle:* {s['angle']}")
                if s.get("action_hint"):
                    st.markdown(f"*Action:* {s['action_hint']}")
                if s.get("source_quote"):
                    st.caption(f"*\"{s['source_quote']}\"*")
                st.markdown("")



# ===== TAB 2: Lens extensions =====
with tab2:
    lens_keys = [k for k in a.keys()
                 if k in res.lenses and isinstance(a[k], (dict, list))]
    if not lens_keys:
        st.info("No lens extensions.")
    for lens in lens_keys:
        val = a[lens]
        if not val or (isinstance(val, dict) and not any(val.values())):
            continue
        with st.expander(f"**{lens}**", expanded=True):
            _render_nested(val)



# ===== TAB 3: Entities =====
with tab3:
    e = a.get("entities", {}) or {}
    col_a, col_b = st.columns(2)
    with col_a:
        _chip_list("Brands", e.get("brands", []))
        _chip_list("Models", e.get("models", []))
        _chip_list("Markets", e.get("markets", []))
    with col_b:
        _chip_list("People", e.get("people", []))


    nums = e.get("numbers", []) or []
    if nums:
        st.markdown("#### Figures")
        st.dataframe(
            [{"Value": f"{n.get('value','')} {n.get('unit','')}",
              "Context": n.get("context", "")} for n in nums],
            use_container_width=True,
            hide_index=True,
        )



# ===== TAB 4: Fuzzy ambiguities =====
with tab4:
    ambs = a.get("fuzzy_ambiguities") or []
    if not ambs:
        st.info("No ambiguous sentences detected.")
    for amb in ambs:
        st.markdown(f"> {amb.get('sentence','')}")
        for i in amb.get("interpretations", []):
            st.markdown(
                f"&nbsp;&nbsp;&nbsp;· **[strength {i.get('strength',0):.2f}]** "
                f"{i.get('reading','')}"
            )
        st.divider()



# ===== TAB 5: Raw content =====
with tab5:
    st.markdown("#### Metadata")
    st.json({
        "article_id": res.detail.get("article_id"),
        "title": res.detail.get("title"),
        "author": res.detail.get("author"),
        "publish_time": res.detail.get("publish_time"),
        "tags": res.detail.get("tags"),
        "word_count": res.detail.get("word_count"),
        "image_count": res.detail.get("image_count"),
    }, expanded=False)


    st.markdown("#### Cleaned content")
    for i, p in enumerate(res.detail.get("paragraphs", []), 1):
        st.markdown(f"**[{i:02d}]** {p}")


    if res.detail.get("figures"):
        st.markdown("#### Images")
        for f in res.detail["figures"]:
            if f.get("src"):
                st.image(f["src"], caption=f.get("caption") or f.get("alt", ""))



# ---------- Download ----------
st.divider()
col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 3])
with col_dl1:
    st.download_button(
        "Download analysis JSON",
        data=json.dumps(a, ensure_ascii=False, indent=2),
        file_name=f"analysis_{res.article_id}.json",
        mime="application/json",
        use_container_width=True,
    )
with col_dl2:
    st.download_button(
        "Download detail JSON",
        data=json.dumps(res.detail, ensure_ascii=False, indent=2),
        file_name=f"detail_{res.article_id}.json",
        mime="application/json",
        use_container_width=True,
    )



