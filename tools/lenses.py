"""
Defines analysis lenses for each article_type.
- LENS_MAP: article_type -> [lens, ...]
- LENS_SCHEMA: lens -> JSON schema snippet (additional fields)
- LENS_INSTRUCTION: lens -> LLM language and evaluation guidelines
"""
from __future__ import annotations
from typing import Any
from utils.logger import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------------------------------ #
# 1. Routing
# ------------------------------------------------------------------ #
LENS_MAP: dict[str, list[str]] = {
    "sales_ranking":     ["competitive_position", "growth_signal"],
    "product_launch":    ["product_positioning", "sales_enablement"],
    "pricing_promotion": ["margin_impact", "demand_lever"],
    "policy_regulation": ["compliance_risk", "demand_lever"],
    "recall_service":    ["reputation_risk", "service_ops"],
    "event_launch":      ["channel_expansion", "brand_presence"],
    "finance_market":    ["investor_signal", "cost_pressure"],
    "partnership":       ["ecosystem_play", "capability_gap"],
    "competition":       ["threat_assessment", "counter_positioning"],
    "technology":        ["product_differentiation", "training_need"],
    "consumer_opinion":  ["perception_signal", "objection_handling"],
    "macro_market":      ["demand_context", "timing_signal"],
    "other":             ["general_signal"],
}


def get_lenses(article_type: str,
               secondary_type: str | None = None) -> list[str]:
    """
    Returns a list of deduplicated lenses. Secondary type adds extra lenses if
    the primary type has fewer than 3 lenses.
    """
    logger.debug("Routing lenses for article_type='%s', secondary_type='%s'", article_type, secondary_type)
    
    lenses = list(LENS_MAP.get(article_type, LENS_MAP["other"]))
    if secondary_type and len(lenses) < 3:
        logger.debug("Primary lenses count < 3 (%d), fetching additional lenses from secondary_type='%s'", len(lenses), secondary_type)
        for l in LENS_MAP.get(secondary_type, []):
            if l not in lenses:
                lenses.append(l)

    selected_lenses = lenses[:4]  # Maximum 4 lenses to prevent prompt bloating
    logger.info("Selected lenses for type '%s': %s", article_type, selected_lenses)
    return selected_lenses


# ------------------------------------------------------------------ #
# 2. Schema snippet for each lens
# ------------------------------------------------------------------ #
LENS_SCHEMA: dict[str, dict[str, Any]] = {
    "competitive_position": {
        "competitors_mentioned": [
            {"name": "string", "metric": "string", "value": "string",
             "delta_vs_vinfast": "string"}
        ],
        "rank_or_share": {"metric": "string", "value": "string",
                          "rank": "integer|null", "band": "string"},
        "gap_analysis": {"to_leader": "string", "to_next": "string",
                         "direction": "narrowing|widening|stable|unknown"},
    },
    "growth_signal": {
        "growth_rate": {"value_pct": 0.0, "period": "string",
                        "vs_industry_pct": 0.0, "band": "string"},
        "sustainability": {"score": 0.0, "band": "string", "reason": "string"},
    },
    "product_positioning": {
        "segment": "string",
        "price": {"value": 0.0, "unit": "string|null"},
        "usps": ["string"],
        "direct_competitors": ["string"],
        "training_points": ["string"],
    },
    "sales_enablement": {
        "collaterals_to_update": ["string"],
        "target_buyer_profile": "string",
        "objection_prep": [{"objection": "string", "response": "string"}],
    },
    "margin_impact": {
        "price_change_pct": 0.0,
        "promo_type": "string",
        "duration": "string",
        "margin_risk": {"score": 0.0, "band": "string", "reason": "string"},
    },
    "demand_lever": {
        "lever_type": "policy|promo|infrastructure|sentiment|other",
        "expected_demand_delta_pct": 0.0,
        "affected_segments": ["string"],
        "timeline": "string",
    },
    "compliance_risk": {
        "regulation": {"name": "string", "authority": "string",
                       "effective_date": "string|null",
                       "expiry_date": "string|null"},
        "deadline": "string|null",
        "impact_on_operations": {
            "price_change_pct": 0.0,
            "documents_to_update": ["string"],
            "training_needed": False,
        },
    },
    "reputation_risk": {
        "risk_type": "recall|scandal|quality|legal|accident|none",
        "scope": {"units_affected": "integer|null", "regions": ["string"]},
        "severity": {"score": 0.0, "band": "string"},
        "response_readiness": {
            "call_center_ready": False,
            "parts_availability": "string",
            "estimated_handling_days": "integer|null",
        },
        "recommended_silence": False,  # whether social media post should be avoided
    },
    "service_ops": {
        "affected_models": ["string"],
        "affected_regions": ["string"],
        "workflow_changes": ["string"],
        "staff_training_required": False,
        "estimated_workload_days": "integer|null",
    },
    "channel_expansion": {
        "channel_type": "showroom|dealer|online|service_center|other",
        "location": {"city": "string", "region": "string",
                     "coordinates": "string|null"},
        "capacity": {"showroom_area_sqm": "integer|null",
                     "service_bays": "integer|null"},
        "competitors_nearby": ["string"],
    },
    "brand_presence": {
        "event_type": "string",
        "audience_reach_est": "integer|null",
        "media_coverage_tier": "local|regional|national|international|unknown",
        "asset_reuse": ["string"],
    },
    "investor_signal": {
        "metric": "string",
        "value": "string",
        "direction": "positive|negative|neutral|mixed",
        "materiality": {"score": 0.0, "band": "string"},
    },
    "cost_pressure": {
        "cost_type": "string",
        "magnitude": "string",
        "transfer_to_price_risk": {"score": 0.0, "band": "string"},
    },
    "ecosystem_play": {
        "partner_name": "string",
        "partner_type": "supplier|distributor|tech|gov|other",
        "strategic_fit": {"score": 0.0, "band": "string"},
        "gaps_filled": ["string"],
    },
    "capability_gap": {
        "capability": "string",
        "current_state": "string",
        "target_state": "string",
        "timeline": "string",
    },
    "threat_assessment": {
        "threat_type": "product|price|channel|marketing|technology",
        "threat_level": {"score": 0.0, "band": "string"},
        "time_horizon": "immediate|3m|6m|12m|unknown",
        "affected_segments": ["string"],
    },
    "counter_positioning": {
        "counter_message": "string",
        "channels": ["string"],
        "risks_of_counter": ["string"],
    },
    "product_differentiation": {
        "tech_name": "string",
        "advantage_vs_competitors": ["string"],
        "customer_benefit": "string",
        "proof_points": ["string"],
    },
    "training_need": {
        "topic": "string",
        "audience": "sales|service|both",
        "priority": {"score": 0.0, "band": "string"},
        "materials_needed": ["string"],
    },
    "perception_signal": {
        "sentiment_score": 0.0,
        "themes": [{"theme": "string", "weight": 0.0}],
        "representative_quotes": ["string"],
    },
    "objection_handling": {
        "objections": [{"objection": "string", "frequency": 0.0,
                        "suggested_response": "string"}],
    },
    "demand_context": {
        "market_size": "string",
        "growth_direction": "up|down|flat|mixed",
        "vinfast_position": "string",
    },
    "timing_signal": {
        "window": "string",
        "action_deadline": "string|null",
        "reason": "string",
    },
    "general_signal": {
        "signal": "string",
        "implication": "string",
        "suggested_action": "string",
    },
}


# ------------------------------------------------------------------ #
# 3. Instruction (language) for each lens
# ------------------------------------------------------------------ #
LENS_INSTRUCTION: dict[str, str] = {
    "competitive_position":
        "Compare VinFast with competitors MENTIONED IN THE ARTICLE ONLY. Do not add competitors "
        "outside the text. If ranking data is missing, set rank_or_share = null.",
    "growth_signal":
        "Assess growth momentum: value_pct + vs industry (if provided in article). "
        "sustainability is a fuzzy assessment and must include a reason.",
    "product_positioning":
        "Product positioning: segment/price/USP/direct competitors. Use ONLY information "
        "present in the article. training_points = key takeaways for the sales team.",
    "sales_enablement":
        "List collateral to update + target buyer persona + anticipated objections.",
    "margin_impact":
        "Analyze profit margin impact. If price is not mentioned, set "
        "price_change_pct = 0, promo_type = 'unknown', band = 'medium'.",
    "demand_lever":
        "Identify demand levers: policy / promo / infrastructure / sentiment. "
        "If unclear, set lever_type='other', timeline='unknown'.",
    "compliance_risk":
        "How policies/regulations affect DISTRIBUTOR OPERATIONS: "
        "deadline, paperwork, pricing, training. DO NOT comment on politics. "
        "If dates are missing, set to null instead of hallucinating.",
    "reputation_risk":
        "STRICT honesty required. DO NOT spin negative news into marketing opportunities. "
        "Focus on: affected scope, severity, operational readiness. "
        "recommended_silence=True if social media posting should be avoided.",
    "service_ops":
        "Service operations: affected models, regions, workflow adjustments, "
        "training requirements, estimated workload in days.",
    "channel_expansion":
        "Channel expansion: channel type, location, capacity, nearby competitors.",
    "brand_presence":
        "Brand presence via events: event type, reach, media coverage tier, asset "
        "reusability for future marketing.",
    "investor_signal":
        "Financial/Investor signals: metric, value, direction, materiality.",
    "cost_pressure":
        "Cost pressures: type, magnitude, risk of passing cost to consumer price.",
    "ecosystem_play":
        "Ecosystem partnership: partner name, type, strategic_fit, capabilities/gaps filled.",
    "capability_gap":
        "Capability gaps: capability, current vs target state, implementation timeline.",
    "threat_assessment":
        "Threat severity from competitors: threat type, level, time horizon, affected market segments.",
    "counter_positioning":
        "Counter positioning: rebuttal messaging, channels, counter-messaging risks.",
    "product_differentiation":
        "Technology differentiation: tech name, competitive advantages, customer benefit, proof points.",
    "training_need":
        "Training requirements: topic, target audience, priority level, required materials.",
    "perception_signal":
        "Perception signals: sentiment, key themes with weights, representative quotes. "
        "Quotes MUST BE EXACT verbatim extracts from the article.",
    "objection_handling":
        "Objection handling: specific objections, frequency (0.0 to 1.0), suggested response.",
    "demand_context":
        "Market demand context: market size, growth trajectory, VinFast positioning.",
    "timing_signal":
        "Timing signals: optimal window for action, action deadline, rationale.",
    "general_signal":
        "General signals: signal statement, operational implication, suggested action.",
}


# ------------------------------------------------------------------ #
# 4. Helpers for builder
# ------------------------------------------------------------------ #
def build_schema_hint(article_type: str, lenses: list[str]) -> dict:
    """Returns a full JSON schema dictionary (base + active lens extensions)."""
    logger.debug("Building schema hint for article_type='%s' with lenses=%s", article_type, lenses)
    
    base = {
        "article_id": "string",
        "article_type": article_type,
        "article_type_confidence": 0.0,
        "secondary_type": "string|null",
        "summary_1line": "string <=25 words",
        "relevance_to_distributor": {
            "score": 0.0, "band": "very_low|low|medium|high|very_high",
            "reason": "string",
        },
        "vinfast_stance": {
            "sentiment": 0.0, "confidence": 0.0,
            "band": "very_low|low|medium|high|very_high",
            "evidence": ["verbatim quote"],
        },
        "entities": {
            "brands": ["string"], "models": ["string"],
            "markets": ["string"], "people": ["string"],
            "numbers": [{"value": "string", "unit": "string",
                         "context": "string"}],
        },
        "key_signals": [
            {
                "signal": "string",
                "polarity": "positive|negative|neutral",
                "strength": 0.0,
                "band": "very_low|low|medium|high|very_high",
                "angle": "string",
                "action_hint": "string",
                "source_quote": "string",
            }
        ],
        "fuzzy_ambiguities": [
            {
                "sentence": "string",
                "interpretations": [
                    {"reading": "string", "strength": 0.0}
                ],
            }
        ],
        "meta": {"model": "string", "analyzed_at": "ISO8601",
                 "prompt_version": "string"},
    }
    
    for lens in lenses:
        if lens not in LENS_SCHEMA:
            logger.warning("Lens '%s' missing from LENS_SCHEMA; falling back to 'general_signal'", lens)
        base[lens] = LENS_SCHEMA.get(lens, LENS_SCHEMA["general_signal"])
        
    logger.debug("Successfully generated schema hint containing %d keys", len(base))
    return base


def build_lens_block(lenses: list[str]) -> str:
    """Generates prompt instruction block for the LLM based on active lenses."""
    logger.debug("Generating prompt instruction block for lenses: %s", lenses)
    
    blocks = []
    for l in lenses:
        instruction = LENS_INSTRUCTION.get(l, "")
        if not instruction:
            logger.warning("Lens '%s' has no instruction defined in LENS_INSTRUCTION", l)
        blocks.append(f"### Lens: {l}\n{instruction}")
        
    return "\n\n".join(blocks)