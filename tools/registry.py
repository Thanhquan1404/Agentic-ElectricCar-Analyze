from tools.vnexpress_tool import (
    vnexpress_fetch_vinfast_articles,
    vinfast_normalize_articles,
)
from tools.classify import classify_article_type
from tools.analyze import information_analyze
from utils.logger import setup_logger 

logger = setup_logger(__name__)

from tools.vnexpress_detail import vnexpress_fetch_article_detail

TOOL_REGISTRY = {
    "vnexpress_fetch_vinfast_articles": vnexpress_fetch_vinfast_articles,
    "vinfast_normalize_articles": vinfast_normalize_articles,
    "vnexpress_fetch_article_detail": vnexpress_fetch_article_detail,
    "classify_article_type": classify_article_type,
    "information_analyze": information_analyze,
}

def dispatch_tool(name: str, arguments: dict):
    """
    Dispatches and executes a tool function registered in TOOL_REGISTRY with the given arguments.
    """
    logger.debug("Dispatching tool '%s' with arguments: %s", name, arguments)
    fn = TOOL_REGISTRY.get(name)
    if not fn:
        logger.error("Tool dispatch failed: unknown tool '%s'", name)
        return {"ok": False, "error": f"unknown_tool: {name}"}
    try:
        result = fn(**arguments)
        logger.info("Tool '%s' executed successfully", name)
        return result
    except TypeError as e:
        logger.error("Tool '%s' failed due to invalid arguments: %s", name, e)
        return {"ok": False, "error": f"bad_arguments: {e}"}