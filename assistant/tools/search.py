"""Web search + page fetch (Phase 3).

web_search returns numbered, citeable text blocks ("[n] title / url / snippet")
rather than raw JSON, so the model can cite [1], [2] inline. The provider is
isolated behind _search_tavily / _search_searxng — switching is a config flip
(SEARCH_PROVIDER), no change to web_search's contract or the agent loop.

Network failures degrade to a readable message the model can relay, never a crash.
"""
import logging

import config

log = logging.getLogger("assistant.search")


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No results found."
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] {r['title']}\n{r['url']}\n{r['content']}")
    return "\n\n".join(blocks)


def _search_tavily(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=config.require_tavily_key())
    resp = client.search(query=query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""),
         "content": (r.get("content") or "").strip()}
        for r in resp.get("results", [])
    ]


def _search_searxng(query: str, max_results: int) -> list[dict]:
    import httpx
    resp = httpx.get(
        f"{config.SEARXNG_URL}/search",
        params={"q": query, "format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("results", [])[:max_results]
    return [
        {"title": it.get("title", ""), "url": it.get("url", ""),
         "content": (it.get("content") or "").strip()}
        for it in items
    ]


def web_search(query: str, max_results: int = config.MAX_SEARCH_RESULTS) -> str:
    """Search the web and return numbered, citeable result blocks."""
    log.debug("web_search provider=%s query=%r", config.SEARCH_PROVIDER, query)
    try:
        if config.SEARCH_PROVIDER == "searxng":
            results = _search_searxng(query, max_results)
        else:
            results = _search_tavily(query, max_results)
    except Exception as e:  # noqa: BLE001 — degrade gracefully (spec §10)
        log.debug("web_search failed: %s", e)
        return f"ERROR: couldn't reach the web ({e}). Tell the user the search failed."
    return _format_results(results)


def fetch_url(url: str) -> str:
    """Fetch a page and return cleaned, truncated text (for when snippets aren't enough)."""
    import httpx
    from bs4 import BeautifulSoup

    log.debug("fetch_url %s", url)
    try:
        resp = httpx.get(
            url, timeout=20, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Kara/1.0)"},
        )
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: couldn't fetch {url} ({e})"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
    text = "\n".join(lines)
    if len(text) > config.MAX_FETCH_CHARS:
        text = text[:config.MAX_FETCH_CHARS] + f"\n... [truncated to {config.MAX_FETCH_CHARS} chars]"
    return text


WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Returns numbered results to cite as [1], [2].",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "description": "How many results (default 5)."},
            },
            "required": ["query"],
        },
    },
}

FETCH_URL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch the readable text of a web page (use when a search snippet isn't enough, or the user pastes a link).",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The full URL to fetch."}},
            "required": ["url"],
        },
    },
}
