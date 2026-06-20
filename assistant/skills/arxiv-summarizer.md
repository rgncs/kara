---
name: arxiv-summarizer
description: Find and summarize recent academic papers from arXiv on a topic.
triggers: arxiv, \bpaper(s)?\b, \bpreprint(s)?\b, recent research on, latest research, literature on
---
When I ask about recent papers, preprints, or research on a topic:

1. Call `web_search` for the topic scoped to arXiv, e.g. `"<topic> arxiv 2026"` or
   `site:arxiv.org <topic>`. Prefer the most recent results.
2. For the 3–5 most relevant hits, use `fetch_url` on the arXiv abstract page
   (`https://arxiv.org/abs/<id>`) to pull the real title, authors, and abstract —
   do not rely on memory for what a paper says.
3. Summarize each as: **Title** (authors, year) — one or two sentences on the
   contribution and why it matters, then the link.
4. End with a 2–3 sentence synthesis of the common themes or open questions.
5. Cite every paper with its arXiv URL. If the search returns nothing usable, say
   so plainly rather than inventing papers or citations.
