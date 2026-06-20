---
name: deep-research
description: Multi-source web research with synthesis and citations.
triggers: research (this|the|on)?, deep dive, do some research, look into, investigate, compare .* (vs|versus|against), pros and cons of, what are the options for
---
When I ask you to research a topic or compare options:

1. Run several `web_search` queries from different angles, not just one — e.g. the
   overview, the criticisms/limitations, and the most recent developments. Today's
   date is after your training cutoff, so search first; don't answer from memory.
2. Use `fetch_url` to read the most authoritative 2–4 sources in full rather than
   trusting snippets, especially for numbers, dates, or contested claims.
3. Cross-check: when sources disagree, say so and note which is more credible
   (primary source, official docs, recent date) instead of picking silently.
4. Answer with a short synthesis, then the key points as bullets, then a
   "Sources" list of the URLs you actually used, cited inline as [1], [2].
5. Separate what you found from what you're inferring. If the evidence is thin or
   conflicting, say that rather than projecting false confidence.
