"""Heuristic extraction of durable facts from a user message.

The write policy (spec §9): store durable facts — names, preferences, constraints,
where someone lives/works — not chitchat. This is code-driven rather than
model-driven: deterministic, testable, no extra model call per turn, and high
precision (we'd rather miss a fact than pollute memory with noise).

Two paths:
  1. An EXPLICIT request ("remember that…", "make a note…", "don't forget…",
     "save this to memory…") — capture exactly what the user asked to store.
  2. Implicit durable statements ("my name is…", "I prefer…", "I live in…").
"""
import re

# Verbs we normalize from first to third person by appending "s" (like -> likes).
_PREF_VERBS = "like|love|enjoy|prefer|hate|dislike"

# Lead-ins by which a user explicitly asks Kara to remember something. The trailing
# group captures the content to store.
_REMEMBER_RE = re.compile(
    r"\b(?:"
    r"remember(?:\s+that|\s+to)?|"
    r"make a note(?:\s+that)?|note(?:\s+that)?|jot down(?:\s+that)?|"
    r"don'?t forget(?:\s+that)?|keep in mind(?:\s+that)?|"
    r"(?:save|store|commit|add)(?:\s+this|\s+that)?\s+to\s+(?:your\s+)?(?:memory|long[- ]term memory)|"
    r"for (?:future reference|the record)"
    r")\s*[:,-]?\s+(.+)",
    re.I,
)


def _norm_pronouns(s: str) -> str:
    """Rewrite first-person phrasing to refer to 'the user' (for clean recall text)."""
    s = s.strip().rstrip(".!?").strip()   # a request phrased as a question ("…1984?")
    s = re.sub(r"\bI have\b", "the user has", s, flags=re.I)
    s = re.sub(r"\bI've\b", "the user has", s, flags=re.I)
    s = re.sub(r"\bI[' ]?a?m\b", "the user is", s, flags=re.I)   # I'm / I am
    s = re.sub(r"\bI do\b", "the user does", s, flags=re.I)
    s = re.sub(r"\bI\b", "the user", s, flags=re.I)
    s = re.sub(r"\bmy\b", "the user's", s, flags=re.I)
    s = re.sub(r"\bme\b", "the user", s, flags=re.I)
    s = re.sub(r"\bmine\b", "the user's", s, flags=re.I)
    return s[0].upper() + s[1:] if s else s


# (pattern, builder) — builders return the fact string to store.
_RULES = [
    (re.compile(r"\bmy name is ([A-Z][\w'-]+)", re.I),
     lambda m: f"The user's name is {m.group(1)}"),
    (re.compile(r"\b(?:call me|i go by) ([A-Z][\w'-]+)", re.I),
     lambda m: f"The user goes by {m.group(1)}"),
    (re.compile(rf"\bi (?:really |also )?({_PREF_VERBS}) (.+)", re.I),
     lambda m: f"The user {m.group(1).lower()}s {m.group(2).strip().rstrip('.!')}"),
    (re.compile(r"\bi (live|work) (in|at|as) (.+)", re.I),
     lambda m: f"The user {m.group(1).lower()}s {m.group(2).lower()} {m.group(3).strip().rstrip('.!')}"),
    (re.compile(r"\bmy favou?rite (.+?) is (.+)", re.I),
     lambda m: f"The user's favorite {m.group(1).strip()} is {m.group(2).strip().rstrip('.!')}"),
    (re.compile(r"\bi['’]?m (?:a|an) ([\w][\w /+-]{1,40})", re.I),
     lambda m: f"The user is a {m.group(1).strip().rstrip('.!')}"),
]


# Junk a "fact" must not be: recursive memory meta-instructions or bare fragments.
_JUNK_RE = re.compile(r"\b(make a note|to memory|remember|note of it|jot down|"
                      r"go deeper|tell me more)\b", re.I)
# Reject "facts" that assert absence/uncertainty rather than information.
_NON_FACT = re.compile(r"\b(not known|unknown|don'?t (?:have|know)|no information|"
                       r"isn'?t (?:known|available)|n/?a)\b", re.I)


def _is_meaningful(fact: str) -> bool:
    """Reject low-value 'facts' — fragments, questions, meta-instructions, or
    assertions of not-knowing (common noise from messy input or a confused model)."""
    f = fact.strip()
    if len(f.split()) < 3:           # too short to be a durable fact ("That?")
        return False
    # A genuine question (starts with an interrogative) isn't a fact — but a
    # statement that merely ended a question-phrased request ("…1984?") is fine,
    # since the trailing "?" was already stripped during normalization.
    if re.match(r"(?i)(what|when|where|why|how|who|which|can|could|would|should|do|does|did|is|are|am)\b", f):
        return False
    if _JUNK_RE.search(f) or _NON_FACT.search(f):
        return False
    return True


def _phrase_fact(content: str) -> str:
    """Turn an explicitly-remembered clause into a clean stored fact: prefer a
    structured rule (e.g. "I prefer X" -> "The user prefers X"), else normalize pronouns."""
    content = content.strip().rstrip(".!").strip()
    for pattern, build in _RULES:
        m = pattern.search(content)
        if m:
            return build(m).strip()
    return _norm_pronouns(content)


# Words that signal the user wants something stored — triggers deterministic capture.
_CUE_RE = re.compile(r"\b(remember|remind|don'?t forget|forget|keep in mind|"
                     r"make a note|note that|memori[sz]e|save (?:this|that|it|to memory)|"
                     r"keep track|for (?:the record|future reference))\b", re.I)


def has_remember_cue(text: str) -> bool:
    """True when the message explicitly signals an intent to remember/remind."""
    return bool(_CUE_RE.search(text))


# Strip the cue/politeness wrapper from a remember-request, wherever it sits in the
# sentence ("remember that X" or "X, can you remember that?").
_CUE_STRIP = re.compile(
    r"(?i)\b(?:can you|could you|would you|please|i want you to|you (?:should|can))?\s*"
    r"(?:do you (?:remember|know|recall)(?:\s+(?:that|if))?|"
    r"remember(?:ing)?(?:\s+(?:that|this|to))?|remind me(?:\s+to)?|don'?t forget(?:\s+that)?|"
    r"keep in mind(?:\s+that)?|make a note(?:\s+(?:that|of it))?|note that|jot down(?:\s+that)?|"
    r"save (?:this|that|it)(?:\s+to memory)?|memori[sz]e(?:\s+that)?|"
    r"for (?:the record|future reference))\s*[:,]?"
)


def remembered_content(text: str) -> list[str]:
    """Deterministically turn an explicit remember-request into a storable fact —
    strip the cue, normalize pronouns, keep the user's actual content. Reliable
    where the model's extraction is flaky (it never silently drops details)."""
    is_reminder = bool(re.search(r"(?i)\bremind\b", text))
    body = _CUE_STRIP.sub(" ", text)
    body = re.sub(r"\s+", " ", body).strip(" ,.?!:;")
    if not body:
        return []
    if is_reminder:
        fact = _norm_pronouns("Remind Wontaek to " + body)
    else:
        fact = _norm_pronouns(body)
    return [fact] if _is_meaningful(fact) else []


def extract_facts(text: str) -> list[str]:
    """Return a de-duplicated list of durable facts found in `text` (often empty)."""
    # 1) Explicit "remember this" request — store exactly what was asked, whole.
    m = _REMEMBER_RE.search(text)
    if m:
        fact = _phrase_fact(m.group(1))
        return [fact] if fact and _is_meaningful(fact) else []

    # 2) Implicit durable statements. Split into clauses so several facts in one
    # message are each captured. Only split on " and " when it begins a new
    # first-person statement (… and I …), so values like "salt and pepper" or
    # "live in X and work at Y" stay intact.
    facts: list[str] = []
    for clause in re.split(r"(?i)[.!?\n]+|\s*,?\s+and\s+(?=(?:I|my)\b)", text):
        clause = clause.strip()
        if not clause:
            continue
        for pattern, build in _RULES:
            m = pattern.search(clause)
            if m:
                fact = build(m).strip()
                if fact and fact not in facts and _is_meaningful(fact):
                    facts.append(fact)
                break  # one fact per clause — first (most specific) rule wins
    return facts
