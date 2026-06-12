from __future__ import annotations

"""Junk-text heuristics for mining and mixture building.

High base CE alone does not mean "hard reasoning": SEO spam, product listings,
and keyword-stuffed pages are hard to predict because they are noise, not
because they need better latent evidence. These filters encode the junk
patterns actually observed in mined hard prefixes (escort/dating spam, outlet
listings, keyword stuffing with no function words, repetitive boilerplate).
"""

import re
from typing import List, Tuple

_WORD_RE = re.compile(r"[A-Za-z']+")

# Function words: keyword-stuffed listings have almost none of these.
_STOPWORDS = frozenset(
    "the and of to a in is for on with that it as are was at by this be or "
    "from an have has had not but they you we he she his her its their which "
    "will would can could been were what when how all if there".split()
)

_SPAM_PATTERNS = (
    "escort", "dating", "flirt", "hookup", "casino", "viagra", "porn",
    "free shipping", "fast shipping", "on sale", "discount", "outlet",
    "best price", "coupon", "promo code", "call or send", "send a sms",
    "click submit", "subscribe now", "limited offer", "buy now",
)


def junk_signals(text: str) -> List[Tuple[str, float]]:
    """Return the list of (reason, value) junk signals that fired."""

    signals: List[Tuple[str, float]] = []
    words = _WORD_RE.findall(text)
    if len(words) < 20:
        signals.append(("too_short", float(len(words))))
        return signals

    lowered = [word.lower() for word in words]
    total_chars = max(len(text), 1)
    alpha_ratio = sum(ch.isalpha() or ch.isspace() for ch in text) / total_chars
    stopword_ratio = sum(word in _STOPWORDS for word in lowered) / len(lowered)
    unique_ratio = len(set(lowered)) / len(lowered)
    lower_text = text.lower()
    spam_hits = sum(lower_text.count(pattern) for pattern in _SPAM_PATTERNS)

    # Keyword stuffing / listings read as nouns glued together without grammar.
    if stopword_ratio < 0.10:
        signals.append(("low_stopword_ratio", round(stopword_ratio, 3)))
    # Markup or symbol-heavy boilerplate.
    if alpha_ratio < 0.55:
        signals.append(("low_alpha_ratio", round(alpha_ratio, 3)))
    # Repeated keyword loops ("battery operated light fixture battery ...").
    if unique_ratio < 0.45:
        signals.append(("low_unique_ratio", round(unique_ratio, 3)))
    if spam_hits >= 2:
        signals.append(("spam_keywords", float(spam_hits)))
    return signals


def looks_like_junk(text: str) -> bool:
    return bool(junk_signals(text))
