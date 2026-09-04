"""
Turns a free-text "what kind of book are you looking for" query into
something the content model can score - without a separate NLP model or
embeddings.

The content model's TF-IDF vocabulary is entirely genre tokens (from
`genre_tags.py`) plus author tokens, so a customer's raw sentence needs to
be reduced to that same vocabulary before it means anything to
`ContentModel`. This does that with whole-word matching against every raw
Goodreads tag string `genre_tags.GENRE_TAG_MAP` already knows how to
collapse into a canonical genre (e.g. typing "sci-fi" or "science fiction"
or "space opera"... only the tags actually in the map are recognized - see
the module-level note there for what's covered).

This is keyword matching, not semantic understanding - "a book that isn't
sad" won't be recognized as anti-tragedy, and a query with no genre words
at all ("something my grandmother would like") returns no matches. That's
an explicit, honest limitation stated in the API response and the
frontend, not hidden behind a vague "no results."
"""

import re

from .genre_tags import GENRE_SECTION, GENRE_TAG_MAP

_PATTERNS = None


def _patterns():
    global _PATTERNS
    if _PATTERNS is None:
        seen = {}
        for raw_tag, genre in GENRE_TAG_MAP.items():
            phrase = raw_tag.replace("-", " ")
            seen.setdefault(phrase, genre)
        _PATTERNS = [(re.compile(r"\b" + re.escape(phrase) + r"\b"), genre) for phrase, genre in seen.items()]
    return _PATTERNS


def detect_genres(query):
    """Returns {genre: match_count} for every canonical genre whose known
    raw-tag phrasing appears as a whole word/phrase in the query.

    Longer phrases are matched first and claim their character span so a
    short generic word contained in a longer, more specific phrase - e.g.
    "novel" (-> literary-fiction) inside "graphic novel" (-> graphic-novels)
    - isn't also counted on its own."""
    q = re.sub(r"[^a-z0-9\s-]", " ", query.lower()).replace("-", " ")
    q = " " + q + " "
    claimed = [False] * len(q)
    scores = {}
    for pattern, genre in sorted(_patterns(), key=lambda pg: -len(pg[0].pattern)):
        for m in pattern.finditer(q):
            if any(claimed[m.start() : m.end()]):
                continue
            scores[genre] = scores.get(genre, 0) + 1
            for i in range(m.start(), m.end()):
                claimed[i] = True
    return scores


def query_to_document(query, max_repeat=4):
    """Free text -> a token string in the same space ContentModel's TF-IDF
    vectorizer was fit on, or None if no genre keywords were recognized."""
    scores = detect_genres(query)
    if not scores:
        return None, {}
    tokens = []
    for genre, count in scores.items():
        tokens.extend([genre.replace("-", "_")] * min(count, max_repeat))
    sections = sorted({GENRE_SECTION.get(g, g) for g in scores})
    return " ".join(tokens), {"genres": sorted(scores), "sections": sections}
