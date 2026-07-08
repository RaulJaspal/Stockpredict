"""Headline sentiment: VADER extended with a finance/politics lexicon.

VADER is a well-validated rule-based sentiment model for short text. Plain
VADER misses market language ("downgrade", "beats estimates", "tariffs"),
so we extend its lexicon with financial terms and apply a phrase-level pass
for multi-word expressions it cannot see ("record high", "cuts guidance").
Scores are in [-1, 1].
"""

import re
import time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Valence values use VADER's native -4..+4 scale.
_FINANCE_LEXICON = {
    # earnings & analyst actions
    "upgrade": 1.5, "upgraded": 1.5, "upgrades": 1.5,
    "downgrade": -1.5, "downgraded": -1.5, "downgrades": -1.5,
    "beats": 1.6, "misses": -1.6, "outperform": 1.3, "underperform": -1.3,
    "overweight": 1.0, "underweight": -1.0, "guidance": 0.0,
    "profit": 1.0, "profits": 1.0, "loss": -1.0, "losses": -1.2,
    "dividend": 0.8, "buyback": 1.2, "buybacks": 1.2,
    # price action
    "surge": 1.9, "surges": 1.9, "surged": 1.9,
    "soar": 2.1, "soars": 2.1, "soared": 2.1,
    "rally": 1.7, "rallies": 1.7, "rallied": 1.7,
    "rebound": 1.3, "rebounds": 1.3, "jump": 1.3, "jumps": 1.3, "jumped": 1.3,
    "plunge": -2.1, "plunges": -2.1, "plunged": -2.1,
    "plummet": -2.3, "plummets": -2.3, "plummeted": -2.3,
    "tumble": -1.7, "tumbles": -1.7, "tumbled": -1.7,
    "slump": -1.7, "slumps": -1.7, "slumped": -1.7,
    "sink": -1.4, "sinks": -1.4, "slide": -1.1, "slides": -1.1,
    "selloff": -1.9, "sell-off": -1.9, "crash": -2.5, "crashes": -2.5,
    # corporate events
    "layoffs": -1.5, "layoff": -1.4, "bankruptcy": -3.0, "bankrupt": -3.0,
    "insolvency": -2.6, "default": -1.7, "restructuring": -0.8,
    "lawsuit": -1.2, "sues": -1.2, "sued": -1.2, "fine": -0.8, "fined": -1.0,
    "probe": -1.0, "investigation": -0.8, "fraud": -2.4, "scandal": -1.8,
    "recall": -1.3, "recalls": -1.3, "breach": -1.6, "hack": -1.4, "outage": -1.2,
    "acquisition": 0.8, "merger": 0.6, "ipo": 0.5, "expansion": 1.0, "hiring": 1.0,
    # macro & politics
    "recession": -2.2, "downturn": -1.6, "slowdown": -1.2, "stagflation": -2.0,
    "inflation": -0.8, "deflation": -1.0, "stimulus": 1.2,
    "tariff": -1.0, "tariffs": -1.0, "sanctions": -1.0, "embargo": -1.2,
    "war": -2.0, "invasion": -2.2, "conflict": -1.4, "ceasefire": 1.2,
    "shutdown": -1.4, "strike": -1.2, "strikes": -1.2, "unrest": -1.4,
    "bailout": -0.6, "austerity": -1.0,
}

# Phrase-level adjustments VADER's unigram lexicon cannot capture.
_PHRASE_RULES = [
    (re.compile(
        r"record high|all[- ]time high|beats?\s+(?:wall street |analyst |market )?"
        r"(?:expectations|estimates|forecasts)|raises?\s+(?:guidance|forecast|outlook)|"
        r"better[- ]than[- ]expected|rate cut|rates? (?:were )?cut"), +0.35),
    (re.compile(
        r"record low|52[- ]week low|misses?\s+(?:expectations|estimates|forecasts)|"
        r"cuts?\s+(?:guidance|forecast|outlook)|worse[- ]than[- ]expected|"
        r"profit warning|rate hike|rate rise"), -0.35),
]

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(_FINANCE_LEXICON)


def score_text(text):
    """Sentiment of one headline/summary in [-1, 1]."""
    score = _analyzer.polarity_scores(text or "")["compound"]
    lowered = (text or "").lower()
    for pattern, boost in _PHRASE_RULES:
        if pattern.search(lowered):
            score += boost
    return max(-1.0, min(1.0, score))


def label(score):
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


def annotate(articles):
    """Attach a sentiment score and label to each article, in place."""
    for article in articles:
        text = f"{article['title']}. {article.get('summary', '')}"
        article["sentiment"] = round(score_text(text), 3)
        article["sentiment_label"] = label(article["sentiment"])
    return articles


def aggregate(articles, half_life_hours=36.0):
    """Recency-weighted average sentiment: a headline `half_life_hours` old
    counts half as much as one published right now. Undated items get a
    72-hour (low) weight."""
    now = time.time()
    weighted_sum = weight_total = 0.0
    for article in articles:
        score = article.get("sentiment")
        if score is None:
            continue
        age_hours = (now - article["published"]) / 3600.0 if article.get("published") else 72.0
        weight = 0.5 ** (max(age_hours, 0.0) / half_life_hours)
        weighted_sum += weight * score
        weight_total += weight
    if weight_total == 0:
        return {"score": 0.0, "n": 0, "label": "neutral"}
    avg = weighted_sum / weight_total
    return {"score": round(avg, 3), "n": len(articles), "label": label(avg)}
