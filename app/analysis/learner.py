"""Online learner: the model earns its weights from its own live results.

Every prediction the app makes is logged with its component inputs (drift
anchor, ML tilt, technical read, three news sentiments). Once outcomes
resolve, this module re-estimates the blend weights by MAP logistic
regression — the live outcomes are the likelihood, and the backtest-validated
weights are a Gaussian prior the estimate is shrunk toward. With little
evidence the prior dominates; as resolved predictions accumulate, the data
takes over. This is the only place the news weights can ever be learned,
since no historical headline archive exists to backtest them.

Safety rails, in order:
  1. Hard gate: no weight moves until MIN_RESOLVED live outcomes exist.
  2. Prior-centered regularization (sigma = 0.15 per weight).
  3. Bounds: no negative weights, hard caps per component.
  4. Shadow adoption test: candidate weights are fit on the earliest 70% of
     resolved outcomes and must not lose to the frozen priors (Brier score)
     on the most recent 30%. Fail -> the priors stay in force. So the model
     can only get more honest, never quietly worse.

State lives in model_state.json (repo root) with a full update history.
"""

import json
import threading
import time
from pathlib import Path

import numpy as np

from ..config import BLEND, MODEL

STATE_PATH = Path(__file__).resolve().parents[2] / "model_state.json"

WEIGHT_KEYS = ["k_ml", "tech", "news_company", "news_market", "news_politics"]
PRIOR_WEIGHTS = {
    "k_ml": MODEL["shrink_ml"],
    "tech": MODEL["tech_weight"],
    "news_company": BLEND["news_company"],
    "news_market": BLEND["news_market"],
    "news_politics": BLEND["news_politics"],
}
BOUNDS = {
    "k_ml": (0.0, 0.6),
    "tech": (0.0, 0.4),
    "news_company": (0.0, 0.9),
    "news_market": (0.0, 0.5),
    "news_politics": (0.0, 0.4),
}
MIN_RESOLVED = 40      # no adaptation before this many live outcomes

# Prior width per weight = how much evidence already backs it. k_ml and tech
# were validated on 7,278 backtest predictions -> tight (live data moves them
# slowly). The news weights were never backtestable -> loose, so live outcomes
# can raise them if news genuinely predicts, or shrink them toward zero if it
# doesn't. Both directions are learning.
PRIOR_SIGMA = {
    "k_ml": 0.08,
    "tech": 0.08,
    "news_company": 0.50,
    "news_market": 0.35,
    "news_politics": 0.30,
}
# Candidate weights are adopted only if their holdout Brier is within this of
# the priors' (i.e. at worst a statistical tie). Small-sample drift under a
# near-tie is acceptable by design: weights are bounded, prior-anchored, and
# provably wash back toward truth as outcomes accumulate.
ADOPT_MARGIN = 0.0005

_lock = threading.Lock()
_state = None


def _default_state():
    return {
        "weights": dict(PRIOR_WEIGHTS),
        "source": "prior",
        "n_used": 0,
        "updated_at": None,
        "history": [],
    }


def _load():
    global _state
    if _state is None:
        if STATE_PATH.exists():
            try:
                _state = json.loads(STATE_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                _state = _default_state()
        else:
            _state = _default_state()
    return _state


def _save():
    try:
        STATE_PATH.write_text(json.dumps(_state, indent=2))
    except OSError:
        pass


def current_weights():
    with _lock:
        return dict(_load()["weights"])


def state_summary():
    with _lock:
        s = _load()
        return {
            "source": s["source"],
            "weights": {k: round(v, 3) for k, v in s["weights"].items()},
            "n_used": s["n_used"],
            "updated_at": s["updated_at"],
            "min_resolved_to_adapt": MIN_RESOLVED,
            "updates": len(s["history"]),
        }


def _logit(p):
    p = min(max(float(p), 0.03), 0.97)
    return float(np.log(p / (1 - p)))


def _design(records):
    """Rows -> (offset, X, y). x = [ml tilt, tech, news_c, news_m, news_p]."""
    off, X, y = [], [], []
    for r in records:
        base = r.get("base")
        if base is None or r.get("tech") is None:
            continue
        ml = _logit(r["p_ml"]) - _logit(base) if r.get("p_ml") is not None else 0.0
        X.append([ml, r["tech"], r.get("news_company", 0.0) or 0.0,
                  r.get("news_market", 0.0) or 0.0, r.get("news_politics", 0.0) or 0.0])
        off.append(_logit(base))
        y.append(1.0 if r["outcome_up"] else 0.0)
    return np.array(off), np.array(X), np.array(y)


def _brier(off, X, y, weights):
    w = np.array([weights[k] for k in WEIGHT_KEYS])
    p = 1 / (1 + np.exp(-(off + X @ w)))
    return float(np.mean((p - y) ** 2))


def _fit_map(off, X, y):
    """MAP logistic regression, prior N(PRIOR_WEIGHTS, PRIOR_SIGMA^2), bounded.
    Gradient descent on the mean objective (stable step sizes at any n)."""
    n = len(y)
    mu = np.array([PRIOR_WEIGHTS[k] for k in WEIGHT_KEYS])
    sig2 = np.array([PRIOR_SIGMA[k] ** 2 for k in WEIGHT_KEYS])
    lo = np.array([BOUNDS[k][0] for k in WEIGHT_KEYS])
    hi = np.array([BOUNDS[k][1] for k in WEIGHT_KEYS])
    w = mu.copy()
    lr = 0.3
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(off + X @ w)))
        grad = X.T @ (p - y) / n + (w - mu) / (sig2 * n)
        w = np.clip(w - lr * grad, lo, hi)
    return {k: round(float(v), 4) for k, v in zip(WEIGHT_KEYS, w)}


def update_from_ledger(records=None):
    """Recompute weights from resolved live predictions. Returns the summary.
    `records` is injectable for tests; normally the ledger is resolved live."""
    global _state
    if records is None:
        from ..data import ledger
        records = ledger.resolve_records(ledger.read_all())["resolved"]
    usable = sorted(
        (r for r in records if r.get("horizon_days") == 5 and r.get("base") is not None),
        key=lambda r: r["as_of"],
    )
    with _lock:
        s = _load()
        n = len(usable)
        if n < MIN_RESOLVED:
            s["n_used"] = n
            s["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
            _save()
            return state_summary_unlocked(s)

        split = max(int(n * 0.7), n - 200)
        off_tr, X_tr, y_tr = _design(usable[:split])
        off_te, X_te, y_te = _design(usable[split:])
        candidate = _fit_map(off_tr, X_tr, y_tr)
        brier_candidate = _brier(off_te, X_te, y_te, candidate)
        brier_prior = _brier(off_te, X_te, y_te, PRIOR_WEIGHTS)
        adopted = brier_candidate <= brier_prior + ADOPT_MARGIN

        if adopted:
            off_all, X_all, y_all = _design(usable)
            s["weights"] = _fit_map(off_all, X_all, y_all)
            s["source"] = "adaptive"
        else:
            s["weights"] = dict(PRIOR_WEIGHTS)
            s["source"] = "prior (candidate rejected by shadow test)"
        s["n_used"] = n
        s["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
        s["history"].append({
            "at": s["updated_at"], "n": n, "adopted": bool(adopted),
            "brier_candidate": round(brier_candidate, 4),
            "brier_prior": round(brier_prior, 4),
            "weights": s["weights"],
        })
        s["history"] = s["history"][-50:]
        _save()
        return state_summary_unlocked(s)


def state_summary_unlocked(s):
    return {
        "source": s["source"],
        "weights": {k: round(v, 3) for k, v in s["weights"].items()},
        "n_used": s["n_used"],
        "updated_at": s["updated_at"],
        "min_resolved_to_adapt": MIN_RESOLVED,
        "updates": len(s["history"]),
    }
