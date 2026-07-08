/* StockPredict dashboard. All feed-derived text is inserted with textContent —
   never innerHTML — because headlines are untrusted data. */
"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  ticker: null,
  data: null,        // /api/predict payload
  candles: [],       // /api/history payload (500 sessions)
  rangeDays: 126,
  newsTab: "company",
  chart: null,
  series: null,
  screener: null,    // /api/screener payload
  confFilter: "all",
};

const QUICK_PICKS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "^GSPC"];
const RANGES = [["3M", 63], ["6M", 126], ["1Y", 252]];

/* ---------------- helpers ---------------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* keep default */ }
    throw new Error(detail);
  }
  return res.json();
}

const fmt = {
  price(v) {
    if (v == null) return "—";
    const dp = Math.abs(v) >= 1000 ? 2 : Math.abs(v) >= 1 ? 2 : 4;
    return v.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
  },
  pct(v, signed = true) {
    if (v == null) return "—";
    return `${signed && v > 0 ? "+" : ""}${v.toFixed(2)}%`;
  },
  compact(v) {
    if (v == null) return "—";
    return Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(v);
  },
  date(iso) {
    if (!iso) return "—";
    const d = new Date(iso + "T00:00:00");
    return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  },
  ago(epoch) {
    if (!epoch) return "";
    const mins = Math.max(0, Math.round((Date.now() / 1000 - epoch) / 60));
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.round(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.round(hours / 24)}d ago`;
  },
};

function sentimentChip(labelText, score, n) {
  const cls = score >= 0.15 ? "pos" : score <= -0.15 ? "neg" : "neu";
  const icon = cls === "pos" ? "▲" : cls === "neg" ? "▼" : "—";
  const word = cls === "pos" ? "positive" : cls === "neg" ? "negative" : "neutral";
  const chip = el("span", `s-chip ${cls}`);
  chip.append(el("span", null, icon), el("span", null, `${labelText}: ${word} (${score >= 0 ? "+" : ""}${score.toFixed(2)})`));
  if (n != null) chip.title = `${n} recent headlines, recency-weighted`;
  return chip;
}

function toast(message) {
  const t = $("#toast");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { t.hidden = true; }, 5000);
}

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function rgba(hex, alpha) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

/* ---------------- market overview ---------------- */

async function loadOverview() {
  try {
    const data = await fetchJSON("/api/market/overview");
    const tiles = $("#tiles");
    tiles.replaceChildren();
    for (const q of data.indices) {
      const tile = el("button", "tile");
      tile.setAttribute("aria-label", `${q.label}: ${fmt.price(q.price)}, ${fmt.pct(q.change_pct)} today. Analyse.`);
      const delta = el("div", `t-delta ${q.change_pct >= 0 ? "delta-up" : "delta-down"}`,
        `${q.change_pct >= 0 ? "▲" : "▼"} ${fmt.pct(q.change_pct)}`);
      tile.append(el("div", "t-label", q.label), el("div", "t-value", fmt.price(q.price)), delta);
      tile.addEventListener("click", () => analyze(q.symbol));
      tiles.append(tile);
    }
    const chips = $("#macro-chips");
    chips.replaceChildren(
      sentimentChip("Business news", data.sentiment.market.score, data.sentiment.market.n),
      sentimentChip("Politics & world", data.sentiment.politics.score, data.sentiment.politics.n),
    );
  } catch (err) {
    console.error("overview failed:", err);
  }
}

/* ---------------- signal screener ---------------- */

const CONF_ORDER = { high: 0, medium: 1, low: 2 };

async function loadScreener() {
  const grid = $("#screener-grid");
  try {
    state.screener = await fetchJSON("/api/screener");
    renderConfToggles();
    renderScreener();
    const when = new Date(state.screener.generated_at * 1000)
      .toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    $("#screener-meta").textContent =
      `${state.screener.cards.length} symbols · updated ${when} · confidence calibrated on a 10-year walk-forward test`;
  } catch (err) {
    grid.replaceChildren(el("p", "screener-status", `Screener unavailable: ${err.message}`));
  }
}

function renderConfToggles() {
  const wrap = $("#conf-toggles");
  wrap.replaceChildren();
  const cards = (state.screener && state.screener.cards) || [];
  const counts = { all: cards.length, high: 0, medium: 0, low: 0 };
  for (const c of cards) counts[c.confidence] = (counts[c.confidence] || 0) + 1;
  for (const [id, label] of [["all", "All"], ["high", "High"], ["medium", "Medium"], ["low", "Low"]]) {
    const btn = el("button", null, `${label} (${counts[id] || 0})`);
    btn.setAttribute("aria-pressed", String(state.confFilter === id));
    btn.addEventListener("click", () => {
      state.confFilter = id;
      renderConfToggles();
      renderScreener();
    });
    wrap.append(btn);
  }
}

function renderScreener() {
  const grid = $("#screener-grid");
  grid.replaceChildren();
  let cards = [...((state.screener && state.screener.cards) || [])];
  if (state.confFilter !== "all") cards = cards.filter((c) => c.confidence === state.confFilter);
  cards.sort((a, b) =>
    (CONF_ORDER[a.confidence] - CONF_ORDER[b.confidence]) ||
    (Math.abs(b.prob_up - 0.5) - Math.abs(a.prob_up - 0.5)));
  if (!cards.length) {
    grid.append(el("p", "screener-status",
      `No ${state.confFilter}-confidence signals right now — that's honesty, not a bug.`));
    return;
  }
  for (const c of cards) {
    const card = el("button", "scard");
    card.setAttribute("aria-label",
      `${c.ticker}: ${c.direction === "up" ? "likely up" : "likely down"} ` +
      `${Math.round((c.direction === "up" ? c.prob_up : 1 - c.prob_up) * 100)} percent, ` +
      `${c.confidence} confidence. Open full analysis.`);

    const top = el("div", "sc-top");
    top.append(el("span", "sc-ticker", c.ticker), el("span", `conf-tag ${c.confidence}`, c.confidence));
    card.append(top);
    card.append(el("div", "sc-name", c.name));

    const priceRow = el("div", "sc-price-row");
    priceRow.append(el("span", "sc-price", fmt.price(c.price)));
    priceRow.append(el("span", `sc-delta ${c.change_pct >= 0 ? "delta-up" : "delta-down"}`,
      `${c.change_pct >= 0 ? "▲" : "▼"} ${fmt.pct(c.change_pct)}`));
    card.append(priceRow);

    const sig = el("div", "sc-signal");
    const up = c.direction === "up";
    const dir = el("span", `dir-chip ${c.direction}`);
    dir.append(el("span", null, up ? "▲" : "▼"),
               el("span", null, `${Math.round((up ? c.prob_up : 1 - c.prob_up) * 100)}% ${up ? "up" : "down"}`));
    sig.append(dir, el("span", "meta-note", "5 sessions"));
    card.append(sig);

    if (c.plan) {
      const planText = c.plan.mode === "buy"
        ? `buy ≲ ${fmt.price(c.plan.buy_at)} · sell ${fmt.price(c.plan.sell_at)} · ~${c.plan.sessions != null ? c.plan.sessions : "?"} sess.`
        : `holders: stop ${fmt.price(c.plan.stop_at)} · sell ${fmt.price(c.plan.sell_at)}`;
      const planLine = el("div", "sc-plan", planText);
      planLine.title = c.plan.mode === "buy"
        ? `Limit-buy level, take-profit level, and the median sessions the sell level took to hit historically (reached before the stop in ${c.plan.sell_odds_pct}% of past cases). Open the card for the full plan.`
        : "Exit levels for existing holders — open the card for the full plan.";
      card.append(planLine);
    }

    card.addEventListener("click", () => analyze(c.ticker));
    grid.append(card);
  }
}

/* ---------------- live track record ---------------- */

async function loadTrackRecord() {
  let data;
  try { data = await fetchJSON("/api/track-record"); }
  catch (err) { return; }
  if (!data.n_logged) return;

  const card = $("#track-card");
  card.replaceChildren();

  const line = el("p", "track-line");
  const bits = [`${data.n_logged} predictions logged since ${fmt.date(data.since)}`];
  if (data.n_resolved) bits.push(`${data.n_resolved} resolved`);
  if (data.n_pending) bits.push(`${data.n_pending} awaiting their 5-session outcome`);
  line.textContent = bits.join(" · ");
  card.append(line);

  if (data.stats) {
    const stats = el("div", "track-stats");
    const fact = (label, value, note) => {
      const f = el("div", "fact");
      f.append(el("div", "f-label", label), el("div", "f-value", value));
      if (note) f.append(el("div", "f-note", note));
      return f;
    };
    stats.append(fact("Hit rate", `${Math.round(data.stats.hit_rate * 100)}%`,
      `vs always-up ${Math.round(data.stats.always_up * 100)}%`));
    if (data.stats.brier != null) {
      stats.append(fact("Brier score", data.stats.brier.toFixed(3),
        data.stats.brier_drift != null ? `drift alone ${data.stats.brier_drift.toFixed(3)} (lower is better)` : "lower is better"));
    }
    for (const tier of ["high", "medium", "low"]) {
      const t = data.stats.by_confidence && data.stats.by_confidence[tier];
      if (t) stats.append(fact(`${tier} conf.`, `${Math.round(t.hit_rate * 100)}%`, `${t.n} calls`));
    }
    card.append(stats);

    const details = el("details", "table-view");
    details.append(el("summary", null, `Recently resolved calls (${Math.min(data.resolved.length, 30)})`));
    const wrap = el("div", "table-wrap");
    const table = el("table");
    const head = el("tr");
    for (const h of ["Made", "Ticker", "Call", "P(up)", "Outcome", "Result"]) head.append(el("th", null, h));
    const thead = el("thead"); thead.append(head);
    const tbody = el("tbody");
    for (const r of data.resolved) {
      const tr = el("tr");
      tr.append(
        el("td", null, r.as_of),
        el("td", null, r.ticker),
        el("td", null, r.direction === "up" ? "▲ up" : "▼ down"),
        el("td", null, `${Math.round(r.p_up * 100)}%`),
        el("td", null, `${r.realized_pct > 0 ? "+" : ""}${r.realized_pct}%`),
        el("td", null, r.correct ? "✓ hit" : "✗ miss"),
      );
      tbody.append(tr);
    }
    table.append(thead, tbody);
    wrap.append(table);
    details.append(wrap);
    card.append(details);
  } else {
    card.append(el("p", "fine",
      "No outcomes yet — the first calls resolve five trading sessions after they were logged. " +
      "Keep the app in use and this becomes the most honest test there is: predictions graded on data that didn't exist when they were made."));
  }

  if (data.learning) {
    const l = data.learning;
    const adaptive = (l.source || "").startsWith("adaptive");
    card.append(el("p", "fine",
      adaptive
        ? `Adaptive model (v2.1): blend weights learned from ${l.n_used} resolved live outcomes — model tilt ${l.weights.k_ml}, technicals ${l.weights.tech}, news ${l.weights.news_company} / ${l.weights.news_market} / ${l.weights.news_politics}. It re-learns every 6 hours while the server runs, and reverts to the backtested priors automatically if a shadow test says they were better.`
        : `Adaptive learning armed (v2.1): weights hold at their backtest-validated priors until ${l.min_resolved_to_adapt} live outcomes resolve (${l.n_used} so far). While the server runs it logs the watchlist and re-learns every 6 hours — no browsing needed.`));
  }
  $("#track").hidden = false;
}

/* ---------------- search ---------------- */

function setupSearch() {
  const input = $("#search");
  const box = $("#search-results");
  let timer = null;
  let items = [];
  let active = -1;

  const close = () => { box.hidden = true; input.setAttribute("aria-expanded", "false"); active = -1; };

  const render = (results) => {
    items = results;
    box.replaceChildren();
    if (!results.length) { close(); return; }
    results.forEach((r, i) => {
      const btn = el("button");
      btn.setAttribute("role", "option");
      btn.append(el("span", "sym", r.symbol), el("span", "nm", r.name), el("span", "ex", [r.exchange, r.type].filter(Boolean).join(" · ")));
      btn.addEventListener("mousedown", (e) => e.preventDefault()); // keep focus
      btn.addEventListener("click", () => { close(); input.value = r.symbol; analyze(r.symbol); });
      box.append(btn);
      if (i === active) btn.classList.add("active");
    });
    box.hidden = false;
    input.setAttribute("aria-expanded", "true");
  };

  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 1) { close(); return; }
    timer = setTimeout(async () => {
      try { render((await fetchJSON(`/api/search?q=${encodeURIComponent(q)}`)).results); }
      catch (e) { close(); }
    }, 280);
  });

  input.addEventListener("keydown", (e) => {
    const buttons = [...box.querySelectorAll("button")];
    if (e.key === "ArrowDown" && buttons.length) {
      e.preventDefault(); active = Math.min(active + 1, buttons.length - 1);
      buttons.forEach((b, i) => b.classList.toggle("active", i === active));
    } else if (e.key === "ArrowUp" && buttons.length) {
      e.preventDefault(); active = Math.max(active - 1, 0);
      buttons.forEach((b, i) => b.classList.toggle("active", i === active));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (active >= 0 && items[active]) { input.value = items[active].symbol; close(); analyze(items[active].symbol); }
      else if (input.value.trim()) { close(); analyze(input.value.trim().toUpperCase()); }
    } else if (e.key === "Escape") close();
  });
  input.addEventListener("blur", () => setTimeout(close, 120));
}

/* ---------------- analysis ---------------- */

async function analyze(ticker) {
  const section = $("#analysis");
  const firstLoad = section.hidden;
  if (firstLoad) {
    $("#status").hidden = false;
    $("#status").textContent = `Analysing ${ticker} — fetching prices, reading the news, training the model…`;
  } else {
    section.classList.add("loading"); // hold previous render at reduced opacity
  }
  try {
    const [data, hist] = await Promise.all([
      fetchJSON(`/api/predict/${encodeURIComponent(ticker)}`),
      fetchJSON(`/api/history/${encodeURIComponent(ticker)}?days=500`),
    ]);
    state.ticker = data.quote.ticker;
    state.data = data;
    state.candles = hist.candles;
    state.newsTab = "company";

    // Unhide before rendering: the chart must be created at its real width.
    $("#empty-state").hidden = true;
    section.hidden = false;

    renderQuote(data.quote);
    renderPrediction(data);
    renderPlan(data.trade_plan);
    renderComponents(data.components);
    renderSignals(data.signals);
    renderChart();
    renderTable();
    renderNewsTabs();
    renderNews();

    document.title = `${data.quote.ticker} · StockPredict`;
  } catch (err) {
    toast(err.message || "Something went wrong.");
  } finally {
    $("#status").hidden = true;
    section.classList.remove("loading");
  }
}

function renderQuote(q) {
  const block = $("#quote-block");
  block.replaceChildren();
  block.append(el("div", "q-name", `${q.name} (${q.ticker})`));
  const meta = [q.exchange, q.sector, q.market_cap ? `Mkt cap ${fmt.compact(q.market_cap)}` : null]
    .filter(Boolean).join("  ·  ");
  block.append(el("div", "q-meta", meta || "—"));
  const row = el("div", "q-price-row");
  row.append(el("span", "q-price", fmt.price(q.price)));
  if (q.currency) row.append(el("span", "q-ccy", q.currency));
  const chip = el("span", `s-chip ${q.change_pct >= 0 ? "pos" : "neg"}`);
  chip.append(el("span", null, q.change_pct >= 0 ? "▲" : "▼"),
              el("span", null, `${fmt.price(Math.abs(q.change))} (${fmt.pct(q.change_pct)})`));
  row.append(chip, el("span", "q-ccy", `close ${fmt.date(q.as_of)}`));
  block.append(row);
}

function renderPrediction(data) {
  const p = data.prediction;
  const card = $("#prediction-card");
  card.replaceChildren();
  card.append(el("h3", null, `Prediction — next ${p.horizon_days} trading sessions`));

  const up = p.direction === "up";
  const top = el("div", "pred-top");
  const dir = el("span", `dir-chip ${p.direction}`);
  dir.append(el("span", null, up ? "▲" : "▼"), el("span", null, up ? "LIKELY UP" : "LIKELY DOWN"));
  top.append(dir, el("span", "conf-badge", `${p.confidence} confidence`));
  card.append(top);

  const probShown = up ? p.prob_up : 1 - p.prob_up;
  card.append(el("div", "pred-hero", `${Math.round(probShown * 100)}%`));
  card.append(el("div", "pred-sub",
    `estimated chance ${data.quote.ticker} closes ${up ? "higher" : "lower"} in ${p.horizon_days} sessions (P(up) = ${(p.prob_up * 100).toFixed(1)}%)`));

  // Diverging probability meter, midpoint at 50%
  const meter = el("div", "meter");
  meter.setAttribute("role", "img");
  meter.setAttribute("aria-label", `Probability of rise: ${(p.prob_up * 100).toFixed(0)} percent`);
  const track = el("div", "meter-track");
  const fill = el("div", `meter-fill ${up ? "up" : "down"}`);
  const pc = p.prob_up * 100;
  if (up) { fill.style.left = "50%"; fill.style.width = `${pc - 50}%`; }
  else { fill.style.left = `${pc}%`; fill.style.width = `${50 - pc}%`; }
  const needle = el("div", "meter-needle");
  needle.style.left = `${pc}%`;
  track.append(fill, el("div", "meter-mid"), needle);
  const scale = el("div", "meter-scale");
  scale.append(el("span", null, "0% — down"), el("span", null, "50%"), el("span", null, "100% — up"));
  meter.append(track, scale);
  card.append(meter);

  const facts = el("div", "fact-grid");
  const fact = (label, value, note) => {
    const f = el("div", "fact");
    f.append(el("div", "f-label", label), el("div", "f-value", value));
    if (note) f.append(el("div", "f-note", note));
    return f;
  };
  if (p.expected_range) {
    facts.append(fact(`Expected ${p.horizon_days}-day range (~80%)`,
      `${fmt.price(p.expected_range.low)} – ${fmt.price(p.expected_range.high)}`,
      `±${p.expected_range.pct}% from realised volatility`));
  }
  facts.append(fact("Next earnings", data.earnings_date ? fmt.date(data.earnings_date) : "—",
    data.earnings_date ? "event risk: moves cluster around earnings" : null));
  if (data.backtest) {
    facts.append(fact("Model backtest", `${Math.round(data.backtest.hit_rate * 100)}% vs ${Math.round(data.backtest.baseline * 100)}% baseline`,
      `direction hit-rate, last ${data.backtest.holdout_days} unseen sessions`));
  } else {
    facts.append(fact("Model backtest", "—", "not enough history; technicals only"));
  }
  facts.append(fact("Data as of", fmt.date(p.as_of), "daily close, Yahoo Finance"));
  card.append(facts);

  if (p.caveat) card.append(el("div", "caveat", `⚠ ${p.caveat}`));
  card.append(el("p", "fine", data.disclaimer));
}

function renderPlan(plan) {
  const card = $("#plan-card");
  if (!plan) { card.hidden = true; return; }
  card.hidden = false;
  card.replaceChildren();
  card.append(el("h3", null, "Trade planner — statistical scenario"));

  const intro = el("p", "plan-intro");
  intro.textContent = plan.mode === "buy"
    ? `Model leans up. A staged plan from the last close (${fmt.price(plan.price)}), with levels sized off this ticker's typical daily range (ATR ${fmt.price(plan.atr)}):`
    : `Model leans down — no new-buy setup here. If you already hold, exit levels sized off its typical daily range (ATR ${fmt.price(plan.atr)}):`;
  card.append(intro);

  const row = (label, value, note) => {
    const r = el("div", "plan-row");
    r.append(el("div", "plan-label", label), el("div", "plan-value", value));
    if (note) r.append(el("div", "plan-note", note));
    card.append(r);
  };

  if (plan.mode === "buy") {
    row("1 · Buy", `${fmt.price(plan.entry_market)} at market — or a limit order at ${fmt.price(plan.entry_pullback)}`,
      plan.entry_fill_pct != null
        ? `waiting for the dip got a fill within 5 sessions in ${plan.entry_fill_pct}% of past cases (better price, but you can miss the move)`
        : null);
    row("2 · Sell order (take-profit)", fmt.price(plan.target),
      `reached before the stop in ${plan.target_first_pct}% of past cases · typically ~${plan.median_sessions_to_target} sessions (≈${plan.median_calendar_days} calendar days) when it gets there · hit within 10 sessions ${plan.target_within_10_pct}% of the time`);
    row("3 · Stop-loss (protection)", fmt.price(plan.stop),
      `hit first in ${plan.stop_first_pct}% of past cases · caps the loss at ≈${plan.risk_pct}%`);
    if (plan.stretch_target != null) {
      row("Stretch sell (range top)", fmt.price(plan.stretch_target),
        "top of the model's 5-session ~80% range — a level for a second, smaller sell order");
    }
  } else {
    row("Sell into strength (limit)", fmt.price(plan.target),
      `touched before the stop in ${plan.target_first_pct}% of past cases · typically ~${plan.median_sessions_to_target} sessions (≈${plan.median_calendar_days} calendar days)`);
    row("Protective stop", fmt.price(plan.stop),
      `hit first in ${plan.stop_first_pct}% of past cases · caps further downside at ≈${plan.risk_pct}%`);
  }

  const expectancy = (plan.target_first_pct * plan.reward_pct - plan.stop_first_pct * plan.risk_pct) / 100;
  card.append(el("p", "fine",
    `Reward:risk ${plan.risk_reward}:1 (+${plan.reward_pct}% vs −${plan.risk_pct}%); historical expectancy of this exact bracket ≈ ${expectancy >= 0 ? "+" : ""}${expectancy.toFixed(2)}% per trade before costs — near zero is normal: levels manage risk, they don't create edge. ` +
    `In ${plan.neither_pct}% of past cases neither level was reached within 20 sessions. Measured on ${plan.sample_n} overlapping windows of this ticker's last 2 years. ${plan.note}`));
}

function renderComponents(c) {
  const card = $("#components-card");
  card.replaceChildren();
  card.append(el("h3", null, "What's driving it"));

  const rows = [
    ["Drift anchor", "historical 5-session up-rate", (c.drift.base - 0.5) * 2,
      `${(c.drift.base * 100).toFixed(0)}%`],
    ["Model tilt", "model P(up) vs anchor", c.ml_model ? c.ml_model.tilt * 2 : null,
      c.ml_model ? `${(c.ml_model.prob_up * 100).toFixed(0)}%` : "n/a"],
    ["Technical read", null, c.technical.score, c.technical.score.toFixed(2)],
    ["Company news", `${c.news_company.n} headlines`, c.news_company.score, scoreText(c.news_company)],
    ["Market news", `${c.news_market.n} headlines`, c.news_market.score, scoreText(c.news_market)],
    ["Politics & world", `${c.news_politics.n} headlines`, c.news_politics.score, scoreText(c.news_politics)],
  ];

  for (const [name, sub, score, valueText] of rows) {
    const row = el("div", "comp-row");
    const nameEl = el("div", "comp-name", name);
    if (sub) nameEl.append(el("small", null, sub));
    const bar = el("div", "comp-bar");
    bar.append(el("div", "mid"));
    if (score != null && Math.abs(score) > 0.005) {
      const fillNode = el("div", `fill ${score >= 0 ? "pos" : "neg"}`);
      const width = Math.min(Math.abs(score), 1) * 50;
      if (score >= 0) { fillNode.style.left = "50%"; fillNode.style.width = `${width}%`; }
      else { fillNode.style.left = `${50 - width}%`; fillNode.style.width = `${width}%`; }
      bar.append(fillNode);
    }
    bar.title = "bar shows pull on the prediction: left = bearish, right = bullish";
    row.append(nameEl, bar, el("div", "comp-val", valueText));
    card.append(row);
  }
  const w = state.data.components.weights;
  const learning = w.learning || {};
  const adaptive = (learning.source || "").startsWith("adaptive");
  card.append(el("p", "fine",
    `Weights around the drift anchor — model tilt ${w.k_ml}, technical ${w.tech}, news: company ${w.news_company}, market ${w.news_market}, politics ${w.news_politics}. ` +
    (adaptive
      ? `Learned online from ${learning.n_used} resolved live outcomes (anchored to backtested priors).`
      : `Backtest-validated priors; they adapt automatically once ${learning.min_resolved_to_adapt || 40} live outcomes resolve (${learning.n_used || 0} so far).`)));
}

function scoreText(agg) {
  if (!agg.n) return "no data";
  return `${agg.score >= 0 ? "+" : ""}${agg.score.toFixed(2)}`;
}

function renderSignals(signals) {
  const card = $("#signals-card");
  card.replaceChildren();
  card.append(el("h3", null, "Technical signals"));
  for (const s of signals) {
    const row = el("div", "signal-row");
    const left = el("div");
    left.append(el("span", "signal-name", s.name), el("span", "signal-text", ` — ${s.text}`));
    const cls = s.score >= 0.15 ? "pos" : s.score <= -0.15 ? "neg" : "neu";
    const chip = el("span", `s-chip ${cls}`);
    chip.append(el("span", null, cls === "pos" ? "▲" : cls === "neg" ? "▼" : "—"),
                el("span", null, `${s.score >= 0 ? "+" : ""}${s.score.toFixed(2)}`));
    row.append(left, chip);
    card.append(row);
  }
}

/* ---------------- chart ---------------- */

function chartTheme() {
  return {
    surface: cssVar("--surface"),
    text: cssVar("--muted"),
    grid: cssVar("--grid"),
    baseline: cssVar("--baseline"),
    up: cssVar("--up"),
    down: cssVar("--down"),
    sma20: cssVar("--series-1"),
    sma50: cssVar("--series-2"),
    ink: cssVar("--ink"),
  };
}

function renderChart() {
  const host = $("#chart");
  if (state.chart) { state.chart.remove(); state.chart = null; state.series = null; }
  host.replaceChildren();

  if (typeof LightweightCharts === "undefined") {
    host.append(el("div", "chart-fallback",
      "Chart library unavailable (offline CDN) — the data table below has every session."));
    renderLegend(null);
    return;
  }

  const t = chartTheme();
  const chart = LightweightCharts.createChart(host, {
    autoSize: true,
    layout: {
      background: { type: "solid", color: "transparent" },
      textColor: t.text,
      fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
      fontSize: 11,
    },
    grid: {
      vertLines: { color: "transparent" },
      horzLines: { color: t.grid },
    },
    rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.06, bottom: 0.26 } },
    timeScale: { borderColor: t.baseline, rightOffset: 3 },
    crosshair: {
      horzLine: { labelBackgroundColor: t.ink },
      vertLine: { labelBackgroundColor: t.ink },
    },
  });

  // Widen autoscale so the expected-range projection lines stay on screen.
  const expected = state.data && state.data.prediction.expected_range;
  const candles = chart.addCandlestickSeries({
    upColor: t.up, downColor: t.down, borderVisible: false,
    wickUpColor: t.up, wickDownColor: t.down,
    autoscaleInfoProvider: (original) => {
      const info = original();
      if (info && info.priceRange && expected) {
        info.priceRange.minValue = Math.min(info.priceRange.minValue, expected.low);
        info.priceRange.maxValue = Math.max(info.priceRange.maxValue, expected.high);
      }
      return info;
    },
  });
  const volume = chart.addHistogramSeries({ priceScaleId: "", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
  chart.priceScale("").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
  const lineOpts = { lineWidth: 2, priceLineVisible: false, lastValueVisible: false };
  const sma20 = chart.addLineSeries({ ...lineOpts, color: t.sma20 });
  const sma50 = chart.addLineSeries({ ...lineOpts, color: t.sma50 });
  const bandOpts = { lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, color: rgba(t.text, 0.55), crosshairMarkerVisible: false };
  const bbUp = chart.addLineSeries(bandOpts);
  const bbLo = chart.addLineSeries(bandOpts);

  state.chart = chart;
  state.series = { candles, volume, sma20, sma50, bbUp, bbLo };
  setChartRange(state.rangeDays);

  // Expected-range projection lines (dashed = projection, not history)
  if (expected) {
    candles.createPriceLine({ price: expected.high, color: rgba(t.up, 0.8), lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "exp. high 5d" });
    candles.createPriceLine({ price: expected.low, color: rgba(t.down, 0.8), lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "exp. low 5d" });
  }

  // Crosshair readout: one readout, every series (also shown for latest bar by default)
  chart.subscribeCrosshairMove((param) => {
    if (param && param.time && param.seriesData && param.seriesData.get(candles)) {
      const c = param.seriesData.get(candles);
      const extras = [];
      const s20 = param.seriesData.get(sma20); if (s20) extras.push(`SMA20 ${fmt.price(s20.value)}`);
      const s50 = param.seriesData.get(sma50); if (s50) extras.push(`SMA50 ${fmt.price(s50.value)}`);
      const v = param.seriesData.get(volume); if (v) extras.push(`Vol ${fmt.compact(v.value)}`);
      setReadout(String(param.time), c.open, c.high, c.low, c.close, extras.join("  "));
    } else {
      defaultReadout();
    }
  });
  defaultReadout();
  renderLegend(t);
}

function setChartRange(days) {
  state.rangeDays = days;
  if (!state.series) return;
  const slice = state.candles.slice(-days);
  const t = chartTheme();
  state.series.candles.setData(slice.map(d => ({ time: d.t, open: d.o, high: d.h, low: d.l, close: d.c })));
  state.series.volume.setData(slice.map(d => ({
    time: d.t, value: d.v || 0,
    color: rgba((d.c >= d.o ? t.up : t.down), 0.34),
  })));
  const line = (key) => slice.filter(d => d[key] != null).map(d => ({ time: d.t, value: d[key] }));
  state.series.sma20.setData(line("sma20"));
  state.series.sma50.setData(line("sma50"));
  state.series.bbUp.setData(line("bb_up"));
  state.series.bbLo.setData(line("bb_lo"));
  state.chart.timeScale().fitContent();
  for (const btn of document.querySelectorAll(".range-row button")) {
    btn.setAttribute("aria-pressed", String(Number(btn.dataset.days) === days));
  }
}

function setReadout(dateStr, o, h, l, c, extra) {
  $("#chart-readout").textContent =
    `${fmt.date(dateStr)}   O ${fmt.price(o)}  H ${fmt.price(h)}  L ${fmt.price(l)}  C ${fmt.price(c)}   ${extra || ""}`;
}

function defaultReadout() {
  const last = state.candles[state.candles.length - 1];
  if (!last) return;
  const extras = [];
  if (last.sma20 != null) extras.push(`SMA20 ${fmt.price(last.sma20)}`);
  if (last.sma50 != null) extras.push(`SMA50 ${fmt.price(last.sma50)}`);
  if (last.v != null) extras.push(`Vol ${fmt.compact(last.v)}`);
  setReadout(last.t, last.o, last.h, last.l, last.c, extras.join("  "));
}

function renderLegend(theme) {
  const legend = $("#chart-legend");
  legend.replaceChildren();
  const key = (label, colorVar, dashed, swatch) => {
    const k = el("span", "lkey");
    if (swatch) {
      const sw = el("i", "swatch"); sw.style.background = colorVar; sw.style.borderTop = "none"; k.append(sw);
    } else {
      const ln = el("i", dashed ? "dashed" : null); ln.style.borderTopColor = colorVar; k.append(ln);
    }
    k.append(el("span", null, label));
    return k;
  };
  legend.append(
    key("Up day", "var(--up)", false, true),
    key("Down day", "var(--down)", false, true),
    key("SMA 20", "var(--series-1)"),
    key("SMA 50", "var(--series-2)"),
    key("Bollinger 20 ± 2σ", "var(--muted)", true),
  );
}

function renderRangeButtons() {
  const row = $("#range-row");
  row.replaceChildren();
  for (const [label, days] of RANGES) {
    const btn = el("button", null, label);
    btn.dataset.days = days;
    btn.setAttribute("aria-pressed", String(days === state.rangeDays));
    btn.addEventListener("click", () => setChartRange(days));
    row.append(btn);
  }
}

function renderTable() {
  const table = $("#sessions-table");
  table.replaceChildren();
  const head = el("tr");
  for (const h of ["Date", "Open", "High", "Low", "Close", "Volume", "RSI 14"]) head.append(el("th", null, h));
  const thead = el("thead"); thead.append(head);
  const tbody = el("tbody");
  for (const d of state.candles.slice(-30).reverse()) {
    const tr = el("tr");
    tr.append(
      el("td", null, d.t), el("td", null, fmt.price(d.o)), el("td", null, fmt.price(d.h)),
      el("td", null, fmt.price(d.l)), el("td", null, fmt.price(d.c)),
      el("td", null, fmt.compact(d.v)), el("td", null, d.rsi != null ? d.rsi.toFixed(0) : "—"),
    );
    tbody.append(tr);
  }
  table.append(thead, tbody);
}

/* ---------------- news ---------------- */

const NEWS_TABS = [["company", "Company"], ["market", "Market"], ["politics", "Politics & world"]];

function renderNewsTabs() {
  const tabs = $("#news-tabs");
  tabs.replaceChildren();
  for (const [id, label] of NEWS_TABS) {
    const btn = el("button", null, label);
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", String(id === state.newsTab));
    btn.addEventListener("click", () => { state.newsTab = id; renderNewsTabs(); renderNews(); });
    tabs.append(btn);
  }
}

function renderNews() {
  const list = $("#news-list");
  list.replaceChildren();
  const articles = (state.data && state.data.news[state.newsTab]) || [];
  if (!articles.length) {
    list.append(el("li", "news-empty", "No recent headlines matched. Sentiment for this stream is treated as neutral."));
    return;
  }
  for (const a of articles) {
    const li = el("li");
    const main = el("div", "news-main");
    let title;
    if (a.link && /^https?:\/\//i.test(a.link)) {
      title = el("a", "news-title", a.title);
      title.href = a.link; title.target = "_blank"; title.rel = "noopener noreferrer";
    } else {
      title = el("span", "news-title", a.title);
    }
    main.append(title);
    main.append(el("div", "news-meta", [a.source, fmt.ago(a.published)].filter(Boolean).join("  ·  ")));
    const cls = a.sentiment >= 0.15 ? "pos" : a.sentiment <= -0.15 ? "neg" : "neu";
    const chip = el("span", `s-chip ${cls}`);
    chip.title = "headline sentiment";
    chip.append(el("span", null, cls === "pos" ? "▲" : cls === "neg" ? "▼" : "—"),
                el("span", null, `${a.sentiment >= 0 ? "+" : ""}${a.sentiment.toFixed(2)}`));
    li.append(main, chip);
    list.append(li);
  }
}

/* ---------------- boot ---------------- */

function init() {
  setupSearch();
  renderRangeButtons();

  const picks = $("#quick-picks");
  for (const symbol of QUICK_PICKS) {
    const chip = el("button", "chip", symbol);
    chip.addEventListener("click", () => analyze(symbol));
    picks.append(chip);
  }

  // Rebuild the chart when the OS theme flips — dark mode is selected, not flipped.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (state.candles.length) { renderChart(); }
    if (state.data) { renderComponents(state.data.components); }
  });

  loadOverview();
  loadScreener().then(loadTrackRecord); // screener logs predictions; then show the ledger
  analyze("AAPL"); // sensible default so the page never starts empty
}

document.addEventListener("DOMContentLoaded", init);
