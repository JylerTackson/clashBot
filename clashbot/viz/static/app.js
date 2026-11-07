/* Decision dashboard (MAC-AO). Plain JS, no libraries, no network beyond this server. Polls every second. */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const POLL_MS = 1000;
const RUNS_REFRESH_MS = 10000;
const HOLD = "hold";
const MODE_COLORS = {
  defend: "#4aa3ff", leak: "#ff6b6b", punish: "#da77f2", counter: "#ffa94d", learned: "#69db7c",
  trigger: "#ced4da", hold: "#868e96", save: "#adb5bd", none: "#868e96",
};
const BUCKET_COLORS = ["#4dabf7", "#69db7c", "#ffa94d", "#da77f2", "#ff6b6b", "#ffd43b", "#38d9a9", "#f783ac", "#a9e34b", "#74c0fc"];
const REL_STYLE = {
  COUNTERS: { color: "#ff6b6b", dash: "", arrow: true, label: "COUNTERS (wiki)" },
  ANSWERED_BY: { color: "#ffa94d", dash: "", arrow: true, label: "ANSWERED_BY (pro games)" },
  SYNERGY: { color: "#69db7c", dash: "7 5", arrow: false, label: "SYNERGY" },
};
const GOLD = "#ffd43b";

const S = {
  gen: 0, current: null, pinned: null, runId: null, meta: null, stats: null, live: false,
  decisions: [], holds: [], buckets: {}, sel: null, auto: null, shot: "a",
  detail: null, stateCache: {}, kgCache: {}, kgKey: null, runs: [], runsAt: 0, runsKey: "", view: {},
};

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

// ------------------------------------------------------------------------------------------------ polling
function resetRun(id) {
  S.gen += 1;
  Object.assign(S, { runId: id, meta: null, stats: null, decisions: [], holds: [], buckets: {}, sel: null, auto: null,
                     shot: "a", detail: null, kgKey: null, view: {} });
}

async function loadRuns() {
  S.runsAt = Date.now();
  try {
    S.runs = (await getJSON("/api/runs")).runs;
  } catch (e) { /* keep the old list */ }
}

function applyData(meta, dec, hold) {
  S.meta = meta;
  const restarted = dec.total < S.decisions.length || hold.total < S.holds.length ||
    (dec.rows.length && dec.rows[0].i !== S.decisions.length) || (hold.rows.length && hold.rows[0].i !== S.holds.length);
  if (restarted) {                      // the files were rewritten: read the run again from the start next tick
    Object.assign(S, { decisions: [], holds: [], sel: null, view: {}, kgKey: null });
    return;
  }
  for (const r of dec.rows) S.decisions.push(r);
  for (const r of hold.rows) S.holds.push(r);
  S.buckets = hold.buckets;
  S.stats = dec.stats;
  S.live = dec.live;
  if (S.auto === null && (S.decisions.length || meta.live !== undefined)) {
    S.auto = !!meta.live;               // live: follow the newest decision; replay: start on the first tap
    const asked = parseInt(new URLSearchParams(location.search).get("d"), 10);     // ?d=12 opens decision 12
    if (asked >= 0 && asked < S.decisions.length) {
      S.auto = false;
      S.sel = asked;
    } else if (!S.auto && S.decisions.length) {
      const first = S.decisions.findIndex((r) => (r.executor || {}).sent);
      S.sel = first >= 0 ? first : 0;
    }
  }
  if (S.auto && S.decisions.length) S.sel = S.decisions.length - 1;
  if (S.sel === null && S.decisions.length) S.sel = 0;
}

async function tick() {
  try {
    const cur = await getJSON("/api/current");
    S.current = cur;
    const want = S.pinned || cur.id;
    if (want !== S.runId) resetRun(want);
    if (S.runId) {
      const gen = S.gen;
      const id = encodeURIComponent(S.runId);
      const [meta, dec, hold] = await Promise.all([
        getJSON(`/api/run/${id}/meta`),
        getJSON(`/api/run/${id}/decisions?since=${S.decisions.length}`),
        getJSON(`/api/run/${id}/holds?since=${S.holds.length}`),
      ]);
      if (gen !== S.gen) return;
      applyData(meta, dec, hold);
    }
    if (Date.now() - S.runsAt > RUNS_REFRESH_MS) await loadRuns();
    renderAll();
  } catch (e) {
    $("badge").textContent = "OFFLINE";
    $("badge").className = "badge";
    $("run-name").textContent = `dashboard server not answering (${e.message})`;
  } finally {
    setTimeout(tick, POLL_MS);
  }
}

// ------------------------------------------------------------------------------------------------ helpers
const fmt = (x, d = 2) => (typeof x === "number" && isFinite(x) ? x.toFixed(d) : "-");
const ms = (x) => (typeof x === "number" ? `${Math.round(x)} ms` : "-");
const fmtT = (t) => (t >= 120 ? `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}` : `${Math.round(t)}s`);
const pretty = (key) => String(key ?? "").replace(/\|/g, " · ");
const uniq = (xs) => [...new Set(xs.filter(Boolean))];
const cardOf = (key) => {
  if (!key || key === HOLD) return null;
  return key.startsWith("ability:") ? key.slice(8).split("|")[0] : key.split("|")[0];
};
const gameClock = (row) => {
  const m = /^\[[^|\]]*\|([0-9?]+:[0-9?]+)\]/.exec(row.context_text || "");
  return m ? m[1] : null;
};
const modeColor = (m) => MODE_COLORS[m] || "#f1f3f5";
const execClass = (row) => {
  const ex = row.executor || {};
  const status = ex.status || "dry_run";
  if (ex.sent) return "sent";
  if (status === "gated") return "gated";
  if (status.startsWith("hold") || row.key === HOLD) return "hold";
  if (status === "dry_run") return "dry_run";
  return "other";
};
const cardName = (slug) => {
  const d = ((S.meta || {}).deck || []).find((c) => c.slug === slug);
  return d ? d.name : String(slug).replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
};

// ------------------------------------------------------------------------------------------------ header
function renderHeader() {
  const m = S.meta;
  const badge = $("badge");
  if (!m) {
    badge.textContent = "WAITING";
    badge.className = "badge";
    const parent = S.current && S.current.parent;
    $("run-name").textContent = parent ? `waiting for a run under ${parent}` : "no run selected";
    $("stats").innerHTML = "";
    $("deck").innerHTML = "";
    $("models").innerHTML = "";
    buildRunSelect();
    return;
  }
  badge.textContent = m.live ? "LIVE" : "REPLAY";
  badge.className = `badge ${m.live ? "live" : "replay"}`;
  $("run-name").textContent = m.name;
  const st = S.stats || m.stats || {};
  const battles = m.battles || [];
  const last = battles[battles.length - 1];
  const row = S.decisions[S.sel];
  let battle = "-";
  if (last) {
    battle = `#${esc(last.n ?? battles.length)} ${esc(last.banner || last.opponent || "")} ${esc(last.verdict || "")}`;
  } else if (row && row.battle !== undefined && row.battle !== null) {
    battle = `#${esc(row.battle)}`;
  }
  $("stats").innerHTML =
    `decisions <b>${st.decisions ?? 0}</b> &middot; taps <b>${st.taps ?? 0}</b> &middot; ` +
    `latency p50 <b>${ms(st.p50_ms)}</b> p95 <b>${ms(st.p95_ms)}</b> &middot; battle <b>${battle}</b>`;
  $("deck").innerHTML = (m.deck || []).map((c) =>
    `<span class="chip card ${c.evo ? "evo" : ""} ${c.hero ? "hero" : ""}">${esc(c.name)}` +
    `${c.evo ? " (EVO)" : ""}${c.hero ? " (HERO)" : ""}</span>`).join("") ||
    '<span class="dim">deck not recorded in config.json</span>';
  const c = m.config || {};
  const model = (p) => esc(String(p || "").split("/").pop());
  const bits = [
    `ranker <b>${esc(c.ranker || "-")}</b>${c.d1_model ? ` (${model(c.d1_model)})` : ""}`,
    `policy <b>${esc(c.policy || "-")}</b>${c.act_model ? ` + ${model(c.act_model)}` : ""}`,
    `tactics ${c.no_tactics ? "off" : "on"}`,
    `KG v2 ${c.no_kg2 ? "off" : "on"}`,
    c.arbiter && c.arbiter !== "none" ? `arbiter ${esc(c.arbiter)}` : "",
    `source ${esc(c.source || "-")}`,
    c.live ? "taps: LIVE flag" : "taps: dry run / watch",
  ].filter(Boolean);
  $("models").innerHTML = bits.join(" &middot; ");
  buildRunSelect();
  $("follow-toggle").checked = !!S.auto;
}

function buildRunSelect() {
  const sel = $("run-select");
  const key = JSON.stringify([S.runs.map((r) => r.id), S.pinned, S.runId, S.current && S.current.follow]);
  if (S.runsKey === key) return;
  S.runsKey = key;
  const auto = S.current && S.current.follow ? `auto: newest run under ${S.current.parent}` : "the run this server was started with";
  const opts = [`<option value="">${esc(auto)}</option>`];
  for (const r of S.runs) {
    const tail = r.has_retrieval ? "" : " (no retrieval log)";
    opts.push(`<option value="${esc(r.id)}">${esc(r.parent)}/${esc(r.name)} - ${r.decisions} decisions${tail}</option>`);
  }
  sel.innerHTML = opts.join("");
  sel.value = S.pinned || "";
}

// ------------------------------------------------------------------------------------------------ timeline
function holdRuns() {
  // consecutive hold rows with the same reason bucket become one band segment
  const runs = [];
  const H = S.holds;
  for (let i = 0; i < H.length; i++) {
    const end = i + 1 < H.length ? Math.min(H[i + 1].t, H[i].t + 4) : H[i].t + 1;
    const last = runs[runs.length - 1];
    if (last && last.bucket === H[i].bucket && H[i].t - last.t1 < 4.5) last.t1 = end;
    else runs.push({ t0: H[i].t, t1: end, bucket: H[i].bucket, act: H[i].act });
  }
  return runs;
}

function renderTimeline() {
  const el = $("timeline");
  const W = Math.max(el.clientWidth, 640);
  const sig = [S.runId, S.decisions.length, S.holds.length, S.sel, W].join("|");
  if (S.view.timeline === sig) return;
  S.view.timeline = sig;
  const D = S.decisions;
  if (!D.length && !S.holds.length) {
    el.innerHTML = '<div class="empty">no decisions or holds logged yet</div>';
    $("timeline-legend").innerHTML = "";
    return;
  }
  let t0 = Infinity, t1 = -Infinity;
  for (const r of D) { t0 = Math.min(t0, r.t); t1 = Math.max(t1, r.t); }
  for (const r of S.holds) { t0 = Math.min(t0, r.t); t1 = Math.max(t1, r.t + 1); }
  if (!(t1 > t0)) t1 = t0 + 1;
  const padL = 14, padR = 14, H = 130;
  const x = (t) => padL + ((t - t0) / (t1 - t0)) * (W - padL - padR);
  const names = Object.keys(S.buckets);
  const color = {};
  names.forEach((b, i) => { color[b] = i < BUCKET_COLORS.length ? BUCKET_COLORS[i] : "#5c6b7f"; });
  const parts = [];
  const span = t1 - t0;
  const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((s) => span / s <= 14) || 600;
  for (let tt = Math.ceil(t0 / step) * step; tt <= t1; tt += step) {
    parts.push(`<line x1="${x(tt)}" y1="20" x2="${x(tt)}" y2="${H - 22}" stroke="#223146"/>`);
    parts.push(`<text x="${x(tt)}" y="${H - 6}" text-anchor="middle">${fmtT(tt)}</text>`);
  }
  for (const b of (S.meta && S.meta.battles) || []) {          // battle bands, when the launcher logged run-time bounds
    if (typeof b.t_start === "number" && typeof b.t_end === "number" && b.t_end > t0 && b.t_start < t1) {
      const a = x(Math.max(b.t_start, t0)), z = x(Math.min(b.t_end, t1));
      parts.push(`<rect x="${a}" y="12" width="${Math.max(z - a, 1)}" height="${H - 34}" fill="#ffffff" opacity="0.05"/>`);
      parts.push(`<text x="${a + 3}" y="12">battle ${esc(b.n ?? "")}</text>`);
    }
  }
  for (const r of holdRuns()) {
    const w = Math.max(x(r.t1) - x(r.t0), 1.5);
    parts.push(`<rect x="${x(r.t0)}" y="76" width="${w}" height="26" fill="${color[r.bucket] || "#5c6b7f"}"><title>${esc(r.bucket)}</title></rect>`);
  }
  parts.push(`<text x="${padL}" y="70">why it waited</text>`);
  parts.push(`<text x="${padL}" y="30">decisions</text>`);
  D.forEach((r, i) => {
    const hollow = r.key === HOLD;
    const c = modeColor(r.mode);
    const tip = `${fmtT(r.t)} ${r.mode}: ${pretty(r.key)} - ${r.reason}`;
    parts.push(`<circle data-i="${i}" cx="${x(r.t)}" cy="46" r="7" fill="${hollow ? "none" : c}" stroke="${c}" stroke-width="2.5"><title>${esc(tip)}</title></circle>`);
  });
  if (S.sel !== null && D[S.sel]) {
    const sx = x(D[S.sel].t);
    parts.push(`<line x1="${sx}" y1="30" x2="${sx}" y2="104" stroke="#fff" stroke-width="1.5"/>`);
    parts.push(`<circle cx="${sx}" cy="46" r="12" fill="none" stroke="#fff" stroke-width="2.5" pointer-events="none"/>`);
  }
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${parts.join("")}</svg>`;
  const modes = {};
  D.forEach((r) => { modes[r.mode] = (modes[r.mode] || 0) + 1; });
  const modeHtml = Object.entries(modes).sort((a, b) => b[1] - a[1]).map(([m, n]) =>
    `<span class="dot" style="background:${modeColor(m)}"></span>${esc(m)} ${n}`).join(" ");
  const holdHtml = names.slice(0, 8).map((b) =>
    `<span class="sw" style="border-color:${color[b]}"></span>${esc(b)} <span class="dim">${S.buckets[b]}</span>`).join(" ");
  $("timeline-legend").innerHTML = `<div>${modeHtml}</div><div>${holdHtml}${names.length > 8 ? ` <span class="dim">+${names.length - 8} more</span>` : ""}</div>`;
}

function select(i, byUser) {
  if (i < 0 || i >= S.decisions.length) return;
  S.sel = i;
  if (byUser) S.auto = false;
  S.view = {};
  renderAll();
}

// ------------------------------------------------------------------------------------------------ screen
function renderScreen(row) {
  const el = $("screen");
  const sig = `${S.runId}|${row ? row.i : "-"}|${S.shot}`;
  if (S.view.screen === sig) return;
  S.view.screen = sig;
  if (!row) { el.innerHTML = '<div class="placeholder">no decision selected</div>'; $("screen-sub").textContent = ""; return; }
  const files = row.shots || {};
  const tags = ["a", "b", "c"].filter((t) => files[t]);
  $("screen-sub").textContent = `d${String(row.idx).padStart(4, "0")} t=${fmt(row.t, 1)}s`;
  if (!tags.length) {
    const why = S.meta && S.meta.has_shots ? "no screenshot for this decision" : "no screenshots in this run (needs --record-shots)";
    el.innerHTML = `<div class="placeholder">${why}</div>`;
    return;
  }
  const which = files[S.shot] ? S.shot : tags[0];
  const url = (f) => `/api/shot/${encodeURIComponent(S.runId)}/${encodeURIComponent(f)}`;
  const label = { a: "decision frame, tap drawn", b: "+0.5 s", c: "+1.5 s" };
  el.innerHTML = `<img src="${url(files[which])}" alt="${esc(label[which])}">` +
    `<div class="thumbs">${tags.map((t) => `<img data-shot="${t}" class="${t === which ? "on" : ""}" src="${url(files[t])}" title="${esc(label[t])}">`).join("")}</div>` +
    `<div class="small">${esc(label[which])}</div>`;
}

// ------------------------------------------------------------------------------------------------ decision
function renderDecision(row) {
  const el = $("decision");
  if (!row) { el.innerHTML = '<div class="empty">no decision yet</div>'; $("decision-sub").textContent = ""; return; }
  const ex = row.executor || {};
  const cls = execClass(row);
  const clock = gameClock(row);
  const cands = row.candidates || [];
  const probs = row.probs || [];
  const haveP = probs.length === cands.length && cands.length > 0;
  const hand = (row.hand || []).map((c) => (c ? `<span class="chip">${esc(cardName(c))}</span>` : '<span class="chip dim">?</span>')).join("");
  const tile = Array.isArray(row.tile) ? `[${fmt(row.tile[0], 1)}, ${fmt(row.tile[1], 1)}]` : "-";
  const kv = [
    ["run time", `${fmt(row.t, 1)} s` + (clock ? ` &nbsp; game clock ${esc(clock)}` : "")],
    ["elixir", esc(row.elixir ?? "?")],
    ["hand", hand],
    ["tile", `${tile} <span class="dim">(${esc(row.tile_source || "none")})</span>`],
  ];
  if (row.battle !== undefined && row.battle !== null) kv.push(["battle", esc(row.battle)]);
  if (row.arbiter_reason) kv.push(["arbiter", esc(row.arbiter_reason)]);
  if (row.plan_t !== undefined && row.plan_t !== null) kv.push(["coach plan", `from t=${fmt(row.plan_t, 1)} s (age ${fmt(row.plan_age_s, 1)} s)`]);
  $("decision-sub").textContent = `#${row.i + 1} of ${S.decisions.length}`;
  const candHtml = cands.map((c, i) => {
    const chosen = c.key === row.key;
    const p = haveP ? probs[i] : null;
    const chips = Object.entries(c.sources || {}).map(([k, v]) =>
      `<span class="chip ${esc(k)}">${esc(k)}${k === "fallback" ? "" : " " + fmt(Number(v), 2)}</span>`).join("");
    const tl = Array.isArray(c.tile) ? ` @[${fmt(c.tile[0], 1)}, ${fmt(c.tile[1], 1)}]` : "";
    return `<div class="cand ${chosen ? "chosen-c" : ""}">` +
      `<div class="top"><span>${chosen ? "&#9654; " : ""}${esc(pretty(c.key))}<span class="dim">${tl}</span></span>` +
      `<span>${p === null ? "" : (p * 100).toFixed(0) + "%"}</span></div>` +
      (p === null ? "" : `<div class="bar"><i style="width:${Math.max(p * 100, 1)}%"></i></div>`) +
      `<div>${chips}</div>${c.note ? `<div class="note">${esc(c.note)}</div>` : ""}</div>`;
  }).join("");
  const held = row.key === HOLD;
  el.innerHTML =
    `<div class="reason"><span class="chip" style="border-color:${modeColor(row.mode)};color:${modeColor(row.mode)}">${esc(row.mode)}</span> ${esc(row.reason)}</div>` +
    `<div class="chosen">${held ? "HOLD (no tap)" : esc(pretty(row.key))}<span class="status ${cls}">${esc(ex.status || "dry_run")}${ex.sent ? " / sent" : ""}${ex.gate ? " / " + esc(ex.gate) : ""}</span></div>` +
    `<div class="kv">${kv.map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join("")}</div>` +
    (cands.length ? `<h2>Candidates <span class="sub">${haveP ? "bar = d1 ranker probability" : "no ranker probabilities logged for this decision"}</span></h2>${candHtml}`
                  : '<div class="empty">no candidates survived the policy</div>');
}

// ------------------------------------------------------------------------------------------------ retrieval
function renderRetrieval(row) {
  const el = $("retrieval");
  const box = $("state-detail");
  if (!row) { el.innerHTML = '<div class="empty">no decision selected</div>'; box.innerHTML = ""; return; }
  if (row.retrieval === undefined) {
    el.innerHTML = '<div class="empty">not logged in this run (logged since MAC-AO)</div>';
    box.innerHTML = "";
    return;
  }
  const hits = row.retrieval;
  if (!hits.length) {
    el.innerHTML = '<div class="empty">no retrieval hits were logged for this decision</div>';
    box.innerHTML = "";
    return;
  }
  const candKeys = new Set((row.candidates || []).map((c) => c.key));
  const votes = {};
  hits.forEach((h) => { if (h.action_key && h.action_key !== HOLD) votes[h.action_key] = (votes[h.action_key] || 0) + h.score; });
  const top = Object.entries(votes).sort((a, b) => b[1] - a[1]).slice(0, 3);
  const rows = hits.map((h, k) => {
    const r = h.ranks || {};
    const v = h.verdict || "";
    return `<tr class="hit ${candKeys.has(h.action_key) ? "cand" : ""} ${S.detail && S.detail.id === h.id ? "on" : ""}" data-id="${esc(h.id)}">` +
      `<td>${k + 1}</td><td>${fmt(h.score, 4)}</td><td>${r.knn ?? "-"}</td><td>${r.bm25 ?? "-"}</td>` +
      `<td class="mono" style="font-size:14px">${esc(String(h.game_id || "").slice(0, 16))}</td>` +
      `<td>${esc(pretty(h.action_key))}</td><td class="verdict-${esc(v)}">${esc(v || "-")}</td>` +
      `<td>${typeof h.elixir_trade === "number" ? (h.elixir_trade > 0 ? "+" : "") + h.elixir_trade.toFixed(1) : "-"}</td></tr>`;
  }).join("");
  el.innerHTML =
    `<div class="small">pros' plays weighted by fused score: ${top.map(([k, s]) => `<b>${esc(pretty(k))}</b> ${fmt(s, 3)}`).join(" &middot; ") || "none"}</div>` +
    `<table><tr><th>#</th><th>fused</th><th>kNN#</th><th>BM25#</th><th>game</th><th>pro's play</th><th>verdict</th><th>trade</th></tr>${rows}</table>` +
    '<div class="small">gold bar = the pros\' play is one of the candidates; click a row for the pro\'s state</div>';
  renderStateDetail();
}

async function showState(id) {
  S.detail = { id, loading: true };
  renderStateDetail();
  document.querySelectorAll("#retrieval tr.hit").forEach((tr) => tr.classList.toggle("on", tr.dataset.id === id));
  try {
    if (!S.stateCache[id]) S.stateCache[id] = await getJSON(`/api/state/${encodeURIComponent(id)}`);
    if (S.detail && S.detail.id === id) S.detail = { id, data: S.stateCache[id] };
  } catch (e) {
    if (S.detail && S.detail.id === id) S.detail = { id, error: e.message };
  }
  renderStateDetail();
}

function renderStateDetail() {
  const box = $("state-detail");
  const d = S.detail;
  if (!d) { box.innerHTML = ""; return; }
  if (d.loading) { box.innerHTML = '<div class="empty">loading the pro state...</div>'; return; }
  if (d.error) { box.innerHTML = `<div class="empty">could not load ${esc(d.id)}: ${esc(d.error)}</div>`; return; }
  const s = d.data;
  const link = s.video_link ? `<a href="${esc(s.video_link)}" target="_blank" rel="noopener" style="color:#74c0fc">${esc(s.video_title || s.video_id || "video")}</a>` : esc(s.video_title || "-");
  box.innerHTML =
    `<h2>Pro state <span class="sub mono">${esc(s.id)}</span></h2>` +
    `<div class="kv"><div>video</div><div>${link} ${s.creator ? `<span class="dim">by ${esc(s.creator)}</span>` : ""}</div>` +
    `<div>time</div><div>${esc(fmtT(s.video_time_s ?? 0))} in the video` + (s.clock ? `, game clock ${esc(s.clock)}` : "") + `</div>` +
    `<div>pro played</div><div><b>${esc(pretty(s.action_key || "-"))}</b> <span class="verdict-${esc(s.verdict || "")}">${esc(s.verdict || "")}</span>` +
    ` elixir trade ${typeof s.elixir_trade === "number" ? s.elixir_trade.toFixed(1) : "-"}</div></div>` +
    `<pre>${esc(s.context_text || "")}</pre>`;
}

// ------------------------------------------------------------------------------------------------ knowledge graph
const NW = 176, NH = 38, ROWH = 56;

function kgRequest(row) {
  const parsed = row.parsed || { enemy: [], hand: [] };
  const used = row.kg_used || [];
  const own = uniq(row.hand || []);
  const enemy = uniq([...(parsed.enemy || []), ...used.map((u) => u.threat)]).slice(0, 7);
  const cards = uniq([...enemy, ...own, ...used.map((u) => u.card)]);
  const deck = ((S.meta || {}).deck || []).map((c) => c.slug);
  const focus = cardOf(row.key) || "";
  return { cards, own, enemy, deck, focus,
           key: [cards, own, enemy, focus].map((x) => [].concat(x).join(",")).join("|") };
}

async function renderKG(row) {
  const box = $("kg");
  if (!row) { box.innerHTML = '<div class="empty">no decision selected</div>'; $("kg-text").innerHTML = ""; $("kg-legend").innerHTML = ""; return; }
  const req = kgRequest(row);
  S.kgKey = req.key;
  renderKGText(row, null);
  if (!req.cards.length) { box.innerHTML = '<div class="empty">no cards to draw for this decision</div>'; return; }
  if (!S.kgCache[req.key]) {
    box.innerHTML = '<div class="empty">loading the knowledge graph...</div>';
    const q = new URLSearchParams({ cards: req.cards.join(","), own: req.own.join(","), enemy: req.enemy.join(","),
                                    focus: req.focus, deck: req.deck.join(",") });
    try {
      S.kgCache[req.key] = await getJSON(`/api/kg?${q}`);
    } catch (e) {
      if (S.kgKey === req.key) box.innerHTML = `<div class="empty">knowledge graph unavailable (${esc(e.message)}; needs db/clashbot.sqlite)</div>`;
      return;
    }
  }
  if (S.kgKey !== req.key) return;                      // another decision was selected meanwhile
  drawKG(S.kgCache[req.key], row, req);
  renderKGText(row, S.kgCache[req.key]);
}

function drawKG(data, row, req) {
  const cols = { own: [], other: [], enemy: [] };
  const nodes = {};
  for (const n of data.nodes) {
    const side = cols[n.side] ? n.side : "other";
    nodes[n.id] = { ...n, side };
  }
  const order = (side, pref) => Object.values(nodes).filter((n) => n.side === side).sort((a, b) => {
    const ia = pref.indexOf(a.id), ib = pref.indexOf(b.id);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.label.localeCompare(b.label);
  });
  cols.own = order("own", req.own);
  cols.enemy = order("enemy", req.enemy);
  cols.other = order("other", []);
  const W = 840, CX = { own: 150, enemy: 420, other: 690 };           // our counters reach the enemy from the left, the KG's from the right
  const maxN = Math.max(cols.own.length, cols.other.length, cols.enemy.length, 1);
  const H = maxN * ROWH + 50;
  for (const side of Object.keys(cols)) {
    const n = cols[side].length;
    const y0 = 40 + (maxN * ROWH - n * ROWH) / 2 + ROWH / 2;
    cols[side].forEach((node, i) => { node.x = CX[side]; node.y = y0 + i * ROWH; });
  }
  const anchor = (a, b) => {
    const dx = a.side === b.side ? (a.side === "enemy" ? 1 : -1) : (b.x > a.x ? 1 : -1);
    return { x: a.x + dx * (NW / 2), y: a.y, dir: dx };
  };
  const path = (a, b) => {
    const p = anchor(a, b), q = anchor(b, a);
    let c1, c2;
    if (a.side === b.side) {
      const bulge = (a.side === "enemy" ? 1 : -1) * (46 + 5 * Math.abs(a.y - b.y) / ROWH);
      c1 = { x: p.x + bulge, y: p.y };
      c2 = { x: q.x + bulge, y: q.y };
    } else {
      const xm = (p.x + q.x) / 2;
      c1 = { x: xm, y: p.y };
      c2 = { x: xm, y: q.y };
    }
    // the label sits 80% of the way along the curve, near the card being countered, where the curves have fanned out
    const t = 0.8, u = 1 - t;
    const at = (k) => u * u * u * p[k] + 3 * u * u * t * c1[k] + 3 * u * t * t * c2[k] + t * t * t * q[k];
    return { d: `M${p.x} ${p.y} C${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${q.x} ${q.y}`, mx: at("x"), my: at("y") };
  };
  const defs = [...Object.entries(REL_STYLE).map(([rel, s]) => [rel, s.color]), ["USED", GOLD]].map(([rel, c]) =>
    `<marker id="ar-${rel}" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="12" markerHeight="12" markerUnits="userSpaceOnUse" orient="auto">` +
    `<path d="M0 1 L12 6 L0 11 z" fill="${c}"/></marker>`).join("");
  const parts = [`<defs>${defs}</defs>`];
  const head = { own: "our hand", enemy: "enemy cards", other: "other counters (KG top 3)" };
  for (const side of Object.keys(CX)) parts.push(`<text x="${CX[side]}" y="22" text-anchor="middle" style="fill:#aebbcc">${head[side]}</text>`);
  const edgeKey = (e) => `${e.counter || e.src}>${e.threat || e.dst}`;
  const usedPairs = {};
  for (const u of row.kg_used || []) {
    const k = `${u.card}>${u.threat}`;
    usedPairs[k] = Math.max(usedPairs[k] || 0, u.score);
  }
  for (const e of data.edges) {
    const style = REL_STYLE[e.rel];
    if (!style) continue;
    const from = nodes[style.arrow ? e.counter : e.src], to = nodes[style.arrow ? e.threat : e.dst];
    if (!from || !to || usedPairs[edgeKey(e)] !== undefined) continue;
    const g = path(from, to);
    const w = 1.6 + Math.min(e.weight || 0, 4) * 0.5;
    parts.push(`<path d="${g.d}" fill="none" stroke="${style.color}" stroke-width="${w}" stroke-dasharray="${style.dash}" opacity="0.7"` +
      `${style.arrow ? ` marker-end="url(#ar-${e.rel})"` : ""}><title>${esc(`${cardName(e.src)} ${e.rel} ${cardName(e.dst)}${e.evidence ? " - " + e.evidence : ""}`)}</title></path>`);
  }
  for (const [pair, score] of Object.entries(usedPairs)) {
    const [c, t] = pair.split(">");
    if (!nodes[c] || !nodes[t]) continue;
    const g = path(nodes[c], nodes[t]);
    parts.push(`<path d="${g.d}" fill="none" stroke="${GOLD}" stroke-width="5.5" marker-end="url(#ar-USED)"><title>${esc(`${cardName(c)} counters ${cardName(t)}: kg score ${score}`)}</title></path>`);
    parts.push(`<rect x="${g.mx - 20}" y="${g.my - 12}" width="40" height="22" rx="5" fill="#231f08" stroke="${GOLD}"/>` +
      `<text x="${g.mx}" y="${g.my + 5}" text-anchor="middle" style="fill:${GOLD};font-weight:700">${fmt(score, 1)}</text>`);
  }
  const fill = { own: "#12304f", other: "#1e2733", enemy: "#4a1a1a" };
  const stroke = { own: "#4aa3ff", other: "#8c9bb0", enemy: "#ff6b6b" };
  const chosen = req.focus;
  for (const node of Object.values(nodes)) {
    const isChosen = node.id === chosen;
    const label = node.label.length > 19 ? node.label.slice(0, 18) + "…" : node.label;
    parts.push(`<rect x="${node.x - NW / 2}" y="${node.y - NH / 2}" width="${NW}" height="${NH}" rx="9" fill="${fill[node.side]}" ` +
      `stroke="${isChosen ? GOLD : stroke[node.side]}" stroke-width="${isChosen ? 5 : 2}"/>` +
      `<text x="${node.x}" y="${node.y + 6}" text-anchor="middle">${isChosen ? "▶ " : ""}${esc(label)}</text>`);
  }
  $("kg").innerHTML = `<svg viewBox="0 0 ${W} ${H}">${parts.join("")}</svg>`;
  const legend = Object.entries(REL_STYLE).map(([rel, s]) =>
    `<span class="sw" style="border-color:${s.color};${s.dash ? "border-top-style:dashed" : ""}"></span>${esc(s.label)}`).join("") +
    `<span class="sw" style="border-color:${GOLD}"></span>used for a candidate's kg score` +
    `<span class="dot" style="background:${GOLD}"></span>card played`;
  $("kg-legend").innerHTML = `<div>${legend}</div>` +
    "<div>Arrow direction: from the card that <b>counters</b> to the card it counters. In the graph COUNTERS is stored " +
    "src = counter &rarr; dst = countered; ANSWERED_BY is stored threat &rarr; answer, so it is drawn reversed here.</div>";
}

function renderKGText(row, data) {
  const lines = (title, xs) => (xs && xs.length ? `<h3>${title}</h3><ul>${xs.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : "");
  let html = "";
  const used = row.kg_used;
  if (used === undefined) html += '<h3>Counters used</h3><div class="empty">not logged in this run (logged since MAC-AO)</div>';
  else if (!used.length) html += '<h3>Counters used</h3><div class="empty">the KG added no candidate this time</div>';
  else html += lines("Counters used by the candidate generator", used.map((u) => `${cardName(u.card)} counters ${cardName(u.threat)}  (${u.key})  score ${fmt(u.score, 2)}`));
  html += lines("KG v2 facts given to d1", row.kg_lines);
  html += lines("Tactics lines given to d1", row.tactics_lines);
  if (data && data.kg2) {
    const out = Object.entries(data.kg2).filter(([, a]) => a.length).map(([t, answers]) =>
      `${cardName(t)}: ` + answers.map((a) => `${cardName(a.card)} ${fmt(a.score, 2)}` +
        (a.support ? ` (pro n=${a.support}${typeof a.held_rate === "number" ? `, held ${Math.round(a.held_rate * 100)}%` : ""})` : "")).join(", "));
    html += lines("KG v2 answers from our deck (live lookup)", out);
  }
  $("kg-text").innerHTML = html;
}

// ------------------------------------------------------------------------------------------------ wiring
function renderAll() {
  renderHeader();
  renderTimeline();
  const row = S.decisions[S.sel] || null;
  renderScreen(row);
  const sig = `${S.runId}|${row ? row.i : "-"}`;
  if (S.view.rest !== sig) {
    S.view.rest = sig;
    if (S.detail && (!row || !(row.retrieval || []).some((h) => h.id === S.detail.id))) S.detail = null;
    renderDecision(row);
    renderRetrieval(row);
    renderKG(row);
  }
}

document.addEventListener("click", (ev) => {
  const t = ev.target;
  if (t.dataset && t.dataset.i !== undefined && t.closest("#timeline")) select(parseInt(t.dataset.i, 10), true);
  else if (t.dataset && t.dataset.shot) { S.shot = t.dataset.shot; renderAll(); }
  else {
    const tr = t.closest && t.closest("tr.hit");
    if (tr) showState(tr.dataset.id);
  }
});
document.addEventListener("keydown", (ev) => {
  if (ev.target && /^(INPUT|SELECT|TEXTAREA)$/.test(ev.target.tagName)) return;
  if (ev.key === "ArrowLeft") select((S.sel ?? 0) - 1, true);
  else if (ev.key === "ArrowRight") select((S.sel ?? -1) + 1, true);
  else if (ev.key === "f") { S.auto = !S.auto; if (S.auto) select(S.decisions.length - 1, false); else renderHeader(); }
});
$("follow-toggle").addEventListener("change", (ev) => {
  S.auto = ev.target.checked;
  if (S.auto && S.decisions.length) select(S.decisions.length - 1, false);
});
$("run-select").addEventListener("change", (ev) => {
  S.pinned = ev.target.value || null;
  const want = S.pinned || (S.current && S.current.id) || null;
  if (want !== S.runId) resetRun(want);
  renderAll();
});
window.addEventListener("resize", () => { S.view.timeline = null; renderTimeline(); });

loadRuns().then(tick);
