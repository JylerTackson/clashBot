/* Recording panel (MAC-AW). Plain JS, no libraries, no network beyond this server; loaded after app.js and sharing its globals
 * (S, $, esc, fmt, fmtT, getJSON, select, cardName, cardOf, pretty, modeColor, execClass, isTap, HOLD).
 *
 *   video run, live    the run's clip plays in step with the agent: the playhead is kept at (server clock - wall start) + the
 *                      video time of the first frame, every decision is drawn on the clip the moment it arrives (cyan ring at
 *                      its tile + card label + fading trail, a banner with mode / card / lane / reason, the reason the
 *                      policy is holding in a strip) and the agent's lag is shown.
 *   video run, replay  normal video controls; taps are drawn by their decision time as the clip plays; clicking a decision
 *                      (timeline, list) seeks there; the pro's plays (pro_plays.jsonl) are magenta rings.
 *   no video           a screenshot player: play / pause / 1x-4x through the decision screenshots in game time.
 */
"use strict";

const Rec = (() => {
  const RING_S = 6;            // a tap's ring is drawn this long (seconds of the clock that drives it) ...
  const TRAIL_S = 12;          // ... and its trail segment this long
  const PRO_S = 3;             // a pro play's ring
  const BANNER_S = 8;          // the banner shows a decision this long, then the "idle" line
  const HOLD_S = 4;            // the hold strip shows a hold verdict this long
  const SYNC_TOL_S = 0.6;      // the live playhead is corrected when it is this far from the wall clock
  const MAX_DWELL_S = 8;       // the screenshot player never holds a frame longer than this (game seconds) at 1x
  const CYAN = "#19e6ff";
  const MAGENTA = "#ff2bd6";
  const FONT = '700 15px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif';
  const DEFAULT_AR = 596 / 1280;

  const R = {
    runId: null, kind: null, lm: null, lmPerf: 0, serverAt: 0, videoFailed: false,
    video: null, canvas: null, img: null, w: 330, h: 700, dpr: 1, arKey: "",
    follow: false, wasFollow: false, started: false, openedAt: false, lastSeek: 0, lastPlay: 0, needClick: false,
    arrived: new Map(), seen: 0, firstBatch: true,
    frames: null, framesAt: -1, idx: 0, playing: false, speed: 1, shownAt: 0, shownFile: null,
    lastDyn: 0, html: {}, ready: false,
  };

  const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
  const setHTML = (key, el, html) => { if (R.html[key] !== html) { R.html[key] = html; el.innerHTML = html; } };
  const shotUrl = (file) => `/api/shot/${encodeURIComponent(S.runId)}/${encodeURIComponent(file)}`;

  // ---------------------------------------------------------------------------------------------- clocks
  const serverNow = () => R.serverAt + (performance.now() - R.lmPerf) / 1000;
  const liveTarget = () => serverNow() - R.lm.wall_start + R.lm.video_t0;
  const playhead = () => (R.video ? R.video.currentTime : 0);

  // ---------------------------------------------------------------------------------------------- setup
  function init() {
    if (R.ready) return;
    R.ready = true;
    R.video = $("rec-video");
    R.canvas = $("rec-canvas");
    R.img = $("rec-img");
    R.video.muted = true;
    R.video.playsInline = true;
    R.video.addEventListener("loadedmetadata", layout);
    R.video.addEventListener("seeked", () => { syncSelection(true); });
    R.video.addEventListener("play", () => { R.started = true; R.needClick = false; });
    R.video.addEventListener("error", () => {                      // a browser that cannot play the file: fall back to the screenshots
      if (R.video.getAttribute("src")) { R.videoFailed = true; R.kind = null; renderAll(); }
    });
    R.img.addEventListener("load", layout);
    $("rec-click").addEventListener("click", () => {
      R.started = true;
      R.needClick = false;
      R.video.play().catch(() => {});
      refresh();
    });
    $("rec-feed").addEventListener("click", (ev) => {
      const it = ev.target.closest && ev.target.closest(".item");
      if (it) select(parseInt(it.dataset.i, 10), true);
    });
    $("rec-controls").addEventListener("click", controlClick);
    $("rec-controls").addEventListener("input", (ev) => {
      if (ev.target.id === "rec-seek") { R.playing = false; gotoFrame(parseInt(ev.target.value, 10)); }
    });
    window.addEventListener("resize", layout);
    requestAnimationFrame(frame);
  }

  function reset(id) {
    R.runId = id;
    Object.assign(R, { kind: null, lm: null, videoFailed: false, follow: false, wasFollow: false, started: false, openedAt: false, needClick: false,
                       frames: null, framesAt: -1, idx: 0, playing: false, speed: 1, shownFile: null, seen: 0, firstBatch: true, arKey: "" });
    R.arrived.clear();
    R.html = {};
    if (R.video) {
      R.video.pause();
      R.video.removeAttribute("src");
      R.video.removeAttribute("data-run");
      R.video.load();
    }
    if (R.img) R.img.removeAttribute("src");
  }

  // Called by app.js every second, after the run's rows were read: fetch what only this panel needs.
  async function poll() {
    init();
    const id = S.runId;
    if (!id) return;
    if (R.runId !== id) reset(id);
    const enc = encodeURIComponent(id);
    const gen = S.gen;
    try {
      if (R.lm === null || !R.lm.finished) {
        const p0 = performance.now();
        const lm = await getJSON(`/api/run/${enc}/livemeta`);
        const p1 = performance.now();
        if (gen !== S.gen) return;
        R.lm = lm;
        R.lmPerf = (p0 + p1) / 2;                              // the server stamped `now` somewhere in the middle of the request
        R.serverAt = lm.now;
      }
      if (wantKind() === "shots" && (R.frames === null || R.framesAt !== S.decisions.length || S.live)) {
        const f = await getJSON(`/api/run/${enc}/player`);
        if (gen !== S.gen) return;
        R.frames = f;
        R.framesAt = S.decisions.length;
      }
    } catch (e) { /* keep what we have; the page's own poll reports a dead server */ }
  }

  function wantKind() {
    const lm = R.lm;
    if (!lm) return null;
    if (lm.has_video && !R.videoFailed) return "video";
    if ((S.meta && S.meta.has_shots) || (R.frames && R.frames.frames.length)) return "shots";
    if (!lm.finished && S.runId && !S.decisions.length) return "wait";
    return null;
  }

  // ---------------------------------------------------------------------------------------------- render (each tick and on selection)
  function render() {
    init();
    if (R.runId !== S.runId) reset(S.runId);
    const lm = R.lm;
    const kind = wantKind();
    if (kind !== R.kind) switchTo(kind);
    $("rec-panel").hidden = !kind;
    $("screen-panel").hidden = kind === "video";
    $("grid").classList.toggle("no-screen", kind === "video");
    if (!kind) return;
    // video: does the page follow the agent's wall clock?
    R.follow = !!(kind === "video" && lm.realtime && lm.wall_start && lm.live && !lm.finished);
    markArrivals();
    if (kind === "shots") syncShotsToSelection();
    refresh();
  }

  function switchTo(kind) {
    R.kind = kind;
    R.html = {};
    const video = kind === "video", shots = kind === "shots";
    R.video.hidden = !video;
    R.canvas.hidden = !video;
    R.img.hidden = !shots;
    document.querySelector("#rec-panel .rec-body").hidden = kind === "wait";
    $("rec-click").hidden = true;
    if (video) {
      const url = `/api/run/${encodeURIComponent(S.runId)}/video`;
      if (R.video.getAttribute("data-run") !== S.runId) {
        R.video.setAttribute("data-run", S.runId);
        R.video.src = url;
        R.video.load();
      }
      $("rec-controls").innerHTML = "";
      $("rec-legend").innerHTML =
        `<span class="rec-legend-ring" style="border-color:${CYAN}"></span>the agent's tap (card at its tile; the trail joins its recent taps)` +
        `<span class="rec-legend-ring" style="border-color:${MAGENTA}"></span>the pro's tap (runs on a pro video)`;
    } else if (shots) {
      R.video.pause();
      $("rec-controls").innerHTML =
        '<button type="button" data-act="prev" title="previous decision">&#9664;&#9664;</button>' +
        '<button type="button" data-act="play" id="rec-play">&#9654; play</button>' +
        '<button type="button" data-act="next" title="next decision">&#9654;&#9654;</button>' +
        '<span class="speeds">speed ' + [1, 2, 3, 4].map((s) => `<button type="button" data-speed="${s}">${s}x</button>`).join("") + "</span>" +
        '<input type="range" id="rec-seek" min="0" max="0" value="0" aria-label="decision">' +
        '<span id="rec-pos"></span>';
      $("rec-legend").innerHTML = "The tap (card slot in cyan, field target in magenta) is drawn on each screenshot by the recorder.";
    } else {
      R.video.pause();
      $("rec-controls").innerHTML = "";
      $("rec-legend").innerHTML = "";
    }
    layout();
  }

  function aspect() {
    const v = R.video;
    if (R.kind === "video" && v.videoWidth) return v.videoWidth / v.videoHeight;
    if (R.kind === "shots" && R.img.naturalWidth) return R.img.naturalWidth / R.img.naturalHeight;
    const fs = R.lm && R.lm.frame_size;
    return fs && fs[0] && fs[1] ? fs[0] / fs[1] : DEFAULT_AR;
  }

  function layout() {
    if (!R.ready || !R.kind || R.kind === "wait") return;
    const ar = aspect();
    const h = Math.round(clamp(window.innerHeight - 300, 460, 720));
    const w = Math.round(h * ar);
    const key = `${w}x${h}@${window.devicePixelRatio || 1}`;
    if (key === R.arKey) return;
    R.arKey = key;
    const stage = $("rec-stage");
    stage.style.width = `${w}px`;
    stage.style.height = `${h}px`;
    $("rec-controls").style.maxWidth = `${Math.max(w, 330)}px`;
    R.w = w;
    R.h = h;
    R.dpr = window.devicePixelRatio || 1;
    R.canvas.width = Math.round(w * R.dpr);
    R.canvas.height = Math.round(h * R.dpr);
  }

  // Decisions the page has just received are "new": in a live run they are drawn from the moment they arrive.
  function markArrivals() {
    const n = S.decisions.length;
    if (n < R.seen) { R.seen = 0; R.arrived.clear(); R.firstBatch = true; }      // the run's files were rewritten
    const target = R.follow ? liveTarget() : 0;
    for (let i = R.seen; i < n; i++) {
      // rows that were already there when the page opened are as old as their video time says; later rows arrived just now
      const age = R.firstBatch ? Math.max(0, target - S.decisions[i].t) : 0;
      R.arrived.set(i, performance.now() - age * 1000);
    }
    if (n) R.firstBatch = false;
    R.seen = n;
  }

  // ---------------------------------------------------------------------------------------------- the video
  function syncVideo() {
    const v = R.video, lm = R.lm;
    if (!v || !lm) return;
    if (R.follow) {
      R.wasFollow = true;
      v.controls = false;
      if (v.readyState < 1) return;
      const target = liveTarget();
      const now = performance.now();
      if (isFinite(v.duration) && target >= v.duration) { if (!v.paused) v.pause(); return; }
      if (Math.abs(v.currentTime - target) > SYNC_TOL_S && now - R.lastSeek > 800) { R.lastSeek = now; v.currentTime = Math.max(target, 0); }
      if (v.paused && !R.needClick && now - R.lastPlay > 1500) {
        R.lastPlay = now;
        const p = v.play();
        if (p && p.catch) p.catch(() => { R.needClick = true; });         // the browser wants a click first
      }
    } else {
      v.controls = !(lm.realtime && !lm.wall_start && !lm.finished);        // not while the agent is still loading
      if (R.wasFollow) { R.wasFollow = false; v.pause(); }                  // the run finished: stop where we are, the controls take over
      if (!R.openedAt && v.readyState >= 1) {                               // a replay opens where the run started (a run on a long video, --start)
        R.openedAt = true;
        if (typeof lm.video_t0 === "number" && lm.video_t0 > 0.5 && v.currentTime < lm.video_t0 - 0.5) v.currentTime = lm.video_t0;
      }
    }
    const pending = lm.realtime && !lm.wall_start && !lm.finished;
    $("rec-click").hidden = !(v.paused && !R.started && v.readyState >= 1 && !pending && (!R.follow || R.needClick));
  }

  function lastAtOrBefore(t) {
    let best = -1;
    for (let i = 0; i < S.decisions.length; i++) if (S.decisions[i].t <= t && (best < 0 || S.decisions[i].t >= S.decisions[best].t)) best = i;
    return best;
  }

  // Replay: the Decision / Retrieval / graph panels follow the playhead.
  function syncSelection(force) {
    if (R.kind !== "video" || R.follow || S.auto) return;
    const v = R.video;
    if (!force && (v.paused || v.seeking)) return;
    const i = lastAtOrBefore(v.currentTime + 0.05);
    if (i >= 0 && i !== S.sel) select(i, false);
  }

  function onSelect(i, byUser) {
    if (R.kind === "video" && !R.follow && byUser && R.video.readyState >= 1 && S.decisions[i]) R.video.currentTime = Math.max(S.decisions[i].t, 0);
  }

  // ---------------------------------------------------------------------------------------------- drawing
  function tilePx(tile) {
    const g = R.lm.arena.homography, grid = R.lm.arena.grid;
    const cx = clamp(tile[0] + 0.5, 0.5, grid.cols - 0.5), cy = clamp(tile[1] + 0.5, 0.5, grid.rows - 0.5);   // the executor's tile_target
    const d = g[2][0] * cx + g[2][1] * cy + g[2][2];
    return [((g[0][0] * cx + g[0][1] * cy + g[0][2]) / d) * R.w, ((g[1][0] * cx + g[1][1] * cy + g[1][2]) / d) * R.h];
  }

  function tileSizePx() {
    const a = R.lm.arena.rect;
    return ((a[2] - a[0]) * R.w) / R.lm.arena.grid.cols;
  }

  function label(ctx, text, x, y, color) {
    ctx.font = FONT;
    const w = ctx.measureText(text).width + 12, h = 22;
    const left = clamp(x - w / 2, 2, R.w - w - 2), top = clamp(y - h, 2, R.h - h - 2);
    ctx.fillStyle = "rgba(8,14,22,0.85)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.rect(left, top, w, h);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, left + 6, top + h / 2 + 1);
  }

  function ring(ctx, x, y, r, color, alpha, width, dash) {
    ctx.globalAlpha = clamp(alpha, 0, 1);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  const ageOf = (i, row, ph, now) => (R.follow ? ((R.arrived.get(i) !== undefined ? now - R.arrived.get(i) : 0) / 1000) : ph - row.t);

  function draw() {
    const canvas = R.canvas;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(R.dpr, 0, 0, R.dpr, 0, 0);
    ctx.clearRect(0, 0, R.w, R.h);
    const lm = R.lm;
    if (!lm || !lm.arena) return;
    const ph = playhead(), now = performance.now();
    const base = tileSizePx();
    const radius = Math.max(base * 1.15, 14);
    const taps = [];
    S.decisions.forEach((row, i) => {
      if (!isTap(row) || !Array.isArray(row.tile)) return;
      const age = ageOf(i, row, ph, now);
      if (age < 0 || age > TRAIL_S) return;
      const [x, y] = tilePx(row.tile);
      taps.push({ row, age, x, y });
    });
    taps.sort((a, b) => a.row.t - b.row.t);
    ctx.lineCap = "round";
    for (let k = 1; k < taps.length; k++) {                     // the fading trail joins consecutive taps
      const a = taps[k - 1], b = taps[k];
      ctx.globalAlpha = clamp(0.75 * (1 - b.age / TRAIL_S), 0, 1);
      ctx.strokeStyle = CYAN;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    taps.forEach((tp, k) => {
      const newest = k === taps.length - 1;
      const fade = tp.age < RING_S - 2 ? 1 : clamp((RING_S - tp.age) / 2, 0, 1);
      if (tp.age > RING_S) {                                      // an old tap: a small dot at the end of its trail
        ctx.globalAlpha = clamp(0.8 * (1 - tp.age / TRAIL_S), 0, 1);
        ctx.fillStyle = CYAN;
        ctx.beginPath();
        ctx.arc(tp.x, tp.y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        return;
      }
      ring(ctx, tp.x, tp.y, radius, CYAN, fade, newest ? 5 : 3);
      if (tp.age < 1.2) ring(ctx, tp.x, tp.y, radius * (1 + tp.age * 1.4), CYAN, 1 - tp.age / 1.2, 3);   // the ripple of a new tap
      ctx.globalAlpha = fade;
      label(ctx, cardName(cardOf(tp.row.key) || "?"), tp.x, tp.y - radius - 4, CYAN);
      ctx.globalAlpha = 1;
    });
    (S.pro || []).forEach((p) => {                              // the pro's taps, on runs made on a pro video
      const age = ph - p.t;
      if (age < 0 || age > PRO_S || !Array.isArray(p.tile)) return;
      const [x, y] = tilePx(p.tile);
      const a = clamp(1 - (age / PRO_S) ** 2, 0, 1);
      ring(ctx, x, y, radius, MAGENTA, a, 4, [6, 4]);
      ctx.globalAlpha = a;
      label(ctx, `pro: ${cardName(p.card || "?")}`, x, y - radius - 4, MAGENTA);
      ctx.globalAlpha = 1;
    });
  }

  // ---------------------------------------------------------------------------------------------- text parts
  function currentRow() {
    // { i, row, age } of the decision the banner is about (null: none yet)
    const n = S.decisions.length;
    if (!n) return null;
    if (R.kind === "shots") {
      const f = R.frames && R.frames.frames[R.idx];
      return f ? { i: f.i, row: S.decisions[f.i], age: 0 } : null;
    }
    if (R.follow) {
      const i = n - 1;
      return { i, row: S.decisions[i], age: ageOf(i, S.decisions[i], 0, performance.now()) };
    }
    const i = lastAtOrBefore(playhead() + 0.05);
    return i < 0 ? null : { i, row: S.decisions[i], age: playhead() - S.decisions[i].t };
  }

  const laneZone = (key) => {
    const p = String(key || "").split("|");
    return { lane: p[1] || "", zone: p[2] || "" };
  };

  function bannerHtml(cur) {
    if (!cur || !cur.row) return { html: '<span class="dim">no decision yet: the agent is watching the recording</span>', color: "#2a3a50", idle: true };
    const r = cur.row, c = modeColor(r.mode), ex = r.executor || {};
    if (R.kind !== "shots" && cur.age > BANNER_S) {
      return { html: `<span class="dim">watching &middot; last decision ${esc(fmtT(r.t))}: ${esc(r.key === HOLD ? "hold" : pretty(r.key))} (${Math.round(cur.age)} s ago)</span>`, color: "#2a3a50", idle: true };
    }
    const held = r.key === HOLD;
    const { lane, zone } = laneZone(r.key);
    const what = held ? "<b>HOLD</b> (no tap)" : `<b>${esc(cardName(cardOf(r.key)))}</b>` + (lane ? ` &middot; ${esc(lane)} lane` : "") + (zone ? ` &middot; ${esc(zone)}` : "") +
      (Array.isArray(r.tile) ? ` <span class="dim">@[${fmt(r.tile[0], 1)}, ${fmt(r.tile[1], 1)}]</span>` : "");
    const html = `<span class="chip" style="border-color:${c};color:${c}">${esc(r.mode)}</span> ${what} ` +
      `<span class="status ${execClass(r)}">${esc(ex.status || "dry_run")}</span> <span class="dim">t=${fmt(r.t, 1)} s</span>` +
      `<span class="why">${esc(r.reason)}</span>`;
    return { html, color: c, idle: false };
  }

  function holdHtml() {
    const H = S.holds;
    let ref = null;
    if (R.kind === "shots") { const f = R.frames && R.frames.frames[R.idx]; ref = f ? f.t : null; }
    else if (R.follow) ref = R.lm.progress && R.lm.progress.latest_t;
    else ref = playhead();
    if (ref === null || ref === undefined || !H.length) return '<span class="dim">no hold verdict logged yet</span>';
    let h = null;
    for (let i = H.length - 1; i >= 0; i--) if (H[i].t <= ref + 0.05) { h = H[i]; break; }
    if (!h || ref - h.t > HOLD_S) return '<span class="dim">no hold verdict in the last few seconds</span>';
    return `<b>${h.act ? "policy: act" : "policy: wait"}</b> &middot; ${esc(h.mode)} &middot; ${esc(h.reason)}` +
      (typeof h.elixir === "number" ? ` <span class="dim">(elixir ${fmt(h.elixir, 1)})</span>` : "");
  }

  function feedHtml(cur) {
    const n = S.decisions.length;
    let last = n - 1;
    if (R.kind === "shots" || !R.follow) last = Math.max(cur ? cur.i : -1, Math.min(n - 1, 8));      // at the start of a replay: the first moves, to click on
    const items = [];
    for (let i = last; i >= 0 && items.length < 9; i--) {
      const r = S.decisions[i], held = r.key === HOLD;
      items.push(`<div class="item ${held ? "hold" : ""} ${i === S.sel ? "sel" : ""}" data-i="${i}"><span class="when">${esc(fmtT(r.t))}</span>` +
        `<span class="what" style="color:${held ? "" : modeColor(r.mode)}">${held ? "hold" : esc(cardName(cardOf(r.key)))}${Array.isArray(r.tile) && !held ? ` @[${fmt(r.tile[0], 0)},${fmt(r.tile[1], 0)}]` : ""}</span>` +
        `<span class="reason">${esc(r.mode)}: ${esc(r.reason)}</span></div>`);
    }
    return items.length ? `<h2 style="font-size:18px;margin:6px 0 2px">The agent's moves <span class="sub">${R.kind === "video" && !R.follow ? "click one to select it and seek the recording there" : "newest first; click one to select it"}</span></h2>${items.join("")}` : "";
  }

  function badge() {
    const b = $("rec-badge"), sub = $("rec-sub");
    const lm = R.lm || {};
    let cls = "rec-replay", text = "REPLAY", note = "";
    if (R.kind === "wait") { cls = "rec-wait"; text = "STARTING"; note = "the agent is loading its models; the clip starts with its first frame"; }
    else if (R.kind === "video" && R.follow) { cls = "rec-live"; text = "LIVE — the agent is watching this recording now"; }
    else if (R.kind === "video" && lm.realtime && !lm.wall_start && !lm.finished) { cls = "rec-wait"; text = "STARTING"; note = "the agent is loading its models; the clip starts with its first frame"; }
    else if (R.kind === "video") note = "video controls below; the agent's taps are drawn at their decision times";
    else if (R.kind === "shots" && S.live) { cls = "rec-live"; text = "LIVE"; note = "decision screenshots as they arrive (no video was recorded for this run)"; }
    else if (R.kind === "shots") note = "screenshot player: no video for this run, so it steps through the decision screenshots in game time";
    b.className = `badge ${cls}`;
    b.textContent = text;
    sub.textContent = note;
  }

  function lagHtml() {
    const lm = R.lm;
    if (R.kind === "video" && R.follow) {
      const p = lm.progress || {};
      if (p.latest_t === null || p.latest_t === undefined) return { html: "agent lag: <b>waiting for the first processed frame</b>", slow: false };
      const lag = typeof p.lag_s === "number" ? p.lag_s : null;
      return { html: `agent lag <b>${lag === null ? "-" : fmt(Math.max(lag, 0), 1) + " s"}</b> &middot; the agent has seen the recording up to <b>${fmt(p.latest_t, 1)} s</b>, the playhead is at <b>${fmt(playhead(), 1)} s</b>`,
               slow: lag !== null && lag > 3 };
    }
    if (R.kind === "video") return { html: `playhead <b>${fmt(playhead(), 1)} s</b> of ${isFinite(R.video.duration) ? fmt(R.video.duration, 0) : "?"} s &middot; ${S.decisions.length} decisions`, slow: false };
    if (R.kind === "shots") {
      const f = R.frames && R.frames.frames[R.idx];
      return { html: f ? `game time <b>${fmt(f.t, 1)} s</b> &middot; decision ${f.i + 1} of ${S.decisions.length}` : "no screenshots yet", slow: false };
    }
    return { html: "", slow: false };
  }

  // Everything that changes with the clock; runs ~4 times a second and after every render.
  function refresh() {
    if (!R.kind) return;
    badge();
    const cur = currentRow();
    if (R.kind === "wait") {
      const b = $("rec-banner");
      b.className = "rec-banner idle";
      b.style.borderLeftColor = "#2a3a50";
      setHTML("banner", b, "waiting for the agent: it loads the perception and ranker models first (a few seconds), then plays the recording in real time");
      return;
    }
    const bn = bannerHtml(cur);
    const b = $("rec-banner");
    b.className = `rec-banner ${bn.idle ? "idle" : ""}`;
    b.style.borderLeftColor = bn.color;
    setHTML("banner", b, bn.html);
    const lag = lagHtml();
    $("rec-lag").className = `rec-lag ${lag.slow ? "lagging" : ""}`;
    setHTML("lag", $("rec-lag"), lag.html);
    setHTML("hold", $("rec-hold"), holdHtml());
    setHTML("feed", $("rec-feed"), feedHtml(cur) + `<!--${S.sel}-->`);
    if (R.kind === "shots") updateControls();
    if (R.kind === "video") syncSelection(false);
  }

  // ---------------------------------------------------------------------------------------------- the screenshot player
  const frameList = () => (R.frames && R.frames.frames) || [];

  function showFrame() {
    const f = frameList()[R.idx];
    if (!f) { R.img.removeAttribute("src"); R.shownFile = null; return; }
    if (R.shownFile !== f.file) {
      R.shownFile = f.file;
      R.img.src = shotUrl(f.file);
      const next = frameList()[R.idx + 1];
      if (next) new Image().src = shotUrl(next.file);          // warm the next one so the step does not flash
    }
  }

  function gotoFrame(k) {
    const F = frameList();
    if (!F.length) return;
    R.idx = clamp(k, 0, F.length - 1);
    R.shownAt = performance.now();
    showFrame();
    if (S.sel !== F[R.idx].i) { S.auto = false; select(F[R.idx].i, false); }
    updateControls();
  }

  // Not playing: the player shows the frame of the selected decision (timeline click, arrow keys, "follow newest").
  function syncShotsToSelection() {
    const F = frameList();
    if (!F.length) return;
    if (R.playing) return;
    let k = F.findIndex((f) => f.i === S.sel);
    if (k < 0 && S.auto) k = F.length - 1;
    if (k >= 0 && k !== R.idx) { R.idx = k; R.shownAt = performance.now(); }
    if (R.idx >= F.length) R.idx = F.length - 1;
    showFrame();
  }

  function stepShots() {
    const F = frameList();
    if (!R.playing || !F.length) return;
    const f = F[R.idx];
    const dwell = (f.until === null ? Infinity : Math.min(f.until - f.t, MAX_DWELL_S)) / R.speed * 1000;
    if (performance.now() - R.shownAt >= dwell && R.idx < F.length - 1) gotoFrame(R.idx + 1);
    else if (R.idx >= F.length - 1 && !S.live && performance.now() - R.shownAt > 1500) { R.playing = false; updateControls(); }
  }

  function controlClick(ev) {
    const t = ev.target.closest && ev.target.closest("button");
    if (!t) return;
    const F = frameList();
    if (t.dataset.speed) { R.speed = parseInt(t.dataset.speed, 10); R.shownAt = performance.now(); }
    else if (t.dataset.act === "play") {
      R.playing = !R.playing;
      if (R.playing) {
        S.auto = false;
        if (F.length && R.idx >= F.length - 1 && !S.live) R.idx = 0;      // play again from the start
        R.shownAt = performance.now();
        gotoFrame(R.idx);
      }
    } else if (t.dataset.act === "prev") { R.playing = false; gotoFrame(R.idx - 1); }
    else if (t.dataset.act === "next") { R.playing = false; gotoFrame(R.idx + 1); }
    updateControls();
  }

  function updateControls() {
    const play = $("rec-play");
    if (!play) return;
    const F = frameList();
    play.innerHTML = R.playing ? "&#10074;&#10074; pause" : "&#9654; play";
    document.querySelectorAll("#rec-controls [data-speed]").forEach((b) => b.classList.toggle("on", parseInt(b.dataset.speed, 10) === R.speed));
    const seek = $("rec-seek");
    if (seek && document.activeElement !== seek) { seek.max = Math.max(F.length - 1, 0); seek.value = R.idx; }
    const f = F[R.idx];
    $("rec-pos").textContent = f ? `${R.idx + 1}/${F.length} · t=${fmt(f.t, 1)} s` : "no screenshots yet";
  }

  // ---------------------------------------------------------------------------------------------- animation loop
  function frame() {
    requestAnimationFrame(frame);
    if (!R.kind) return;
    if (R.kind === "video") { syncVideo(); draw(); }
    else if (R.kind === "shots") stepShots();
    const now = performance.now();
    if (now - R.lastDyn > 250) { R.lastDyn = now; refresh(); }
  }

  return { poll, render, onSelect, _state: R };
})();
