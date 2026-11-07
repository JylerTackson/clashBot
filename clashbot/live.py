"""Live loop CLI (D-007, D-031, D-041): capture -> perception -> live state -> act policy -> candidates -> ranker -> executor.

Dry run by default: the executor only logs the clicks it would make. Packages A-F (detector, mirroring source, live
state, ranker, planner, executor) are imported lazily inside the builders so this module stays light and testable
with fakes. The act policy is reactive by default (clashbot.policy: defend / leak / punish, else hold and save
elixir); `--policy learned --act-model PATH` adds the d1 act-head (MAC-X); `--policy trigger` restores the
fixed-interval `should_decide`. The loop plays only cards it can see in the hand (`require_hand`) and asks the
policy for a spell's aim point (`target_tile`) when it has one. With a `TacticalEngine` (default; `--no-tactics` turns
it off, MAC-Z3) the candidates start with the engine's concrete moves, every candidate carries a note, and the ranker
sees the engine's lines after the context; `--ranker tactics` plays the engine's own pick without any model.
The hero ability is used sparingly (D-049, MAC-AL P4): a candidate `ability:musketeer|lane|zone` (MAC-AI) joins the
ranker's options only when the engine proposes it (the turret simulated at its spawn tile prevents 1,000+ tower HP) AND
the policy lists `ability:musketeer` among the decision's allowed cards, at most once per `ABILITY_MIN_GAP_S` game
seconds; its note carries the engine's reason and "3 elixir, use rarely", and it is offered even when no card candidate
survives. When the ranker picks it, `executor.tap_ability` taps the button instead of a card (no hand check) and the 3
elixir enter the state builder's accounting.
The knowledge graph v2 (`db/kg2.sqlite`, MAC-AK K4b; `--no-kg2` turns it off) gives the engine its pro answers and
placement offsets, the policy its reserve for predicted cards, and the ranker up to 3 `KG:` fact lines after the
tactics lines (the act policy's context is unchanged).
`--record-shots` (MAC-AF, off by default) saves what the bot saw and tapped: an annotated screenshot per decision and
the frames 0.5 s / 1.5 s after a tap, written by a background thread (`clashbot.recorder_shots`, review with
tools/eval/review_shots.py); `step` only queues them.
`--planner coach` adds the Qwen3-14B coach (D-046): a `CoachWorker` thread rewrites a short plan every
`--coach-period` game seconds and the loop hands the latest one to the ranker; `step` never waits for it.
Usage: python3 -m clashbot.live --calib CALIB --deck c1,...,c8 [--source video --video PATH]
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import queue
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from clashbot.actions import (ABILITY_COST, ABILITY_KEY, ABILITY_MIN_GAP_S, ABILITY_USE_RARELY, is_ability_key,
                              load_placement_prior, option_text, placement_tile)
from clashbot.arbiter import ArbiterConfig, RuleArbiter, urgent_from_report
from clashbot.candidates import AbilityMove, CandidateGenerator
from clashbot.config import HOLD_KEY
from clashbot.game_rules import load_cards
from clashbot.kg import KnowledgeGraph
from clashbot.loop import should_decide
from clashbot.planner import (COACH_MODEL_ID, COACH_PERIOD_S, COACH_PREFILL_STEP, COACH_TOKEN_YIELD_S, COACH_YIELD_S,
                              Plan, lookup_summary, should_update)
from clashbot.policy import ActDecision, ReactivePolicy
from clashbot.retrieval import StateIndex
from clashbot.tactics import build_engine, kg_fact_lines, rank_context

STAGES = ("capture", "perception", "live_state", "trigger", "candidates", "rank", "execute", "total")
FRAME_STAGES = ("capture", "perception", "live_state")
TARGET_P95_MS = 300.0
COACH_TICK_S = 1.0            # the loop offers the coach worker a job at most once per game second
COACH_QUIET_ETA_S = 10.0      # the coach starts no job while a push is this close to a tower (or we defend / save)
COACH_BUSY_MODES = ("defend", "save")
_TOWER_ETA_RE = re.compile(r"tower in (\d+(?:\.\d+)?)s")      # the tactics engine's push line: "..., tower in 6.2s; ..."
FALLBACK_MIN_ELIXIR = 9.5     # a "fallback" candidate (no knn/kg support) is only played from a (nearly) full bar
FALLBACK_OK_MODES = ("counter", "spell", "leak", "punish", "learned", "defend")     # modes whose fallback candidates may be played
ABILITY_RETRY_S = 10.0        # the turret lives this long after a tap; the gap between two taps is ABILITY_MIN_GAP_S (D-049)
TOS_NOTE = ("Live runs only in Training Camp or friendly battles "
            "(Supercell terms of service prohibit bots in normal play).")


@dataclass
class LiveDecision:
    t: float
    idx: int
    key: str
    tile: list[float] | None
    candidates: list[dict]        # [{"key", "score", "sources", "tile", "note"}]
    probs: list[float]
    executor: dict                # the executor's returned entry
    timings_ms: dict              # every STAGES name
    mode: str = "trigger"         # policy mode: defend | leak | punish (reactive) or trigger
    reason: str = ""              # the policy's human-readable reason
    tactics_lines: list = field(default_factory=list)      # the engine's lines shown to the ranker ([] without one)
    arbiter_reason: str | None = None                      # which arbiter rule narrowed d1's options (None without one)
    arbiter_subset: list | None = None                     # [{"key", "p"}]: the options d1 chose among, renormalised
    plan_t: float | None = None                            # game time of the coach plan d1 read (None: the pregame plan)
    plan_age_s: float | None = None                        # decision time minus plan_t
    tile_source: str | None = None                         # where the tile came from: spell | pro_offset | engine | candidate | prior | ability
    kg_lines: list = field(default_factory=list)          # the knowledge graph's fact lines shown to the ranker ([] without KG2)


def _accepts_notes(fn) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "notes" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def rank_keys(ranker, context: str, keys: list[str], plan: str | None,
              notes: dict[str, str] | None = None) -> tuple[int, list[float]]:
    """Pick one of `keys`; returns (index, probs). A single key needs no model call (d1 wants >= 2 options).
    `notes` (key -> one line, MAC-Z3) go to rankers whose `choose_keys` accepts them; others are called as before."""
    if not keys:
        raise ValueError("rank_keys needs at least one key")
    if len(keys) == 1:
        return 0, [1.0]
    if hasattr(ranker, "choose_keys"):
        if notes and _accepts_notes(ranker.choose_keys):
            res = ranker.choose_keys(context, keys, plan, notes=notes)
        else:
            res = ranker.choose_keys(context, keys, plan)
    else:
        res = ranker.choose(context, [option_text(k) for k in keys], plan)
    return res.index, list(res.probs)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def ability_cooling(state: dict) -> dict:
    """`state` with the hero ability marked unaffordable (a shallow copy): after a tap the engine and the policy must not
    ask for it again while the turret lives; a state without an affordable ability button comes back as it is."""
    ab = (state.get("own") or {}).get("hero_ability")
    if not isinstance(ab, dict) or not ab.get("affordable"):
        return state
    return {**state, "own": {**state["own"], "hero_ability": {**ab, "affordable": False}}}


def _roles_of(tactics) -> dict:
    """The card role table: the tactics engine's own, else the committed Z1 table."""
    roles = getattr(tactics, "roles", None)
    if isinstance(roles, dict) and roles:
        return roles["cards"] if isinstance(roles.get("cards"), dict) else roles
    from clashbot.card_roles import load_roles
    return load_roles()


def fallback_allowed(verdict, card: str, roles: dict) -> bool:
    """A fallback-only candidate is not held when the policy decision is an act of a mode in FALLBACK_OK_MODES and the
    card is a win condition (role table `roles`) or one of the decision's allowed cards (MAC-AG): the hold only guards
    against random filler plays."""
    if getattr(verdict, "mode", None) not in FALLBACK_OK_MODES:
        return False
    role = roles.get(card) or {}
    return "win_condition" in (role.get("roles") or []) or card in (getattr(verdict, "allowed_cards", None) or ())


def top_source(sources: dict) -> str:
    """Which channel a candidate came from: tac (the tactics engine), ability (the hero ability option), knn / kg (the
    larger), else fallback."""
    if "tac" in sources:
        return "tac"
    if "ability" in sources:
        return "ability"
    knn, kg = sources.get("knn") or 0.0, sources.get("kg") or 0.0
    if knn <= 0 and kg <= 0:
        return "fallback"
    return "knn" if knn >= kg else "kg"


def decision_source(d: "LiveDecision") -> str:
    """`top_source` of the chosen candidate, or "hold" when the decision held."""
    if d.key == HOLD_KEY:
        return "hold"
    for c in d.candidates:
        if c["key"] == d.key:
            return top_source(c.get("sources") or {})
    return "fallback"


def _pcts(vals: list[float]) -> dict:
    if not vals:
        return {"p50": None, "p95": None}
    return {"p50": round(float(np.percentile(vals, 50)), 2), "p95": round(float(np.percentile(vals, 95)), 2)}


log = logging.getLogger("clashbot.coach")


def coach_quiet(mode: str | None, tactics_lines: list[str] | None, eta_s: float = COACH_QUIET_ETA_S) -> bool:
    """Is it quiet enough to run the coach? Its GPU work slows perception and d1, so no job starts while the policy
    is defending or saving for a threat (`mode`), or while a tactics push line says "tower in X s" with X <= eta_s."""
    if mode in COACH_BUSY_MODES:
        return False
    for line in tactics_lines or []:
        m = _TOWER_ETA_RE.search(line)
        if m and float(m.group(1)) <= eta_s:
            return False
    return True


@dataclass
class CoachPlan:
    """A coach plan as the loop reads it: its text, the game time it was made for, and the model's wall time."""
    text: str
    t: float
    latency_s: float


class CoachWorker:
    """The slow loop's background thread (D-046): `planner.update` runs one job at a time, so a plan never delays a
    decision. `submit` is non-blocking: it does nothing while a job runs or before the next one is due (the first
    time, `period_s` game seconds after the last job, or a new opponent card) and, given the policy `mode` and the
    tactics lines, only when it is quiet (`coach_quiet`): a due job waits for the next offer. A finished plan is
    published as
    `latest_plan` and appended to <out_dir>/plans.jsonl; an exception is logged and the old plan stays. Every model
    call, the pre-game plan included, runs on this one thread (MLX streams are per thread). The planner is a
    `CoachPlanner` (or anything with `pregame` and `update(plan, state, t, tactics_lines, lookup, context_text)`);
    `lookup_fn(state, t, ctx) -> list[str]`, run on the worker thread, supplies the lookup summary when `submit` is
    given none."""

    def __init__(self, planner, period_s: float = COACH_PERIOD_S, out_dir: Path | None = None, lookup_fn=None,
                 clock=time.perf_counter):
        self.planner = planner
        self.period_s = period_s
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.lookup_fn = lookup_fn
        self.clock = clock
        self.pregame_plan: Plan | None = None
        self.pregame_s: float | None = None
        self.latest_plan: CoachPlan | None = None
        self.jobs = 0              # plans published
        self.errors = 0            # jobs (or the pregame) that raised
        self.empty = 0             # jobs whose model wrote nothing usable
        self.latencies: list[float] = []
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False
        self._last_t: float | None = None      # game time of the last accepted job
        self._seen: set[str] = set()           # opponent cards the last accepted job knew
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    # -- the worker thread ----------------------------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, fut = item
            try:
                fut.set_result(fn())
            except Exception as exc:        # the caller (pregame) or the job itself decides what a failure means
                fut.set_exception(exc)

    def _enqueue(self, fn) -> Future:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="coach", daemon=True)
            self._thread.start()
        fut: Future = Future()
        self._queue.put((fn, fut))
        return fut

    # -- pre-game -------------------------------------------------------------------------------------------------
    def pregame(self, own_deck: list[str], deck_key: str | None = None, ctx=None) -> Plan:
        """The pre-game plan, made on the worker thread; blocks (the match has not started). A failure keeps the deck."""
        t0 = self.clock()
        try:
            plan = self._enqueue(lambda: self._pregame(own_deck, deck_key, ctx)).result()
        except Exception as exc:
            log.warning("coach pregame failed: %r", exc)
            self.errors += 1
            plan = Plan("Deck: " + ", ".join(own_deck), "rules_fallback")
        self.pregame_s = round(self.clock() - t0, 2)
        self.pregame_plan = plan
        return plan

    def _pregame(self, own_deck: list[str], deck_key: str | None, ctx) -> Plan:
        plan = self.planner.pregame(own_deck, deck_key, ctx)
        prime = getattr(self.planner, "prime", None)          # prefill the static prompt prefix: the first update is quick
        if callable(prime):
            try:
                prime(own_deck)
            except Exception as exc:
                log.warning("coach prime failed: %r", exc)
        return plan

    # -- in game --------------------------------------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._busy

    def attach(self, out_dir: Path) -> None:
        """Write plans.jsonl into `out_dir` (a fresh file)."""
        self.out_dir = Path(out_dir)
        (self.out_dir / "plans.jsonl").write_text("")

    def _due(self, state: dict, t: float) -> bool:
        known = (state.get("opponent") or {}).get("deck_known") or []
        return not self._busy and not self._closed and should_update(self._last_t, t, known, self._seen, self.period_s)

    def is_due(self, state: dict, t: float) -> bool:
        """Would `submit` take a job now? Cheap: lets the loop skip building the job's inputs."""
        with self._lock:
            return self._due(state, t)

    def submit(self, state: dict, ctx: str | None, tactics_lines: list[str] | None, lookup: list[str] | None,
               t: float, mode: str | None = None) -> bool:
        """Offer a job; True when it was accepted. Never blocks on the model. `mode` is the policy's decision mode."""
        with self._lock:
            if not self._due(state, t) or not coach_quiet(mode, tactics_lines):
                return False
            job = (copy.deepcopy(state), ctx, list(tactics_lines or []), None if lookup is None else list(lookup), t)
            self._busy = True
            self._last_t = t
            self._seen.update((state.get("opponent") or {}).get("deck_known") or [])
        self._enqueue(lambda: self._job(*job))
        return True

    def _current(self) -> Plan:
        if self.latest_plan is not None:
            return Plan(self.latest_plan.text, "coach")
        return self.pregame_plan or Plan("", "rules")

    def _job(self, state: dict, ctx: str | None, tactics_lines: list[str], lookup: list[str] | None, t: float) -> None:
        try:
            if self._closed:
                return
            t0 = self.clock()
            if lookup is None and self.lookup_fn is not None:
                try:
                    lookup = self.lookup_fn(state, t, ctx)
                except Exception as exc:        # no lookup is a worse prompt, not a failed job
                    log.warning("coach lookup failed: %r", exc)
                    lookup = []
            current = self._current()
            new = self.planner.update(current, state, t, tactics_lines=tactics_lines, lookup=lookup, context_text=ctx)
            latency = round(self.clock() - t0, 2)
            if new is current:
                self.empty += 1
                return
            self.latest_plan = CoachPlan(new.text, t, latency)
            self.jobs += 1
            self.latencies.append(latency)
            self._append({"t": t, "plan": new.text, "latency_s": latency})
        except Exception as exc:
            log.warning("coach job at t=%s failed: %r", t, exc)
            self.errors += 1
        finally:
            with self._lock:
                self._busy = False

    def _append(self, row: dict) -> None:
        if self.out_dir is None:
            return
        with open(self.out_dir / "plans.jsonl", "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self, timeout: float = 60.0) -> None:
        """Stop the thread: a job already running finishes (killing a GPU job mid-way is worse); a queued one is dropped."""
        with self._lock:
            self._closed = True
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout)

    def summary(self) -> dict:
        return {"pregame_s": self.pregame_s, "plans": self.jobs, "errors": self.errors, "empty": self.empty,
                "latency_s": _pcts(self.latencies), "period_s": self.period_s}


class LiveLoop:
    def __init__(self, source, perception, builder, generator, ranker, executor, cards: dict, prior: dict,
                 plan_text: str | None = None, out_dir: Path | None = None, filters: dict | None = None,
                 clock=time.perf_counter, min_gap_s: float = 1.0, policy=None, require_hand: bool = True,
                 tactics=None, arbiter=None, coach=None, shots=None, kg2=None):
        self.source = source
        self.perception = perception
        self.builder = builder
        self.generator = generator
        self.ranker = ranker
        self.executor = executor
        self.cards = cards
        self.prior = prior
        self.plan_text = plan_text
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.filters = dict(filters or {})
        self.clock = clock
        self.min_gap_s = min_gap_s
        # None / "reactive" -> ReactivePolicy(cards); "trigger" -> the old should_decide path (self.policy stays None)
        if policy is None or policy == "reactive":
            policy = ReactivePolicy(cards, kg=kg2)
        self.policy = None if policy == "trigger" else policy
        self.kg2 = kg2                               # a KG2 (MAC-AK K4b) or None: the policy, the engine and the fact lines use it
        if kg2 is not None:
            attach = getattr(getattr(self.policy, "rules", self.policy), "attach_kg", None)
            if callable(attach) and getattr(getattr(self.policy, "rules", self.policy), "kg", None) is None:
                attach(kg2)
            if getattr(tactics, "kg2", False) is None:
                tactics.kg2 = kg2
        self.require_hand = require_hand             # never tap a card the hand read does not show (FIX_LIST #27)
        self.tactics = tactics                       # a TacticalEngine (MAC-Z3) or None = lookup candidates only
        self.arbiter = arbiter                       # a RuleArbiter (MAC-AB A4) narrowing d1's options when unsure, or None
        self.coach = coach                           # a CoachWorker (D-046): its latest plan replaces `plan_text`
        self.shots = shots                           # a ShotRecorder (MAC-AF) or None = no screenshots
        self._shots_ms: list[float] = []             # per frame: time the loop's thread spent in the recorder's hooks
        self._coach_tick_t = float("-inf")           # game time of the last offer to the coach
        self.hand_unknown_holds = 0                  # decisions held because no candidate card was in the read hand
        self.policy_holds: Counter = Counter()       # reactive hold decisions by mode (one per evaluated frame)
        self.last_decision_t: float | None = None
        self._ability_t = float("-inf")              # game time of the last ability tap
        self.ability_frames = 0                      # frames whose state showed the hero ability button
        self.ability_offered = 0                     # decisions that had the ability among the ranker's options
        self.ability_taps = 0                        # decisions that chose it
        self._handled_opp_t = float("-inf")       # newest opponent play already seen at a decision
        self.skipped_min_gap = 0
        self._rect_changes_seen = 0
        self.frames: list[dict] = []
        self.decisions: list[LiveDecision] = []
        self._fh: dict[str, object] = {}
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            for name in ("frames.jsonl", "decisions.jsonl"):
                (self.out_dir / name).write_text("")
            if self.coach is not None:
                self.coach.attach(self.out_dir)

    def _write(self, name: str, obj: dict) -> None:
        if self.out_dir is None:
            return
        fh = self._fh.get(name)
        if fh is None or fh.closed:
            fh = self._fh[name] = open(self.out_dir / name, "a")
        fh.write(json.dumps(obj, default=_json_default) + "\n")
        fh.flush()

    def _close_files(self) -> None:
        for fh in self._fh.values():
            fh.close()

    def _follow_rect(self) -> bool:
        """The capture source follows a moved window (`rect_changes`); tell the executor so its screen map follows too."""
        rc = getattr(self.source, "rect_changes", 0)
        if rc <= self._rect_changes_seen:
            return False
        self._rect_changes_seen = rc
        update_rect = getattr(self.executor, "update_rect", None)
        if callable(update_rect):
            update_rect(self.source.rect)
        return True

    def _fresh_opponent_state(self, state: dict) -> dict:
        """Copy of `state` whose opponent `recent_plays` drops plays a decision already saw (each play triggers once)."""
        opp = state.get("opponent") or {}
        plays = [p for p in opp.get("recent_plays", []) if p.get("t") is None or p["t"] > self._handled_opp_t]
        return {**state, "opponent": {**opp, "recent_plays": plays}}

    def _coach_tick(self, state: dict, ctx: str, t: float, mode: str, report=None) -> None:
        """Offer the coach worker a job (at most once per game second, and only when one is due); never blocks. The
        worker takes it only when quiet: the policy `mode` is not defend / save and no push is close to a tower."""
        if self.coach is None or t - self._coach_tick_t < COACH_TICK_S:
            return
        self._coach_tick_t = t
        if not self.coach.is_due(state, t):
            return
        if report is None and self.tactics is not None:
            report = self.tactics.analyse(state, t)
        self.coach.submit(state, ctx, list(report.lines) if report is not None else [], None, t, mode=mode)

    def _kg_lines(self, state: dict) -> list[str]:
        """Up to 3 `KG:` lines (the pro answers to the threats on the board, the opponent cards still to come) for the
        ranker's context; [] without a graph or when it fails."""
        return kg_fact_lines(self.kg2, state)               # (the dataset builder renders them through the same helper)

    def _plan_at(self, t: float) -> tuple[str | None, float | None, float | None]:
        """(text, plan_t, plan_age_s) of the plan the ranker reads now: the coach's latest, else the pregame plan."""
        plan = self.coach.latest_plan if self.coach is not None else None
        if plan is None:
            return self.plan_text, None, None
        return plan.text, plan.t, round(t - plan.t, 2)

    def step(self, frame, t: float, idx: int, capture_ms: float) -> LiveDecision | None:
        clock = self.clock
        rect_changed = self._follow_rect()
        a = clock()
        gs = self.perception.process(frame, t, idx)
        events = self.perception.drain_events()
        b = clock()
        self.builder.observe_events(events)
        built = self.builder.build(gs)
        c = clock()
        perception_ms = (b - a) * 1000.0
        live_state_ms = (c - b) * 1000.0
        row = {"t": t, "idx": idx, "readiness": getattr(gs, "readiness", None), "capture": capture_ms,
               "perception": perception_ms, "live_state": live_state_ms, "rect_changed": rect_changed}
        self.frames.append(row)
        self._write("frames.jsonl", row)
        shots_ms = 0.0
        if self.shots is not None:                   # only queues work for the recorder's thread
            s0 = clock()
            self.shots.on_frame(frame, t, idx, built[0] if built is not None else None, gs)
            shots_ms = (clock() - s0) * 1000.0
            self._shots_ms.append(shots_ms)
        if built is None:
            return None

        state, ctx = built
        if (state["own"].get("hero_ability") or {}).get("visible"):
            self.ability_frames += 1
        if t - self._ability_t < ABILITY_MIN_GAP_S:
            state = ability_cooling(state)           # one tap per ABILITY_MIN_GAP_S (D-049): neither the engine nor the policy asks for it
        d = clock()
        if self.last_decision_t is not None and t - self.last_decision_t < self.min_gap_s:
            self.skipped_min_gap += 1           # `should_decide` stays true for seconds; never tap twice this close
            return None
        if self.policy is None:
            go = should_decide(self._fresh_opponent_state(state), t, self.last_decision_t, self.cards)
            verdict = ActDecision(go, "trigger", None, "should_decide")
        else:
            verdict = self.policy.decide(state, t, context_text=ctx)
        e = clock()
        trigger_ms = (e - d) * 1000.0
        if not verdict.act:
            if self.policy is not None:
                self.policy_holds[verdict.mode] += 1
            self._coach_tick(state, ctx, t, verdict.mode)
            return None

        report = self.tactics.analyse(state, t) if self.tactics is not None else None
        self._coach_tick(state, ctx, t, verdict.mode, report)
        plan_text, plan_t, plan_age_s = self._plan_at(t)
        kg_lines = self._kg_lines(state)
        rank_ctx = rank_context(ctx, report, kg_lines)    # ctx, the tactics lines, then the graph's facts (the act policy keeps `ctx`)
        extra = dict(self.filters)
        ability = None
        if t - self._ability_t >= ABILITY_MIN_GAP_S:
            engine_move = next(iter(getattr(report, "abilities", None) or []), None)
            if engine_move is not None and ABILITY_KEY in (getattr(verdict, "allowed_cards", None) or ()):
                # D-049: the ability is an option only when the engine proposed it (a simulated saving of 1,000+ tower HP)
                # AND the policy listed it among the allowed cards; the note says what it costs
                reason = engine_move.reason.removesuffix(f", {ABILITY_COST} elixir")
                ability = AbilityMove(engine_move.key, list(engine_move.tile),
                                      f"hero ability: {reason}; {ABILITY_USE_RARELY}", engine_move.score)
        if ability is not None:
            extra["ability_moves"] = [ability]
        if report is None:
            cands = generated = self.generator.generate(state, t, ctx, n=4, **extra)
        else:
            cands = generated = self.generator.generate(state, t, ctx, n=4, report=report, **extra)
        ability_cands = [x for x in generated if is_ability_key(x.key)]
        cands = [x for x in generated if not is_ability_key(x.key)]      # the policy and the hand check judge cards only
        if self.policy is not None:
            cands = self.policy.filter_candidates(cands, verdict, state)
        # the ability is offered only when the engine proposed it AND the policy listed it among the allowed cards (D-049),
        # so it may be the only option; a moment with nothing else to play and no such offer never turns into an ability tap
        ability_alone = bool(ability_cands)
        policy_hold = self.policy is not None and not cands and not ability_alone   # wants to act, but nothing fits
        hand_hold = False
        if self.require_hand and cands:
            in_hand = {c for c in state["own"]["hand"] if c}
            cands = [x for x in cands if x.key.split("|")[0] in in_hand]
            hand_hold = not cands and not ability_alone          # every candidate is a card we cannot see in the hand
        if ability_cands and (cands or ability_alone):
            # the ability is one more option for the ranker (it has no card to be in the hand)
            order = {x.key: j for j, x in enumerate(generated)}
            cands = sorted(cands + ability_cands, key=lambda x: order.get(x.key, len(order)))
            self.ability_offered += 1
        f = clock()
        candidates_ms = (f - e) * 1000.0
        chosen = None
        arbiter_reason = arbiter_subset = None
        if not cands:
            key, probs, rank_ms, g = HOLD_KEY, [], 0.0, f
        else:
            if len(cands) == 1:
                i, probs = 0, [1.0]
            elif hasattr(self.ranker, "choose_candidates"):         # e.g. the tactics ranker: reads the candidates
                res = self.ranker.choose_candidates(cands)
                i, probs = res.index, list(res.probs)
            else:
                notes = {x.key: x.note for x in cands if getattr(x, "note", None)}
                i, probs = rank_keys(self.ranker, rank_ctx, [x.key for x in cands], plan_text,
                                     notes=notes or None)
                if self.arbiter is not None:        # d1 unsure: it chooses among the tactical / pro-supported subset
                    urgent = urgent_from_report(report, self.arbiter.cfg.urgent_eta_s)
                    narrowed = self.arbiter.decide(cands, probs, urgent)
                    i, arbiter_reason = narrowed.index, narrowed.reason
                    arbiter_subset = [{"key": cands[j].key, "p": round(p, 4)}
                                      for j, p in zip(narrowed.subset, narrowed.probs)]
            chosen = cands[i]
            key = chosen.key
            g = clock()
            rank_ms = (g - f) * 1000.0

        elixir = state["own"]["elixir"]
        tile_source = None
        if policy_hold:
            # the policy wanted to play but no candidate fits it: record a hold, never call the executor
            key, tile = HOLD_KEY, None
            entry = {"t": t, "key": HOLD_KEY, "tile": None, "status": "hold_policy", "clicks": [], "sent": False}
            cands = generated          # logged unfiltered: the record shows what the policy turned down
        elif hand_hold:
            # the hand is unread (greyed cards, a lost slot): tapping would only be skipped, so hold instead
            key, tile = HOLD_KEY, None
            entry = {"t": t, "key": HOLD_KEY, "tile": None, "status": "hold_hand_unknown", "clicks": [],
                     "sent": False}
            cands = generated
            self.hand_unknown_holds += 1
        elif is_ability_key(key):
            # the hero ability: one tap on its button (no card slot, no hand check); the 3 elixir enter the accounting
            tile = None if getattr(chosen, "tile", None) is None else list(chosen.tile)    # where the turret lands
            tile_source = "ability"
            self.ability_taps += 1
            side = (state["own"].get("hero_ability") or {}).get("side") or "right"
            tap = getattr(self.executor, "tap_ability", None)
            entry = tap(side, t) if callable(tap) else {"t": t, "key": ABILITY_KEY, "tile": None, "status": "no_ability_tap",
                                                         "clicks": [], "sent": False}
            note_play = getattr(self.builder, "note_own_play", None)
            if entry.get("sent") and callable(note_play):
                note_play(ABILITY_KEY, None, t, cost=ABILITY_COST)
            if entry.get("sent") or entry.get("status") == "planned_ability":
                self._ability_t = t
        elif (cands and "fallback" in cands[i].sources and (elixir is None or elixir < FALLBACK_MIN_ELIXIR)
              and not fallback_allowed(verdict, key.split("|")[0], _roles_of(self.tactics))):
            # a filler candidate (no knn/kg support) is only worth playing from a full bar; otherwise wait
            key, tile = HOLD_KEY, None
            entry = {"t": t, "key": HOLD_KEY, "tile": None, "status": "hold_fallback", "clicks": [], "sent": False}
        else:
            # tile precedence: a spell's aim, the engine's tile for the card against this push (it places what the
            # model picked), the candidate's own tile, the pro-median prior
            aim = getattr(self.policy, "target_tile", None)
            tile = aim(key, state) if callable(aim) else None
            tile_source = "spell" if tile is not None else None
            if tile is None:
                card = key.split("|")[0]
                engine_tile = (getattr(verdict, "engine_tiles", None) or {}).get(card)
                if engine_tile is not None:
                    tile = list(engine_tile)
                    pro = (getattr(verdict, "tile_sources", None) or {}).get(card) == "pro_offset"
                    tile_source = "pro_offset" if pro else "engine"
            if tile is None and getattr(chosen, "tile", None) is not None:
                tile, tile_source = list(chosen.tile), "candidate"  # a tactical candidate carries its own tile
                if any(m.key == key and getattr(m, "tile_source", None) == "pro_offset"
                       for m in getattr(report, "moves", None) or []):
                    tile_source = "pro_offset"
            if tile is None:
                tile, tile_source = placement_tile(key, self.prior), "prior"
            entry = self.executor.execute(key, tile, t, hand=state["own"]["hand"])
            note_play = getattr(self.builder, "note_own_play", None)
            if entry.get("sent") and entry.get("slot") is not None and callable(note_play):
                note_play(key.split("|")[0], entry["slot"], t, tile=tile)    # the hand memory follows confirmed taps only
        h = clock()
        execute_ms = (h - g) * 1000.0

        timings = {"capture": capture_ms, "perception": perception_ms, "live_state": live_state_ms,
                   "trigger": trigger_ms, "candidates": candidates_ms, "rank": rank_ms, "execute": execute_ms}
        timings["total"] = sum(timings.values())
        self.last_decision_t = t
        opp_ts = [p["t"] for p in state.get("opponent", {}).get("recent_plays", []) if p.get("t") is not None]
        if opp_ts:
            self._handled_opp_t = max(self._handled_opp_t, max(opp_ts))
        decision = LiveDecision(
            t=t, idx=idx, key=key, tile=tile,
            candidates=[{"key": x.key, "score": x.score, "sources": x.sources, "tile": getattr(x, "tile", None),
                         "note": getattr(x, "note", None)} for x in cands],
            probs=list(probs), executor=entry, timings_ms=timings, mode=verdict.mode, reason=verdict.reason,
            tactics_lines=list(report.lines) if report is not None else [], arbiter_reason=arbiter_reason,
            arbiter_subset=arbiter_subset, plan_t=plan_t, plan_age_s=plan_age_s, tile_source=tile_source,
            kg_lines=list(kg_lines))
        self.decisions.append(decision)
        if self.shots is not None:
            s0 = clock()
            self.shots.on_decision(frame, decision, state, self.executor, gs)
            hook_ms = (clock() - s0) * 1000.0
            self._shots_ms[-1] += hook_ms
            timings["shots"] = shots_ms + hook_ms        # the recorder's cost counts in the decision's total
            timings["total"] += timings["shots"]
        self._write("decisions.jsonl", {**asdict(decision), "elixir": state["own"]["elixir"],
                                        "hand": state["own"]["hand"], "context_text": ctx})
        return decision

    def run(self, max_seconds: float | None = None, max_frames: int | None = None,
            max_decisions: int | None = None) -> dict:
        n_frames = 0
        t_first: float | None = None
        it = iter(self.source.frames())
        try:
            while True:
                if max_frames is not None and n_frames >= max_frames:
                    break
                if max_decisions is not None and len(self.decisions) >= max_decisions:
                    break
                g0 = self.clock()
                try:
                    frame, t, idx = next(it)
                except StopIteration:
                    break
                measured_ms = (self.clock() - g0) * 1000.0
                grab_ms = getattr(self.source, "last_grab_ms", None)
                capture_ms = grab_ms if grab_ms is not None else measured_ms
                if t_first is None:
                    t_first = t
                if max_seconds is not None and t - t_first >= max_seconds:
                    break
                n_frames += 1
                self.step(frame, t, idx, capture_ms)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                close_it = getattr(it, "close", None)
                if close_it is not None:
                    close_it()
                self.source.close()
            finally:
                try:
                    close_perception = getattr(self.perception, "close", None)      # drains the async OCR worker
                    if callable(close_perception):
                        close_perception()
                finally:
                    try:
                        if self.coach is not None:
                            self.coach.close()               # lets a running plan finish: no GPU job is cut off
                    finally:
                        try:
                            if self.shots is not None:
                                self.shots.close()           # writes the index rows still waiting for a frame
                        finally:
                            self._close_files()
        summary = self.summary()
        if self.out_dir is not None:
            (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
        return summary

    def summary(self) -> dict:
        readiness = Counter("none" if r["readiness"] is None else str(r["readiness"]) for r in self.frames)
        timings = [d.timings_ms for d in self.decisions]
        total_p95 = _pcts([tm["total"] for tm in timings])["p95"]
        stats = getattr(self.source, "stats", None)
        timing_report = getattr(self.perception, "timing_report", None)
        return {
            "frames": len(self.frames),
            "decisions": len(self.decisions),
            "readiness_counts": dict(readiness),
            "frame_ms": {s: _pcts([r[s] for r in self.frames if r[s] is not None]) for s in FRAME_STAGES},
            "decision_ms": {s: _pcts([tm[s] for tm in timings]) for s in STAGES},
            "target_p95_ms": TARGET_P95_MS,
            "target_met": None if total_p95 is None else bool(total_p95 < TARGET_P95_MS),
            "decisions_by_status": dict(Counter(d.executor.get("status", "dry_run") for d in self.decisions)),
            "decisions_by_mode": dict(Counter(d.mode for d in self.decisions)),
            "decisions_by_source": dict(Counter(decision_source(d) for d in self.decisions)),
            "policy_holds": dict(self.policy_holds),
            "hand_unknown_holds": self.hand_unknown_holds,
            "rect_updates": sum(1 for r in self.frames if r.get("rect_changed")),
            "skipped_min_gap": self.skipped_min_gap,
            "source_stats": stats() if callable(stats) else None,
            "perception_timing": timing_report() if callable(timing_report) else None,
            **({"hero_ability": {"button_frames": self.ability_frames, "offered": self.ability_offered,
                                 "taps": self.ability_taps}} if self.ability_frames else {}),
            **({"coach": self.coach.summary()} if self.coach is not None else {}),
            **({"shots": {**self.shots.summary(), "hook_ms": _pcts(self._shots_ms)}} if self.shots is not None else {}),
        }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python3 -m clashbot.live", description=TOS_NOTE)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="log the clicks that would be made (default)")
    mode.add_argument("--live", action="store_true",
                      help="send real taps (needs CLASHBOT_ALLOW_LIVE=1; Training Camp / friendly battles only)")
    ap.set_defaults(dry_run=True)
    ap.add_argument("--calib", required=True, help="calibration JSON (tools/calibrate.py)")
    ap.add_argument("--deck", required=True, help="own deck: 8 comma-separated card slugs")
    ap.add_argument("--hero-cards", default="", help="comma-separated slugs of hero-variant cards in the deck")
    ap.add_argument("--deck-key", default=None)
    ap.add_argument("--source", choices=("mirroring", "video"), default="mirroring")
    ap.add_argument("--video", default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--rect", default=None, help="manual window rect in points: l,t,w,h")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--detector", choices=("katacr", "none"), default="katacr")
    ap.add_argument("--detect-every", type=int, default=3)
    ap.add_argument("--sync-ocr", action="store_true", help="run OCR inline instead of on the background worker")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", choices=("cpu", "mps"), default=None)
    ap.add_argument("--ranker", choices=("score", "d1", "tactics"), default="score",
                    help="tactics = the tactics engine's own pick (needs the engine, i.e. no --no-tactics)")
    ap.add_argument("--no-tactics", action="store_true",
                    help="no tactics engine: lookup candidates only, no notes (the pre-MAC-Z3 behaviour)")
    ap.add_argument("--no-kg2", action="store_true",
                    help="do not load db/kg2.sqlite (MAC-AK K4b): no pro answers / offsets, air reserve or KG: fact lines")
    ap.add_argument("--d1-backend", choices=("mlx", "transformers"), default="mlx")
    ap.add_argument("--d1-model", default=None, help="d1 model path / repo id (e.g. a fine-tuned checkpoint)")
    ap.add_argument("--planner", choices=("rules", "llm", "coach", "none"), default="rules",
                    help="coach = the Qwen3-14B coach: a pregame plan, then a short plan every --coach-period game "
                         "seconds from a background thread (D-046)")
    ap.add_argument("--coach-model", default=COACH_MODEL_ID, help="mlx-lm model for --planner coach")
    ap.add_argument("--coach-period", type=float, default=COACH_PERIOD_S,
                    help="game seconds between coach plans (a new opponent card also triggers one); a plan starts "
                         "only in quiet time: not while defending / saving or with a push <= 10 s from a tower")
    ap.add_argument("--coach-prefill-step", type=int, default=COACH_PREFILL_STEP,
                    help="coach prompt tokens per prefill chunk (smaller = shorter GPU bursts for perception)")
    ap.add_argument("--coach-yield-ms", type=float, default=COACH_YIELD_S * 1000,
                    help="coach sleep after every prefill chunk, ms (lets perception's GPU work in)")
    ap.add_argument("--coach-token-yield-ms", type=float, default=COACH_TOKEN_YIELD_S * 1000,
                    help="coach sleep after every generated token, ms")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--max-decisions", type=int, default=None)
    ap.add_argument("--min-gap", type=float, default=1.0, help="minimum seconds between two decisions (0 = off)")
    ap.add_argument("--policy", choices=("reactive", "learned", "fast", "trigger"), default="reactive",
                    help="when to play: reactive = defend / leak / punish, else hold (D-041); learned = the rules' "
                         "defend / leak plus a d1 act-head (needs --act-model); fast = the same with the LightGBM "
                         "act model of tools/train/train_act_gbdt.py (--act-model DIR, threshold from its "
                         "meta.json); trigger = every 3 s etc.")
    ap.add_argument("--act-model", default=None, help="d1 act-head checkpoint for --policy learned")
    ap.add_argument("--act-threshold", type=float, default=None,
                    help="p(act) needed to play (default: `threshold` of <act-model>/act_threshold.json, else 0.5)")
    ap.add_argument("--act-backend", choices=("mlx", "transformers"), default="transformers",
                    help="backend of the act-head model")
    ap.add_argument("--out", default=None, help="output dir (default runs/poc/<timestamp>)")
    ap.add_argument("--record-shots", action="store_true",
                    help="save <out>/shots/: an annotated screenshot per decision, the frames 0.5 s and 1.5 s after a "
                         "tap and index.jsonl (MAC-AF; review with tools/eval/review_shots.py)")
    ap.add_argument("--shots-max", type=int, default=400,
                    help="stop recording shots after this many decisions (about 60 KB per image)")
    ap.add_argument("--arbiter", choices=("none", "rules"), default="none",
                    help="rules = when d1 is unsure it chooses among the tactical / pro-supported candidates only "
                         "(MAC-AB A4, D-045: d1 still decides); not used with --ranker tactics")
    ap.add_argument("--arbiter-conf", type=float, default=ArbiterConfig.conf,
                    help="d1 top probability at which d1 chooses among all candidates")
    ap.add_argument("--arbiter-knn", type=float, default=ArbiterConfig.knn_min,
                    help="kNN support at which a candidate counts as pro-supported")
    ap.add_argument("--arbiter-eta", type=float, default=ArbiterConfig.urgent_eta_s,
                    help="a push this close (seconds) counts as urgent")
    return ap


def _split(text: str | None) -> list[str]:
    return [p.strip() for p in (text or "").split(",") if p.strip()]


def _build_source(args):
    if args.source == "video":
        from cr_perception.sources import VideoFrameSource
        return VideoFrameSource(args.video, start=args.start, end=args.end, stride=args.stride)
    from cr_perception.sources import MirroringSource, WindowRect
    rect = None
    if args.rect:
        left, top, width, height = (int(round(float(p))) for p in _split(args.rect))
        rect = WindowRect(left, top, width, height, "manual")
    return MirroringSource(rect=rect, target_fps=args.fps)


def _build_perception(args, source):
    from cr_perception import Perception
    detector = None
    if args.detector == "katacr":
        from cr_perception.detect import default_katacr_detector
        detector = default_katacr_detector(imgsz=args.imgsz, device=args.device)
    # the deck limits what the hand and unit readers may report (MAC-PF); the opponent deck admits cards on evidence
    return Perception(config=args.calib, source=source, detector=detector, detect_every=args.detect_every,
                      async_ocr=not args.sync_ocr, own_deck=_split(args.deck), opp_evidence=True)


def _build_ranker(args):
    if args.ranker == "d1":
        from clashbot.ranker import D1Ranker
        return D1Ranker(model_path=args.d1_model, backend=args.d1_backend)
    if args.ranker == "tactics":
        from clashbot.ranker import TacticsRanker
        return TacticsRanker()
    from clashbot.ranker import ScoreRanker
    return ScoreRanker()


def resolve_act_threshold(args) -> float:
    """`--act-threshold`, else `threshold` of <act-model>/act_threshold.json (written by eval_act_head), else 0.5."""
    if args.act_threshold is not None:
        return float(args.act_threshold)
    path = Path(args.act_model) / "act_threshold.json"
    if path.is_file():
        return float(json.loads(path.read_text())["threshold"])
    return 0.5


def _build_policy(args, cards: dict):
    """The policy argument of `LiveLoop`: a name (reactive / trigger), a `LearnedActPolicy` on the act-head model or a
    `FastActPolicy` on the LightGBM act model (its threshold: `--act-threshold`, else meta.json; recorded in `args`)."""
    if args.policy == "fast":
        from clashbot.act_model import ActGBDT, FastActPolicy
        model = ActGBDT.load(args.act_model)
        if args.act_threshold is None:
            args.act_threshold = model.threshold
        return FastActPolicy(model, cards, args.act_threshold)
    if args.policy != "learned":
        return args.policy
    from clashbot.policy import LearnedActPolicy
    from clashbot.ranker import D1Ranker
    scorer = D1Ranker(model_path=args.act_model, backend=args.act_backend)
    return LearnedActPolicy(scorer, cards, args.act_threshold)


def _load_kg2(args):
    """The knowledge graph v2 (`db/kg2.sqlite`), or None with `--no-kg2` or when it is missing / unreadable."""
    if getattr(args, "no_kg2", False):
        return None
    try:
        from clashbot.kg2 import KG2
        return KG2()
    except Exception as exc:                                 # noqa: BLE001  the bot runs without the graph
        print(f"note: knowledge graph v2 not loaded ({exc}); running without it", file=sys.stderr)
        return None


def _build_plan_text(args, deck: list[str]) -> str | None:
    if args.planner == "rules":
        from clashbot.planner import RuleBasedPlanner
        return RuleBasedPlanner().pregame(deck, args.deck_key).text
    if args.planner == "llm":
        from clashbot.planner import LLMPlanner
        return LLMPlanner().pregame(deck, args.deck_key).text
    if args.planner == "coach":         # `main` builds a CoachWorker instead (`_build_coach`); this is the one-shot form
        from clashbot.planner import CoachPlanner
        return CoachPlanner(model_path=args.coach_model, prefill_step=args.coach_prefill_step,
                            yield_s=args.coach_yield_ms / 1000.0,
                            token_yield_s=args.coach_token_yield_ms / 1000.0).pregame(deck, args.deck_key).text
    return None


def _build_coach(args, deck: list[str], index=None, kg=None) -> CoachWorker:
    """The coach worker with its pre-game plan made (tools get the shared index and KG). The model loads on the worker
    thread; a failing load or pregame leaves the worker on the bare-deck plan and every later job logs its error."""
    from clashbot.planner import CoachPlanner
    from clashbot.tools import ToolContext
    planner = CoachPlanner(model_path=args.coach_model, prefill_step=args.coach_prefill_step,
                           yield_s=args.coach_yield_ms / 1000.0, token_yield_s=args.coach_token_yield_ms / 1000.0)
    lookup_fn = None
    if index is not None:
        lookup_fn = lambda state, t, ctx: lookup_summary(index.search(state, t, ctx, k=10))    # noqa: E731
    worker = CoachWorker(planner, period_s=args.coach_period, lookup_fn=lookup_fn)
    worker.pregame(deck, args.deck_key, ToolContext(index=index, kg=kg, db_path=planner.db_path))
    return worker


def format_table(summary: dict) -> str:
    lines = ["| stage | p50 ms | p95 ms |", "|---|---|---|"]
    for stage in STAGES:
        p = summary["decision_ms"][stage]
        lines.append(f"| {stage} | {p['p50']} | {p['p95']} |")
    lines.append(f"target_met (total p95 < {summary['target_p95_ms']:.0f} ms): {summary['target_met']}")
    return "\n".join(lines)


def resolve_device(device: str | None) -> str:
    """`--device` unset -> mps when torch reports it, else cpu (torch imported here so `import clashbot.live` stays light)."""
    if device is not None:
        return device
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(argv: list[str] | None = None) -> int:
    print(TOS_NOTE)
    args = build_parser().parse_args(argv)
    args.dry_run = not args.live
    deck = _split(args.deck)
    if len(deck) != 8:
        print(f"error: --deck needs 8 comma-separated card slugs, got {len(deck)}", file=sys.stderr)
        return 2
    heroes = _split(args.hero_cards)
    if args.source == "video" and not args.video:
        print("error: --source video needs --video PATH", file=sys.stderr)
        return 2
    if args.live and args.source != "mirroring":
        print("error: --live needs --source mirroring", file=sys.stderr)
        return 2
    if args.rect and len(_split(args.rect)) != 4:
        print("error: --rect needs l,t,w,h", file=sys.stderr)
        return 2
    if args.ranker == "tactics" and args.no_tactics:
        print("error: --ranker tactics needs the tactics engine (drop --no-tactics)", file=sys.stderr)
        return 2
    if args.policy == "learned":
        if not args.act_model:
            print("error: --policy learned needs --act-model PATH", file=sys.stderr)
            return 2
        try:
            args.act_threshold = resolve_act_threshold(args)         # recorded in config.json below
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"error: bad act_threshold.json in {args.act_model}: {exc}", file=sys.stderr)
            return 2
    if args.policy == "fast" and not args.act_model:
        print("error: --policy fast needs --act-model DIR (the act-gbdt directory)", file=sys.stderr)
        return 2
    args.device = resolve_device(args.device)         # recorded in config.json below

    source = None
    perception = None
    try:
        source = _build_source(args)
        try:
            if args.source == "mirroring":
                from clashbot.executor import MirroringExecutor
                executor = MirroringExecutor.from_calib(args.calib, source.rect, live=args.live)
            else:
                from clashbot.executor import DryRunExecutor
                executor = DryRunExecutor()
        except PermissionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            source.close()
            return 5
        perception = _build_perception(args, source)
        from clashbot.live_state import LiveConfig, LiveStateBuilder
        builder = LiveStateBuilder(LiveConfig(own_deck=deck, hero_cards=heroes, hand_tracker=True,
                                              opp_evidence=True, strict_units=True))
        cards = load_cards()
        kg = KnowledgeGraph()
        kg2 = _load_kg2(args)
        index = StateIndex()
        generator = CandidateGenerator(index, kg, cards)
        tactics = None if args.no_tactics else build_engine(cards, kg, kg2)
        ranker = _build_ranker(args)
        policy = _build_policy(args, cards)
        coach = _build_coach(args, deck, index, kg) if args.planner == "coach" else None
        plan_text = coach.pregame_plan.text if coach is not None else _build_plan_text(args, deck)
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        close_perception = getattr(perception, "close", None)
        if callable(close_perception):
            close_perception()
        if source is not None:
            source.close()
        return 3

    arbiter = None
    if args.arbiter == "rules" and args.ranker != "tactics":        # the tactics ranker is the engine's own pick
        arbiter = RuleArbiter(ArbiterConfig(conf=args.arbiter_conf, urgent_eta_s=args.arbiter_eta,
                                            knn_min=args.arbiter_knn))
    out_dir = Path(args.out) if args.out else Path("runs") / "poc" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    shots = None
    if args.record_shots:
        from clashbot.recorder_shots import ShotRecorder
        shots = ShotRecorder(out_dir / "shots", calib=args.calib, max_decisions=args.shots_max)
    loop = LiveLoop(source, perception, builder, generator, ranker, executor, cards, load_placement_prior(),
                    plan_text=plan_text, out_dir=out_dir, min_gap_s=args.min_gap, policy=policy, tactics=tactics,
                    arbiter=arbiter, coach=coach, shots=shots, kg2=kg2)
    mode = "LIVE" if args.live else "dry run"
    print(f"{mode}: source={args.source} ranker={args.ranker} planner={args.planner} policy={args.policy} "
          f"tactics={'off' if args.no_tactics else 'on'} kg2={'on' if kg2 is not None else 'off'} "
          f"arbiter={'rules' if arbiter else 'none'} "
          f"shots={'on' if shots else 'off'} out={out_dir}")
    summary = loop.run(max_seconds=args.seconds, max_decisions=args.max_decisions)
    print(f"frames={summary['frames']} decisions={summary['decisions']} "
          f"readiness={summary['readiness_counts']} status={summary['decisions_by_status']}")
    print(format_table(summary))
    if summary.get("coach"):
        print(f"coach: {summary['coach']}")
    if summary.get("shots"):
        print(f"shots: {summary['shots']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
