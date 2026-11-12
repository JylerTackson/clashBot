"""One-command demo launcher (MAC-AN; D-055 removed the opponent gate of D-050): asks the deck, loads, says READY, watches
iPhone Mirroring and plays every battle it sees until Ctrl-C.

    CLASHBOT_ALLOW_LIVE=1 scripts/demo.sh --live        # taps ON (both D-031 switches); without them: watch only
    scripts/demo.sh                                     # watch only: the bot decides and logs, nothing is tapped

1. Prompts: the 8 cards, which are evolutions, which are heroes (names are resolved against the card index, with typo
   tolerance). Everything else is today's live command, fixed (d1-ft-joint-tac + act-gbdt, `--policy fast`, shots on).
2. Load every model before looking for the window, warm up the detector and d1, print READY.
3. Watch loop (`DemoSession`): WAITING_WINDOW -> WATCHING -> CHECKING (a battle started: it is played from its first
   frame; the opponent banner is read on up to 10 frames over 5 s, only to record it) -> PLAYING ("battle vs <banner> -
   playing"). The banner decides nothing: any battle is played once taps are ON, whoever the opponent is. The verdict kept in
   battles.jsonl (TRAINING_CAMP when two reads say "Trainer ...", else NOT_TRAINING_CAMP, UNKNOWN when the battle ended
   first) is a record, not a gate. A battle ends when the screen is not a match for 3 s; a closed window goes back to
   WAITING_WINDOW. Frames are driven through `LiveLoop.step`; its `tap_gate` is open for the whole of a battle and closed
   between battles. D-031 is unchanged: taps need `--live` AND CLASHBOT_ALLOW_LIVE=1, else the launcher says so and
   watches only (the executor then plans the clicks and sends none). Use it in Training Camp or friendly battles.
Session files: runs/demo/<timestamp>/ (frames / decisions / holds .jsonl with a `battle` field, battles.jsonl,
summary.json, config.json, shots/, battles/). `battles/<n>_end_<k>.jpg` are the last (up to) 3 frames of battle n seen
after the screen left "match" (k = 0 oldest ... 2 newest; the last match frame when there are none): the result screen,
for reading win / loss and crowns. They are listed in that battle's `end_frames` in battles.jsonl.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from clashbot.config import HOLD_KEY
from clashbot.executor import LIVE_ENV
from clashbot.game_rules import is_playable, load_cards
from clashbot.live import (LiveLoop, Stack, _build_executor, _build_source, build_parser as live_parser, build_stack,
                           resolve_device)

DEMO_DIR = Path("runs") / "demo"
SETUP_FILE = DEMO_DIR / "last_setup.json"
CALIB = "runs/poc/mirror_calib.json"
# today's live command, fixed (the user's own flags are the deck / evolutions / heroes only)
FIXED_FLAGS = ["--calib", CALIB, "--ranker", "d1", "--d1-backend", "transformers", "--d1-model",
               "db/models/d1-ft-joint-tac", "--policy", "fast", "--act-model", "db/models/act-gbdt", "--record-shots"]

TOS_NOTE = ("With live taps enabled (--live and CLASHBOT_ALLOW_LIVE=1) this plays every battle it sees and does not check the "
            "opponent. Use it in Training Camp or friendly battles: playing ranked matches with a bot violates Supercell's "
            "terms of service and risks a ban.")
WAITING_WINDOW, WATCHING, CHECKING, PLAYING = "WAITING_WINDOW", "WATCHING", "CHECKING", "PLAYING"
WINDOW_RETRY_S = 2.0              # try to open the mirroring window this often
CHECK_FRAMES = 10                 # banner reads per battle ...
CHECK_WINDOW_S = 5.0              # ... within this many seconds of the battle start
CHECK_GAP_S = 0.5                 # ... at most one every this many seconds (OCR runs only while CHECKING)
BATTLE_END_S = 3.0                # the screen is not a match for this long: the battle is over
FRAMES_KEEP = 20000               # `loop.frames` is trimmed to the newest half of this when it grows past it
END_FRAMES = 3                    # frames kept for the result screen of each battle ...
END_JPEG_QUALITY = 85             # ... saved as battles/<n>_end_<k>.jpg


# ----------------------------------------------------------------------------------------------- card names
def _norm(text: str) -> str:
    """Lower case without spaces, dots, hyphens, apostrophes and underscores ("Mini P.E.K.K.A." -> "minipekka")."""
    return re.sub(r"[\s.\-'_’]+", "", str(text).lower())


def _tokens(text: str) -> list[str]:
    return [t for t in (_norm(p) for p in re.split(r"[\s\-_]+", str(text).strip())) if t]


def _prefixes(query: list[str], name: list[str]) -> bool:
    """Every query token is the start of the name token at the same place ("e dragon" -> Electro Dragon)."""
    return 0 < len(query) <= len(name) and all(n.startswith(q) for q, n in zip(query, name))


def resolve_card(text: str, cards: dict, pool: list[str] | None = None) -> tuple[str | None, list[str], bool]:
    """(slug, [], False) for a name that is one card; (None, up to 3 closest slugs, ambiguous?) for one that is not:
    `ambiguous` is True when several cards fit the name equally. Steps: the name or slug without spaces / dots / hyphens /
    apostrophes equals the query; the query's words are the starts of the name's words and only one card fits ("e
    dragon"); `difflib.get_close_matches(cutoff=0.75)` with one clear winner. `pool` limits the cards (the deck, for the
    evolution and hero questions)."""
    slugs = list(pool) if pool is not None else [s for s in cards if is_playable(s, cards)]
    q = _norm(text)
    if not q:
        return None, [], False
    names = {s: {_norm(s), _norm(cards[s].get("name") or s)} for s in slugs}
    exact = [s for s in slugs if q in names[s]]
    if len(exact) == 1:
        return exact[0], [], False
    if exact:
        return None, exact[:3], True
    if len(q) >= 3:
        words = _tokens(text)
        starts = [s for s in slugs if _prefixes(words, _tokens(cards[s].get("name") or s))]
        if len(starts) == 1:
            return starts[0], [], False
        if starts:
            return None, starts[:3], True
    by_key = {k: s for s in slugs for k in names[s]}
    close = difflib.get_close_matches(q, list(by_key), n=6, cutoff=0.75)
    ranked = list(dict.fromkeys(by_key[k] for k in close))
    if len(ranked) == 1:
        return ranked[0], [], False
    if ranked:
        ratios = [difflib.SequenceMatcher(None, q, k).ratio() for k in close]
        if by_key[close[0]] != by_key[close[1]] and ratios[0] - ratios[1] >= 0.05:      # one clear winner
            return by_key[close[0]], [], False
        return None, ranked[:3], True
    loose = difflib.get_close_matches(q, list(by_key), n=6, cutoff=0.5)
    return None, list(dict.fromkeys(by_key[k] for k in loose))[:3], False


class SetupError(ValueError):
    """A deck / evolution / hero answer that cannot be used; the message is what the presenter is told."""


def parse_cards(text: str, cards: dict, pool: list[str] | None = None) -> list[str]:
    """Comma-separated card names -> slugs (in the order given); SetupError for an unknown or ambiguous name or a duplicate."""
    out: list[str] = []
    for part in [p.strip() for p in re.split(r"[,;]", text or "") if p.strip()]:
        slug, near, ambiguous = resolve_card(part, cards, pool)
        if slug is None:
            closest = f" Closest: {', '.join(near)}." if near else ""
            raise SetupError(f"'{part}' " + ("matches several cards." if ambiguous else "is not a card.") + closest)
        if slug in out:
            raise SetupError(f"{slug} is listed twice.")
        out.append(slug)
    return out


def check_deck(deck: list[str]) -> list[str]:
    if len(deck) != 8:
        raise SetupError(f"A deck has 8 cards; got {len(deck)}.")
    return deck


def check_variants(picked: list[str], deck: list[str], cards: dict, key: str, label: str, other: list[str] | None = None,
                   other_label: str = "") -> list[str]:
    """Evolutions (`key` "has_evolution") or heroes ("has_hero_variant"): a subset of the deck whose cards have that
    variant; a card cannot be both an evolution and a hero (`other` is the other list)."""
    for slug in picked:
        if slug not in deck:
            raise SetupError(f"{slug} is not in the deck.")
        if not cards[slug].get(key):
            can = [c for c in deck if cards[c].get(key)]
            raise SetupError(f"{slug} has no {label}. In this deck: {', '.join(can) if can else 'none'}.")
        if other and slug in other:
            raise SetupError(f"{slug} cannot be both {label} and {other_label}.")
    return picked


# ----------------------------------------------------------------------------------------------- prompts
@dataclass
class Setup:
    deck: list[str]
    evo_cards: list[str] = field(default_factory=list)
    hero_cards: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"deck: {', '.join(self.deck)}\n  evolutions: {', '.join(self.evo_cards) or 'none'}"
                f"\n  heroes: {', '.join(self.hero_cards) or 'none'}")


def load_setup(path: Path) -> Setup | None:
    try:
        raw = json.loads(Path(path).read_text())
        return Setup(list(raw["deck"]), list(raw.get("evo_cards") or []), list(raw.get("hero_cards") or []))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_setup(path: Path, setup: Setup) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"deck": setup.deck, "evo_cards": setup.evo_cards, "hero_cards": setup.hero_cards}, indent=2))


def _ask_until(ask, prompt: str, parse, say):
    """Ask until `parse(answer)` stops raising SetupError (the error is printed)."""
    while True:
        try:
            return parse(ask(prompt))
        except SetupError as exc:
            say(f"  {exc}")


def collect_setup(args, cards: dict, ask=input, say=print, setup_file: Path = SETUP_FILE) -> Setup:
    """The deck, evolutions and heroes: from the flags where given, else asked; the last setup can be reused with Enter;
    `--yes` skips the confirmation. SetupError for a bad flag."""
    asked_nothing = args.deck is None and args.evo_cards is None and args.hero_cards is None
    last = load_setup(setup_file) if asked_nothing else None
    if last is not None:
        say("Last setup:\n  " + last.summary())
        if ask("Press Enter to reuse it, or type n for a new one: ").strip().lower() not in ("n", "new", "no"):
            return last
    if args.deck is not None:
        deck = check_deck(parse_cards(args.deck, cards))
    else:
        say("Cards can be typed by name: knight, mini pekka, electro dragon ...")
        deck = _ask_until(ask, "Your 8 cards (comma-separated): ", lambda a: check_deck(parse_cards(a, cards)), say)
    say("  " + ", ".join(deck))
    if args.evo_cards is not None:
        evo = check_variants(parse_cards(args.evo_cards, cards, deck), deck, cards, "has_evolution", "evolution")
    else:
        evo = _ask_until(ask, "Which are evolutions? (Enter for none): ",
                         lambda a: check_variants(parse_cards(a, cards, deck), deck, cards, "has_evolution", "evolution"), say)
    if args.hero_cards is not None:
        hero = check_variants(parse_cards(args.hero_cards, cards, deck), deck, cards, "has_hero_variant", "hero variant",
                              evo, "an evolution")
    else:
        hero = _ask_until(ask, "Which are heroes? (Enter for none): ",
                          lambda a: check_variants(parse_cards(a, cards, deck), deck, cards, "has_hero_variant",
                                                   "hero variant", evo, "an evolution"), say)
    setup = Setup(deck, evo, hero)
    say(setup.summary())
    if not args.yes and ask("Start with this deck? [Y/n] ").strip().lower() in ("n", "no"):
        raise SetupError("cancelled")
    save_setup(setup_file, setup)
    return setup


# ----------------------------------------------------------------------------------------------- taps / gate
def tap_status(live: bool, environ=None) -> tuple[bool, str]:
    """(taps ON?, the reason): ON only with `--live` AND CLASHBOT_ALLOW_LIVE=1 on macOS (D-031; the executor checks again)."""
    env = os.environ if environ is None else environ
    if not live:
        return False, "no --live"
    if sys.platform != "darwin":
        return False, "live taps need macOS + iPhone Mirroring"
    if env.get(LIVE_ENV) != "1":
        return False, f"--live without {LIVE_ENV}=1"
    return True, "live"


D031_MESSAGE = (f"live taps are disabled; the user enables them with --live and {LIVE_ENV}=1 "
                "(for Training Camp or friendly battles). Continuing watch-only.")
TAPS_ON_NOTE = "Taps are ON: it plays every battle it sees, so start only Training Camp or friendly battles."


class TapGate:
    """`LiveLoop.tap_gate`: open for the whole of a battle (any battle, D-055), closed between battles. `reason` says why it is
    closed. Whether a tap is really sent is the executor's business (D-031)."""

    def __init__(self):
        self.open = False
        self.reason = "no battle"

    def __call__(self) -> bool:
        return self.open

    def close(self, reason: str) -> None:
        self.open, self.reason = False, reason

    def allow(self) -> None:
        self.open, self.reason = True, ""


# ----------------------------------------------------------------------------------------------- end-of-battle frames
def write_jpeg(path: Path, frame, quality: int = END_JPEG_QUALITY) -> bool:
    """One frame (BGR / BGRA / grey, full size) as a JPEG; True when the file was written."""
    import cv2
    import numpy as np
    from clashbot.recorder_shots import _to_bgr
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), _to_bgr(np.asarray(frame)), [cv2.IMWRITE_JPEG_QUALITY, quality]))


class EndFrameWriter:
    """Saves each battle's last frames as `<out_dir>/battles/<n>_end_<k>.jpg` on one small daemon thread (started at the first
    battle end), so the loop never waits for the JPEG encoder. `submit` only queues (a full queue drops the job: it returns
    []); a failing write is counted in `errors`, never raised. `writer(path, frame) -> bool` replaces `write_jpeg` (tests).
    `close` waits for the queued jobs."""

    def __init__(self, out_dir: Path, writer=None, queue_size: int = 4, quality: int = END_JPEG_QUALITY):
        self.dir = Path(out_dir) / "battles"
        self.writer = writer or (lambda path, frame: write_jpeg(path, frame, quality))
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.thread: threading.Thread | None = None
        self.written = self.errors = self.dropped = 0

    def submit(self, n: int, frames: list) -> list[str]:
        """Queue battle `n`'s frames (oldest first); the file names (relative to the session dir) that will be written."""
        if not frames:
            return []
        jobs = [(self.dir / f"{n}_end_{k}.jpg", f"battles/{n}_end_{k}.jpg", frame) for k, frame in enumerate(frames)]
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name="end-frames", daemon=True)
            self.thread.start()
        try:
            self.queue.put_nowait(jobs)
        except queue.Full:
            self.dropped += 1
            return []
        return [name for _, name, _ in jobs]

    def _run(self) -> None:
        while True:
            jobs = self.queue.get()
            if jobs is None:
                return
            for path, _, frame in jobs:
                try:
                    ok = self.writer(path, frame)
                except Exception:                  # noqa: BLE001  a frame that cannot be encoded must not stop the thread
                    ok = False
                if ok:
                    self.written += 1
                else:
                    self.errors += 1
            jobs = frame = None                    # frees the frames' references

    def close(self, timeout: float = 10.0) -> None:
        if self.thread is None:
            return
        try:
            self.queue.put(None, timeout=timeout)
        except queue.Full:
            return
        self.thread.join(timeout)
        self.thread = None


# ----------------------------------------------------------------------------------------------- the session
@dataclass
class Battle:
    n: int
    t_start: float
    wall_start: float
    decisions_at: int
    verdict: str = "CHECKING"                       # a record, not a gate: CHECKING, TRAINING_CAMP, NOT_TRAINING_CAMP
    banner: str = ""
    reads: list = field(default_factory=list)       # [(t, text, is_trainer)]
    check_t0: float = 0.0
    last_read_t: float = float("-inf")


class DemoSession:
    """The watch loop's state machine over a `LiveLoop` (see the module docstring). `reader(frame) -> str` reads the
    opponent banner (`trainer_banner.read_banner` on the real OCR) and `is_trainer` labels it, both only for the record: the
    tap gate opens at the start of every battle. Everything printed goes through `say`; `clock` is wall time (battle rows)."""

    def __init__(self, loop: LiveLoop, out_dir: Path, reader, gate: TapGate, taps_on: bool, say=print,
                 executor_factory=None, reset_perception=None, clock=time.time, end_frames: EndFrameWriter | None = None):
        from cr_perception.trainer_banner import is_trainer, verdict
        self.loop, self.out_dir, self.reader, self.gate, self.taps_on = loop, Path(out_dir), reader, gate, taps_on
        self.say, self.clock = say, clock
        self.executor_factory = executor_factory
        self.reset_perception = reset_perception
        self.end_frames = end_frames or EndFrameWriter(self.out_dir)
        self._tail: deque = deque(maxlen=END_FRAMES)       # the last frames seen since the screen left "match" (references)
        self._last_match_frame = None                      # ... and the last match frame, for a battle that ends in a match
        self._is_trainer, self._verdict = is_trainer, verdict
        self.state: str | None = None
        self.stop = False
        self.battle: Battle | None = None
        self.n_battles = 0
        self.totals = {"battles": 0, "training_camp": 0, "not_training_camp": 0, "unknown": 0, "decisions": 0, "taps": 0,
                       "gated": 0, "planned": 0, "window_losses": 0, "frames": 0, "banner_reads": 0, "step_errors": 0}
        self.source = None
        self._cur_t = 0.0
        self._last_ready_t = 0.0
        self._overhead_s = 0.0                 # time spent in `on_frame` outside `loop.step` (state machine, banner OCR)
        self._overhead_watch_s = 0.0           # ... of which in frames that did not read the banner
        self._overhead_frames = 0
        self._errors: set[str] = set()
        self._t_off = 0.0
        self._idx_off = 0
        self._last_t = 0.0
        self._last_idx = 0
        self._seen_errors: set[str] = set()
        self._opened = 0
        loop.tap_gate = gate
        loop.tag_battle = True
        loop.on_match_start = self._on_match_start
        self.started = clock()

    # -- states -----------------------------------------------------------------------------------------------
    def set_state(self, state: str, message: str) -> None:
        if state != self.state:
            self.state = state
            self.say(message)

    # -- the battle ---------------------------------------------------------------------------------------------
    def _on_match_start(self, loop: LiveLoop) -> None:
        """The frame's readiness turned "match" (called inside `step`, before the state is built). A match that returns
        within BATTLE_END_S of the last one is the same battle; else a new battle starts clean."""
        if self.battle is not None:
            return
        self.n_battles += 1
        self.battle = Battle(self.n_battles, self._cur_t, self.clock(), len(loop.decisions))
        self.battle.check_t0 = self._cur_t
        loop.reset_battle(self.n_battles)
        self.gate.allow()                                  # every battle is played (D-055); the banner is only recorded
        self.set_state(CHECKING, "match started — reading the opponent banner…")

    def _playing(self) -> None:
        """The banner check is settled (its text and verdict are recorded): say who the battle is against."""
        who = self.battle.banner.strip() or "(banner not read)"
        how = "playing" if self.taps_on else "watch only (taps OFF)"
        self.set_state(PLAYING, f"battle vs {who} — {how}")

    def _check_banner(self, frame, t: float) -> None:
        """One banner read (at most every CHECK_GAP_S), then the record as soon as it is settled: two trainer reads =
        Training Camp; the limit (CHECK_FRAMES reads, CHECK_WINDOW_S seconds) or too few reads left = not. Neither
        outcome gates anything."""
        b = self.battle
        if t - b.last_read_t >= CHECK_GAP_S:
            b.last_read_t = t
            text = self.reader(frame) or ""
            hit = self._is_trainer(text)
            b.reads.append((round(t, 2), text, hit))
            self.totals["banner_reads"] += 1
            if hit and not b.banner:
                b.banner = text
        hits = sum(1 for r in b.reads if r[2])
        if self._verdict([r[2] for r in b.reads]):
            b.verdict = "TRAINING_CAMP"
            self._playing()
        elif (len(b.reads) >= CHECK_FRAMES or t - b.check_t0 >= CHECK_WINDOW_S
              or hits + CHECK_FRAMES - len(b.reads) < 2):
            texts = [r[1] for r in b.reads if r[1]]
            b.banner = b.banner or (texts[0] if texts else "")
            b.verdict = "NOT_TRAINING_CAMP"
            self._playing()

    def _end_battle(self, t: float, why: str) -> None:
        b, loop = self.battle, self.loop
        if b is None:
            return
        rows = loop.decisions[b.decisions_at:]
        taps = sum(1 for d in rows if d.executor.get("sent"))
        gated = sum(1 for d in rows if d.executor.get("status") == "gated")
        planned = sum(1 for d in rows if not d.executor.get("sent")
                      and d.executor.get("status", "dry_run") in ("planned", "planned_ability", "dry_run"))
        holds = sum(1 for d in rows if d.key == HOLD_KEY)
        verdict = b.verdict if b.verdict != "CHECKING" else "UNKNOWN"
        duration = max(0.0, t - b.t_start)
        tail = list(self._tail) or ([self._last_match_frame] if self._last_match_frame is not None else [])
        self._tail.clear()
        self._last_match_frame = None
        end_frames = self.end_frames.submit(b.n, tail)         # queued for the writer thread; the loop does not wait
        row = {"n": b.n, "t_start": round(b.t_start, 2), "t_end": round(t, 2), "duration_s": round(duration, 1),
               "wall_start": round(b.wall_start, 1), "banner": b.banner, "verdict": verdict, "decisions": len(rows),
               "holds": holds, "taps": taps, "planned": planned, "gated": gated, "ended": why,
               "end_frames": end_frames, "banner_reads": [list(r) for r in b.reads]}
        with open(self.out_dir / "battles.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        tot = self.totals
        tot["battles"] += 1
        tot[{"TRAINING_CAMP": "training_camp", "NOT_TRAINING_CAMP": "not_training_camp"}.get(verdict, "unknown")] += 1
        tot["decisions"] += len(rows)
        tot["taps"] += taps
        tot["gated"] += gated
        tot["planned"] += planned
        mins, secs = divmod(int(round(duration)), 60)
        extra = f" ({planned} planned)" if not self.taps_on and planned else ""
        self.say(f"battle over: {mins}:{secs:02d}, {len(rows)} decisions, {taps} taps{extra}"
                 + (f" [{why}]" if why != "battle over" else ""))
        self.battle = None
        loop.battle = None
        self.gate.close("no battle")

    # -- frames -------------------------------------------------------------------------------------------------
    def on_frame(self, frame, t: float, idx: int, capture_ms: float = 0.0):
        """One frame through the loop and the state machine; `t` / `idx` are session time (see `watch`)."""
        self._cur_t = t
        self._last_t, self._last_idx = t, idx
        loop = self.loop
        decision = None
        t0 = time.perf_counter()
        try:
            decision = loop.step(frame, t, idx, capture_ms)
        except Exception as exc:                   # noqa: BLE001  one bad frame must not end a demo that runs for hours
            msg = f"{type(exc).__name__}: {exc}"
            self.totals["step_errors"] += 1
            if msg not in self._errors and len(self._errors) < 20:
                self._errors.add(msg)
                self.say(f"  (frame error, continuing: {msg})")
        t1 = time.perf_counter()
        self.totals["frames"] += 1
        ready = bool(loop.frames) and loop.frames[-1]["readiness"] == "match"
        if len(loop.frames) > FRAMES_KEEP:
            del loop.frames[:FRAMES_KEEP // 2]
        read = False
        if ready:
            self._last_ready_t = t
            if self.battle is not None:
                self._last_match_frame = frame
                if self._tail:
                    self._tail.clear()                  # the screen came back: only the last gap before the end counts
                if self.battle.verdict == "CHECKING":
                    self._check_banner(frame, t)
                    read = True
        elif self.battle is not None:
            self._tail.append(frame)                    # a reference: the 3 newest are kept, the rest are freed
            if t - self._last_ready_t >= BATTLE_END_S:
                self._end_battle(t, "battle over")
                self.set_state(WATCHING, "watching (no battle)")
        spent = time.perf_counter() - t1
        self._overhead_s += spent
        if not read:
            self._overhead_watch_s += spent
            self._overhead_frames += 1
        return decision

    # -- the window -----------------------------------------------------------------------------------------------
    def attach(self, source) -> None:
        """A new frame source (the window opened or came back): the executor for its rect, perception's gate and OCR worker
        reset when it is a return, and the clock continues so a restarting source cannot turn game time backwards."""
        if self.executor_factory is not None:
            self.loop.executor = self.executor_factory(source)
        self.loop.source = source
        self.loop._rect_changes_seen = 0
        if hasattr(self.loop.perception, "source"):
            self.loop.perception.source = source
        if self._opened and self.reset_perception is not None:
            self.reset_perception(self.loop.perception)
        self.loop._in_match = False
        self._opened += 1
        self._t_off = self._last_t + (1.0 if self._opened > 1 else 0.0)
        self._idx_off = self._last_idx + (1 if self._opened > 1 else 0)
        self._last_ready_t = self._last_t
        self.source = source
        self.set_state(WATCHING, "watching (no battle)")

    def detach(self, why: str) -> None:
        """The window closed or the capture failed: end the battle, close the source, wait for the window."""
        if self.battle is not None:
            self._end_battle(self._last_t, "window closed")
        src, self.source = self.source, None
        self.loop.source = None
        if src is not None:
            try:
                src.close()
            except Exception:                      # noqa: BLE001  a source that failed may not close either
                pass
            self.totals["window_losses"] += 1
        self.state = None
        self.set_state(WAITING_WINDOW, "waiting for the iPhone Mirroring window…" + (f" ({why})" if why else ""))

    def watch(self, open_source, max_seconds: float | None = None, sleep=None) -> str:
        """Run until `stop` ("stopped"), Ctrl-C ("Ctrl-C"), the end of a video ("source ended") or `max_seconds` of session
        time ("time limit"); returns why. `open_source()` returns a frame source or raises (the window is not there):
        retried every WINDOW_RETRY_S."""
        sleep = sleep or time.sleep
        it = None
        t_first: float | None = None
        self.set_state(WAITING_WINDOW, "waiting for the iPhone Mirroring window…")
        try:
            while not self.stop:
                if self.source is None:
                    try:
                        src = open_source()
                    except Exception as exc:           # noqa: BLE001  not there yet (or no Screen Recording permission)
                        msg = f"{type(exc).__name__}: {exc}"
                        if "window not found" not in msg and msg not in self._seen_errors:
                            self._seen_errors.add(msg)
                            self.say(f"  ({msg})")
                        sleep(WINDOW_RETRY_S)
                        continue
                    self.attach(src)
                    it = iter(src.frames())
                try:
                    frame, t, idx = next(it)
                except StopIteration:
                    return "source ended"
                except Exception as exc:               # noqa: BLE001  the window closed / the capture failed
                    self.detach(f"{type(exc).__name__}: {exc}")
                    continue
                ts = t + self._t_off
                if t_first is None:
                    t_first = ts
                if max_seconds is not None and ts - t_first >= max_seconds:
                    return "time limit"
                capture_ms = getattr(self.source, "last_grab_ms", None)
                self.on_frame(frame, ts, idx + self._idx_off, 0.0 if capture_ms is None else capture_ms)
        except KeyboardInterrupt:
            return "Ctrl-C"
        return "stopped"

    def finish(self, reason: str, extra: dict | None = None) -> dict:
        """End an open battle, close the loop (summary.json, shots, files) and the source, print the session totals."""
        if self.battle is not None:
            self._end_battle(self._last_t, reason)
        src, self.source = self.source, None
        if src is not None:
            try:
                src.close()
            except Exception:                      # noqa: BLE001
                pass
        self.end_frames.close()                    # the last battle's images are written before the session ends
        tot = dict(self.totals)
        tot["end_frames_written"], tot["end_frames_errors"] = self.end_frames.written, self.end_frames.errors + self.end_frames.dropped
        tot["runtime_s"] = round(self.clock() - self.started, 1)
        tot["watch_overhead_ms"] = (round(1000.0 * self._overhead_watch_s / self._overhead_frames, 4)
                                    if self._overhead_frames else None)       # mean per frame outside the banner check
        tot["check_ms_total"] = round(1000.0 * (self._overhead_s - self._overhead_watch_s), 1)   # frames that read the banner
        summary = self.loop.finish({"demo": {**tot, "stopped_by": reason, **(extra or {})}})
        self.say(f"session: {tot['battles']} battles ({tot['training_camp']} Training Camp, {tot['not_training_camp']} other), "
                 f"{tot['decisions']} decisions, {tot['taps']} taps, {tot['gated']} gated, {tot['frames']} frames, "
                 f"{tot['runtime_s']:.0f} s — files in {self.out_dir}")
        return summary


# ----------------------------------------------------------------------------------------------- load / warm up
def warm_up(stack: Stack, setup: Setup, say=print) -> dict:
    """One detector pass on a blank 720x1280 frame (the arena crop of the calibration) and one d1 ranking of a synthetic
    4-candidate prompt, each twice so the first (compile / cache) and the steady time both show. Returns the seconds."""
    import numpy as np
    out: dict[str, float] = {}
    det = getattr(stack.perception, "detector", None)
    if det is not None:
        frame = np.zeros((1280, 720, 3), np.uint8)
        calib = stack.perception.calib
        cx, cy, cw, ch = calib.content_rect(720, 1280)
        content = frame[cy:cy + ch, cx:cx + cw]
        crop = calib.arena_crop(cw, ch)
        for name in ("detector_first_s", "detector_next_s"):
            t0 = time.perf_counter()
            det.detect(content, crop)
            out[name] = round(time.perf_counter() - t0, 3)
        say(f"  warm-up detector: {out['detector_first_s']:.2f} s first pass, {out['detector_next_s']:.2f} s next")
    ranker = stack.ranker
    if hasattr(ranker, "choose_keys"):
        load = getattr(ranker, "_load", None)
        if callable(load):
            t0 = time.perf_counter()
            load()
            out["d1_load_s"] = round(time.perf_counter() - t0, 3)
            say(f"  d1 model load: {out['d1_load_s']:.2f} s")
        from clashbot.state_text import render_context
        deck = setup.deck
        state = {"clock": "2:30", "phase": "single_elixir", "match_seconds": 30,
                 "own": {"elixir": 6.0, "hand_confidence": None, "hero_cards": list(setup.hero_cards),
                         "towers": {"king": None, "left": 4000, "right": 4000}, "hand": deck[:4], "next_card": deck[4],
                         "deck": list(deck), "deck_status": "valid", "recent_plays": []},
                 "opponent": {"elixir_estimate": 5.0, "towers": {"king": None, "left": 4000, "right": 4000},
                              "deck_known": [], "deck_complete": False, "recent_plays": []},
                 "units": [], "threats": []}
        keys = [f"{deck[0]}|left|back", f"{deck[1]}|right|back", f"{deck[2]}|left|mid", f"{deck[3]}|right|mid"]
        for name in ("d1_first_rank_s", "d1_next_rank_s"):
            t0 = time.perf_counter()
            ranker.choose_keys(render_context(state, 30.0), keys, stack.plan_text)
            out[name] = round(time.perf_counter() - t0, 3)
        say(f"  warm-up d1 ranking of 4 candidates: {out['d1_first_rank_s']:.2f} s first, {out['d1_next_rank_s']:.2f} s next")
    return out


def reset_perception(perception) -> None:
    """Perception after a window loss: a fresh match gate (so the next battle's entry resets the match state, whatever the
    old gate believed) and a drained OCR worker. Safe on anything without those parts."""
    gate = getattr(perception, "gate", None)
    if gate is not None:
        perception.gate = type(gate)(gate.enter_frames, gate.exit_seconds)
        perception.match_started = False
    close = getattr(perception, "close", None)
    if callable(close):
        close()


def start_dashboard(follow: Path, say=print):
    """`python -m clashbot.viz --follow <runs/demo> --open` when MAC-AO's module exists (never imported here)."""
    try:
        found = importlib.util.find_spec("clashbot.viz") is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        say("dashboard not installed")
        return None
    return subprocess.Popen([sys.executable, "-m", "clashbot.viz", "--follow", str(follow), "--open"])


# ----------------------------------------------------------------------------------------------- command line
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m clashbot.demo", description=TOS_NOTE)
    ap.add_argument("--live", action="store_true",
                    help=f"send real taps in every battle it sees (also needs {LIVE_ENV}=1); without both the demo only watches")
    ap.add_argument("--deck", default=None, help="the 8 cards, names or slugs, comma-separated (skips the question)")
    ap.add_argument("--evo-cards", default=None, help="evolution cards of the deck (skips the question; '' = none)")
    ap.add_argument("--hero-cards", default=None, help="hero cards of the deck (skips the question; '' = none)")
    ap.add_argument("--yes", action="store_true", help="do not ask for the confirmation")
    ap.add_argument("--dashboard", action="store_true", help="open the browser dashboard (MAC-AO) when it is installed")
    ap.add_argument("--out", default=None, help="session directory (default runs/demo/<timestamp>)")
    # test / verification flags
    ap.add_argument("--source", choices=("mirroring", "video"), default="mirroring", help=argparse.SUPPRESS)
    ap.add_argument("--video", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--start", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--seconds", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--setup-file", default=str(SETUP_FILE), help=argparse.SUPPRESS)
    return ap


def live_args(demo_args, setup: Setup, out_dir: Path):
    """The `clashbot.live` namespace of today's live command (FIXED_FLAGS) for this setup; taps are decided by the caller."""
    argv = [*FIXED_FLAGS, "--deck", ",".join(setup.deck), "--hero-cards", ",".join(setup.hero_cards),
            "--source", demo_args.source, "--out", str(out_dir)]
    if demo_args.source == "video":
        argv += ["--video", demo_args.video or "", "--start", str(demo_args.start)]
    args = live_parser().parse_args(argv)
    args.evo_cards = ",".join(setup.evo_cards)
    args.live = False
    args.dry_run = True
    args.seconds = demo_args.seconds
    args.device = resolve_device(args.device)
    return args


def main(argv: list[str] | None = None, ask=input, say=print, environ=None) -> int:
    say(TOS_NOTE)
    demo = build_parser().parse_args(argv)
    if demo.live and demo.source != "mirroring":
        print("error: --live needs --source mirroring", file=sys.stderr)
        return 2
    if demo.source == "video" and not demo.video:
        print("error: --source video needs --video PATH", file=sys.stderr)
        return 2
    cards = load_cards()
    try:
        setup = collect_setup(demo, cards, ask=ask, say=say, setup_file=Path(demo.setup_file))
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print("error: no answer (give --deck / --evo-cards / --hero-cards / --yes to run without a keyboard)", file=sys.stderr)
        return 2
    out_dir = Path(demo.out) if demo.out else DEMO_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    taps_on, why = tap_status(demo.live, environ)
    if demo.live and not taps_on:
        say(D031_MESSAGE)
    args = live_args(demo, setup, out_dir)
    args.live = taps_on                                  # the executor's own check (D-031) still applies
    args.dry_run = not taps_on

    say("Loading models (nothing is tapped while they load)…")
    t_load = time.perf_counter()
    try:
        stack = build_stack(args, with_source=False, on_stage=lambda name, s: say(f"  loaded {name}: {s:.1f} s"))
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("interrupted while loading", file=sys.stderr)
        return 130
    warm = warm_up(stack, setup, say)
    import numpy as np
    from cr_perception.hud import OcrReader
    from cr_perception.trainer_banner import read_banner
    t0 = time.perf_counter()
    ocr = OcrReader()                                    # its own engine: the perception's OCR worker keeps its own
    reader = lambda frame: read_banner(frame, args.calib, ocr)       # noqa: E731
    try:
        reader(np.zeros((1392, 632, 3), np.uint8))       # warm-up: the first recognition pass is the slow one
    except Exception:                                    # noqa: BLE001  a failing OCR is "no banner" (fail closed), not a crash
        pass
    warm["banner_ocr_s"] = round(time.perf_counter() - t0, 3)
    say(f"  banner OCR load + warm-up: {warm['banner_ocr_s']:.2f} s")
    load_s = {**stack.load_s, **warm, "total_s": round(time.perf_counter() - t_load, 1)}
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "taps_on": taps_on, "taps_reason": why,
                                                     "evo_cards_list": setup.evo_cards, "hero_cards_list": setup.hero_cards},
                                                    indent=2, default=str))

    def executor_for(source):
        try:
            return _build_executor(args, source)
        except PermissionError as exc:                   # the guard says no (D-031): never taps
            say(f"  {exc}; continuing watch-only")
            args.live = False
            return _build_executor(args, source)

    shots = None
    if args.record_shots:
        from clashbot.recorder_shots import ShotRecorder
        shots = ShotRecorder(out_dir / "shots", calib=args.calib, max_decisions=args.shots_max)
    from clashbot.actions import load_placement_prior
    gate = TapGate()
    loop = LiveLoop(None, stack.perception, stack.builder, stack.generator, stack.ranker, None, stack.cards,
                    load_placement_prior(), plan_text=stack.plan_text, out_dir=out_dir, min_gap_s=args.min_gap,
                    policy=stack.policy, tactics=stack.tactics, coach=stack.coach, shots=shots, kg2=stack.kg2)
    session = DemoSession(loop, out_dir, reader, gate, taps_on, say=say, executor_factory=executor_for,
                          reset_perception=reset_perception)

    def on_sigint(signum, frame):                        # the first Ctrl-C stops cleanly, a second one exits at once
        if session.stop:
            os._exit(130)
        session.stop = True
        say("\nstopping… (Ctrl-C again to exit at once)")

    previous = signal.signal(signal.SIGINT, on_sigint)
    dashboard = None
    try:
        if taps_on:
            say(TAPS_ON_NOTE)
        say(f"READY — open iPhone Mirroring and start a Training Camp battle. Taps: {'ON' if taps_on else 'OFF'} "
            f"({why}). Ctrl-C to stop.")
        if demo.dashboard:
            dashboard = start_dashboard(DEMO_DIR if not demo.out else out_dir, say)
        stopped = session.watch(lambda: _build_source(args), max_seconds=demo.seconds)
        session.finish("Ctrl-C" if stopped == "stopped" else stopped,
                       {"load": load_s, "taps_on": taps_on, "taps_reason": why, "device": args.device})
    finally:
        signal.signal(signal.SIGINT, previous)
        if dashboard is not None and dashboard.poll() is None:
            dashboard.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
