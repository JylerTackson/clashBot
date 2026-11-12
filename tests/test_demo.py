"""Demo launcher (MAC-AN; D-055 removed the opponent gate): card-name resolution, the prompts, the tap guard, and the watch
loop's state machine (every battle is played once taps are ON, the banner is only recorded) with a fake source / perception /
builder / executor around a real LiveLoop. No model is loaded and nothing is ever sent."""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clashbot.demo as demo  # noqa: E402
from clashbot.candidates import Candidate  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402
from clashbot.live import LiveLoop, Stack  # noqa: E402
from clashbot.policy import ActDecision  # noqa: E402
from clashbot.ranker import RankResult  # noqa: E402

CARDS = load_cards()
DECK = ["electro-dragon", "musketeer", "tesla", "fireball", "knight", "zap", "ram-rider", "mini-p-e-k-k-a"]
PRIOR = {"by_key": {"giant|left|back": {"median_tile": [3.0, 2.0]}}, "by_lane_zone": {"left|mid": {"median_tile": [4.0, 8.0]}}}


# ------------------------------------------------------------------------------------------------ name resolution
@pytest.mark.parametrize("text, slug", [
    ("knight", "knight"), ("Knight", "knight"), ("  KNIGHT ", "knight"), ("electro dragon", "electro-dragon"),
    ("Electro-Dragon", "electro-dragon"), ("mini pekka", "mini-p-e-k-k-a"), ("Mini P.E.K.K.A", "mini-p-e-k-k-a"),
    ("mini p.e.k.k.a.", "mini-p-e-k-k-a"), ("pekka", "p-e-k-k-a"), ("e dragon", "electro-dragon"),
    ("knigth", "knight"), ("muskateer", "musketeer"), ("ram rider", "ram-rider"), ("RamRider", "ram-rider")])
def test_resolve_card_exact_prefix_and_fuzzy(text, slug):
    assert demo.resolve_card(text, CARDS) == (slug, [], False)


def test_resolve_card_ambiguous_lists_three_closest():
    slug, near, ambiguous = demo.resolve_card("mini", CARDS)
    assert slug is None and ambiguous and 2 <= len(near) <= 3 and "mini-p-e-k-k-a" in near
    slug, near, ambiguous = demo.resolve_card("e dragon", CARDS, pool=DECK + ["inferno-dragon"])
    assert slug == "electro-dragon"                                  # inside a deck the short name is unambiguous


def test_resolve_card_unknown_and_empty():
    slug, near, ambiguous = demo.resolve_card("xyzzy", CARDS)
    assert slug is None and not ambiguous
    assert demo.resolve_card("", CARDS) == (None, [], False) and demo.resolve_card("...", CARDS) == (None, [], False)


def test_resolve_card_pool_limits_the_choice():
    assert demo.resolve_card("knight", CARDS, pool=DECK)[0] == "knight"
    assert demo.resolve_card("giant", CARDS, pool=DECK)[0] is None


def test_parse_cards_errors_are_specific():
    assert demo.parse_cards("Knight, Zap ,mini pekka", CARDS) == ["knight", "zap", "mini-p-e-k-k-a"]
    assert demo.parse_cards("", CARDS) == [] and demo.parse_cards(" , ", CARDS) == []
    with pytest.raises(demo.SetupError, match="listed twice"):
        demo.parse_cards("knight, Knight", CARDS)
    with pytest.raises(demo.SetupError, match="not a card"):
        demo.parse_cards("knight, xyzzy", CARDS)
    with pytest.raises(demo.SetupError, match="matches several cards.*Closest"):
        demo.parse_cards("mini", CARDS)


def test_check_deck_needs_eight():
    assert demo.check_deck(DECK) == DECK
    for n in (0, 7, 9):
        with pytest.raises(demo.SetupError, match="8 cards"):
            demo.check_deck((DECK * 2)[:n])


def test_check_variants_evolutions_heroes_and_both():
    assert demo.check_variants(["tesla", "electro-dragon"], DECK, CARDS, "has_evolution", "evolution") == ["tesla", "electro-dragon"]
    assert demo.check_variants(["musketeer"], DECK, CARDS, "has_hero_variant", "hero variant") == ["musketeer"]
    with pytest.raises(demo.SetupError, match="fireball has no evolution.*In this deck: electro-dragon"):
        demo.check_variants(["fireball"], DECK, CARDS, "has_evolution", "evolution")
    with pytest.raises(demo.SetupError, match="giant is not in the deck"):
        demo.check_variants(["giant"], DECK, CARDS, "has_evolution", "evolution")
    with pytest.raises(demo.SetupError, match="cannot be both"):
        demo.check_variants(["knight"], DECK, CARDS, "has_hero_variant", "hero variant", ["knight"], "an evolution")


# ------------------------------------------------------------------------------------------------ prompts
def _args(*argv):
    return demo.build_parser().parse_args(list(argv))


class Keys:
    """Scripted answers for `ask`; a prompt after the script is a test failure."""

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        assert self.answers, f"unexpected prompt: {prompt!r}"
        return self.answers.pop(0)


def test_prompts_ask_the_deck_evolutions_heroes_and_confirm(tmp_path):
    keys = Keys("knight, zap, tesla", "knight, zap, tesla, fireball, mini pekka, e dragon, musketeer, ram rider",
                "tesla, e dragon", "musketeer", "")
    out = []
    setup = demo.collect_setup(_args(), CARDS, ask=keys, say=out.append, setup_file=tmp_path / "last.json")
    assert setup == demo.Setup(["knight", "zap", "tesla", "fireball", "mini-p-e-k-k-a", "electro-dragon", "musketeer", "ram-rider"],
                               ["tesla", "electro-dragon"], ["musketeer"])
    assert [p.split("(")[0].strip() for p in keys.prompts] == [
        "Your 8 cards", "Your 8 cards", "Which are evolutions?", "Which are heroes?", "Start with this deck? [Y/n]"]
    assert any("3 cards" in line or "8 cards" in line for line in out)                       # the count error was explained
    assert json.loads((tmp_path / "last.json").read_text()) == {
        "deck": setup.deck, "evo_cards": ["tesla", "electro-dragon"], "hero_cards": ["musketeer"]}


def test_prompts_reask_on_unknown_ambiguous_duplicate_and_bad_variants(tmp_path):
    deck = ",".join(DECK)
    keys = Keys("xyzzy, " + deck, "mini, " + deck, "knight, " + ",".join(DECK[:7]), deck,
                "fireball", "tesla", "tesla", "musketeer", "y")
    out = []
    setup = demo.collect_setup(_args(), CARDS, ask=keys, say=out.append, setup_file=tmp_path / "last.json")
    text = "\n".join(out)
    assert "'xyzzy' is not a card" in text and "'mini' matches several cards" in text and "knight is listed twice" in text
    assert "fireball has no evolution" in text and "tesla has no hero variant" in text
    assert setup.evo_cards == ["tesla"] and setup.hero_cards == ["musketeer"] and not keys.answers


def test_a_card_cannot_be_both_evolution_and_hero(tmp_path):
    keys = Keys(",".join(DECK), "knight", "knight", "", "y")
    out = []
    setup = demo.collect_setup(_args(), CARDS, ask=keys, say=out.append, setup_file=tmp_path / "last.json")
    assert "knight cannot be both hero variant and an evolution" in "\n".join(out)
    assert setup.evo_cards == ["knight"] and setup.hero_cards == []


def test_flags_skip_their_prompts_and_yes_skips_the_confirmation(tmp_path):
    never = Keys()
    setup = demo.collect_setup(_args("--deck", ",".join(DECK), "--evo-cards", "tesla,electro-dragon", "--hero-cards", "musketeer",
                                     "--yes"), CARDS, ask=never, say=lambda s: None, setup_file=tmp_path / "last.json")
    assert setup.deck == DECK and setup.evo_cards == ["tesla", "electro-dragon"] and setup.hero_cards == ["musketeer"]
    empty = demo.collect_setup(_args("--deck", ",".join(DECK), "--evo-cards", "", "--hero-cards", "", "--yes"), CARDS,
                               ask=never, say=lambda s: None, setup_file=tmp_path / "last.json")
    assert empty.evo_cards == [] and empty.hero_cards == []
    keys = Keys("")                                                   # only the confirmation is asked
    demo.collect_setup(_args("--deck", ",".join(DECK), "--evo-cards", "", "--hero-cards", ""), CARDS, ask=keys,
                       say=lambda s: None, setup_file=tmp_path / "last.json")
    assert keys.prompts == ["Start with this deck? [Y/n] "]


def test_a_bad_flag_is_an_error_not_a_prompt(tmp_path):
    for flags in (["--deck", "knight,zap"], ["--deck", ",".join(DECK), "--evo-cards", "fireball"],
                  ["--deck", ",".join(DECK), "--evo-cards", "", "--hero-cards", "zap"],
                  ["--deck", ",".join(DECK), "--evo-cards", "knight", "--hero-cards", "knight"]):
        with pytest.raises(demo.SetupError):
            demo.collect_setup(_args(*flags, "--yes"), CARDS, ask=Keys(), say=lambda s: None, setup_file=tmp_path / "l.json")


def test_declining_the_confirmation_cancels_and_saves_nothing(tmp_path):
    keys = Keys(",".join(DECK), "", "", "n")
    with pytest.raises(demo.SetupError, match="cancelled"):
        demo.collect_setup(_args(), CARDS, ask=keys, say=lambda s: None, setup_file=tmp_path / "last.json")
    assert not (tmp_path / "last.json").exists()


def test_the_last_setup_is_offered_and_enter_reuses_it(tmp_path):
    path = tmp_path / "last.json"
    demo.save_setup(path, demo.Setup(DECK, ["tesla"], ["musketeer"]))
    out, keys = [], Keys("")
    setup = demo.collect_setup(_args(), CARDS, ask=keys, say=out.append, setup_file=path)
    assert setup == demo.Setup(DECK, ["tesla"], ["musketeer"]) and len(keys.prompts) == 1
    assert "Last setup" in out[0] and "tesla" in out[0] and "Press Enter" in keys.prompts[0]
    keys = Keys("n", ",".join(DECK[::-1]), "", "", "y")               # "n": a new one is asked
    setup = demo.collect_setup(_args(), CARDS, ask=keys, say=lambda s: None, setup_file=path)
    assert setup.deck == DECK[::-1] and setup.evo_cards == [] and json.loads(path.read_text())["deck"] == DECK[::-1]
    path.write_text("not json")                                       # a broken file is ignored, not an error
    assert demo.load_setup(path) is None
    keys = Keys(",".join(DECK), "", "", "y")
    assert demo.collect_setup(_args(), CARDS, ask=keys, say=lambda s: None, setup_file=path).deck == DECK
    given = demo.collect_setup(_args("--deck", ",".join(DECK), "--evo-cards", "", "--hero-cards", "", "--yes"), CARDS,
                               ask=Keys(), say=lambda s: None, setup_file=path)       # flags: the last setup is not offered
    assert given.deck == DECK


# ------------------------------------------------------------------------------------------------ the tap guard
def test_tap_status_needs_live_and_the_env_switch():
    assert demo.tap_status(False, {}) == (False, "no --live")
    assert demo.tap_status(False, {"CLASHBOT_ALLOW_LIVE": "1"}) == (False, "no --live")
    assert demo.tap_status(True, {}) == (False, "--live without CLASHBOT_ALLOW_LIVE=1")
    assert demo.tap_status(True, {"CLASHBOT_ALLOW_LIVE": "0"})[0] is False
    assert demo.tap_status(True, {"CLASHBOT_ALLOW_LIVE": "1"}) == (True, "live")


def test_tap_status_is_off_away_from_macos(monkeypatch):
    monkeypatch.setattr(demo.sys, "platform", "linux")
    assert demo.tap_status(True, {"CLASHBOT_ALLOW_LIVE": "1"})[0] is False


def test_the_gate_is_closed_between_battles_and_opens_for_a_battle():
    gate = demo.TapGate()
    assert gate() is False and gate.reason == "no battle"
    gate.allow()
    assert gate() is True
    gate.close("no battle")
    assert gate() is False and gate.reason == "no battle"


def test_live_needs_the_mirroring_source_and_a_video_source_needs_a_path(capsys):
    assert demo.main(["--live", "--source", "video", "--video", "x.mp4"]) == 2
    assert "--live needs --source mirroring" in capsys.readouterr().err
    assert demo.main(["--source", "video"]) == 2
    assert "needs --video PATH" in capsys.readouterr().err


def test_the_assume_training_camp_flag_is_gone():
    with pytest.raises(SystemExit):
        demo.build_parser().parse_args(["--assume-training-camp"])


def test_the_demo_note_and_help_say_it_plays_every_battle_and_recommend_training_camp_and_friendlies(capsys):
    note = " ".join(demo.TOS_NOTE.split())
    assert "plays every battle it sees" in note and "does not check the opponent" in note
    assert "Training Camp or friendly battles" in note and "ranked" in note and "ban" in note
    assert "Training Camp" in demo.D031_MESSAGE and "CLASHBOT_ALLOW_LIVE=1" in demo.D031_MESSAGE
    with pytest.raises(SystemExit):
        demo.build_parser().parse_args(["--help"])
    text = " ".join(capsys.readouterr().out.split())                      # argparse wraps the lines
    assert "plays every battle it sees" in text and "send real taps in every battle it sees" in text


def test_the_only_place_that_sends_input_is_still_the_executor():
    text = (ROOT / "clashbot" / "demo.py").read_text() + (ROOT / "cr_perception" / "trainer_banner.py").read_text()
    for forbidden in ("quartz_click", "CGEventPost", "import Quartz", "pyautogui", "pynput"):
        assert forbidden not in text


# ------------------------------------------------------------------------------------------------ fakes for the loop
class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.001
        return self.now


class FakePerception:
    """readiness and banner come from the frame dict: {"ready": bool, "banner": str}."""

    def __init__(self):
        self.closed = 0
        self.source = None

    def process(self, frame, t, idx):
        return types.SimpleNamespace(readiness="match" if frame["ready"] else "menu")

    def drain_events(self):
        return []

    def close(self):
        self.closed += 1


class FakeBuilder:
    def __init__(self):
        self.resets = 0
        self.state = {
            "phase": "double_elixir", "match_seconds": 150,
            "own": {"elixir": 5.0, "hand": ["zap", "archers", "knight", "giant"], "deck": ["zap", "archers", "knight", "giant"],
                    "towers": {"king": None, "left": 4000, "right": 5000}, "recent_plays": []},
            "opponent": {"elixir_estimate": 4.0, "towers": {"king": None, "left": 3000, "right": 4858}, "deck_known": [],
                         "recent_plays": []},
            "units": [], "threats": []}

    def observe_events(self, events):
        pass

    def build(self, gs):
        return (self.state, "CTX") if gs.readiness == "match" else None

    def reset(self):
        self.resets += 1


class FakeGenerator:
    def generate(self, state, t, ctx, n=4, **filters):
        return [Candidate("zap|left|mid", 1.0, {"knn": 1.0}), Candidate("giant|left|back", 0.9, {"knn": 1.0})]


class FirstKey:
    name = "first"

    def choose_keys(self, context, keys, plan=None, notes=None):
        return RankResult(index=0, probs=[1.0] + [0.0] * (len(keys) - 1), model="fake")


class Policy:
    """Always acts; has the per-battle memory a ReactivePolicy has (so the reset can be seen)."""

    def __init__(self):
        self._acted, self._commits, self.last_report = {("left", "giant"): (1.0, "zap")}, {}, None

    def decide(self, state, t, context_text=None):
        return ActDecision(True, "defend", None, "x")

    def filter_candidates(self, candidates, decision, state):
        return candidates


class FakeExecutor:
    """Records every call; `sent` follows `live` (the real executor sends only when it was built live, D-031)."""

    def __init__(self):
        self.calls = []
        self.live = True

    def execute(self, key, tile, t, hand=None):
        self.calls.append((key, t))
        return {"t": t, "key": key, "tile": tile, "status": "planned", "slot": 0, "clicks": [], "sent": self.live}


class ScriptSource:
    """Yields (frame, t, idx) from `items` = [(ready, banner, t)], then raises `end` (an exception) or stops."""

    def __init__(self, items, end=None):
        self.items, self.end, self.closed = items, end, 0
        self.last_grab_ms = 1.0
        self.rect_changes = 0

    def frames(self):
        for i, (ready, banner, t) in enumerate(self.items):
            yield {"ready": ready, "banner": banner}, t, i
        if self.end is not None:
            raise self.end

    def close(self):
        self.closed += 1


def run_of(banner, n, t0=0.0, dt=0.5, ready=True):
    return [(ready, banner, t0 + k * dt) for k in range(n)]


def timeline(*parts):
    """Concatenate runs: each part is (ready, banner, n); t is continuous at 0.5 s steps."""
    items, t = [], 0.0
    for ready, banner, n in parts:
        items += run_of(banner, n, t, ready=ready)
        t += n * 0.5
    return items


def make_session(tmp_path, taps_on=True, executor=None):
    ex = executor or FakeExecutor()
    perception, builder = FakePerception(), FakeBuilder()
    loop = LiveLoop(None, perception, builder, FakeGenerator(), FirstKey(), None, load_cards(), PRIOR, plan_text="P",
                    out_dir=tmp_path, clock=FakeClock(), min_gap_s=0.0, policy=Policy())
    out = []
    gate = demo.TapGate()
    resets = []
    session = demo.DemoSession(loop, tmp_path, lambda frame: frame["banner"], gate, taps_on, say=out.append,
                               executor_factory=lambda source: ex, reset_perception=lambda p: resets.append(p))
    return session, out, ex, loop, resets


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]




def tags(lines):
    """One letter per printed state line: W waiting, S watching, R reading the banner, P playing ("battle vs ..."),
    O battle over, T session totals (anything else is shown as is)."""
    out = []
    for x in lines:
        for prefix, tag in (("waiting for the iPhone", "W"), ("watching (no battle)", "S"), ("match started", "R"),
                            ("battle vs", "P"), ("battle over", "O"), ("session:", "T")):
            if x.startswith(prefix):
                out.append(tag)
                break
        else:
            out.append(x)
    return out


# ------------------------------------------------------------------------------------------------ the state machine
def test_the_whole_session_waits_plays_every_battle_records_the_banner_survives_the_window_and_stops_on_ctrl_c(tmp_path):
    src1 = ScriptSource(timeline((False, "", 4), (True, "Trainer Cheddar II", 24), (False, "", 10),
                                 (True, "Tyler", 24), (False, "", 10)), end=RuntimeError("window capture failed"))
    src2 = ScriptSource(timeline((False, "", 3), (True, "TraiNeR Bob", 16)), end=KeyboardInterrupt())
    gone = RuntimeError("iPhone Mirroring window not found: open iPhone Mirroring")
    queue = [gone, src1, gone, gone, src2]                       # missing, appears, closes -> missing twice -> returns
    sleeps = []

    def open_source():
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    session, out, ex, loop, resets = make_session(tmp_path)
    reason = session.watch(open_source, sleep=sleeps.append)
    assert reason == "Ctrl-C" and sleeps == [demo.WINDOW_RETRY_S] * 3
    summary = session.finish(reason, {"load": {"total_s": 1.0}})

    # the one line printed on each change of state (nothing per frame)
    assert tags(out) == ["W", "S", "R", "P", "O", "S", "R", "P", "O", "S", "W", "S", "R", "P", "O", "T"]
    assert out[3] == "battle vs Trainer Cheddar II — playing"
    assert out[7] == "battle vs Tyler — playing"                              # a stranger is played too: the banner decides nothing
    assert out[4].startswith("battle over: 0:") and "24 decisions, 24 taps" in out[4]
    assert out[10] == "waiting for the iPhone Mirroring window… (RuntimeError: window capture failed)"
    assert out[13] == "battle vs TraiNeR Bob — playing" and out[14].endswith("16 decisions, 16 taps [Ctrl-C]")
    assert out[15].startswith("session: 3 battles (2 Training Camp, 1 other)")

    # battles.jsonl: one row per battle, with the banner, the verdict and the counts
    b = rows(tmp_path / "battles.jsonl")
    assert [r["n"] for r in b] == [1, 2, 3] and [r["verdict"] for r in b] == ["TRAINING_CAMP", "NOT_TRAINING_CAMP", "TRAINING_CAMP"]
    assert [r["banner"] for r in b] == ["Trainer Cheddar II", "Tyler", "TraiNeR Bob"]
    assert [r["ended"] for r in b] == ["battle over", "battle over", "Ctrl-C"]
    assert (b[0]["decisions"], b[0]["gated"], b[0]["taps"]) == (24, 0, 24)             # played from the first frame
    assert (b[1]["decisions"], b[1]["gated"], b[1]["taps"]) == (24, 0, 24)             # the stranger too
    assert (b[2]["decisions"], b[2]["gated"], b[2]["taps"]) == (16, 0, 16)
    assert len(b[0]["banner_reads"]) == 2 and len(b[1]["banner_reads"]) == 9          # two hits settle it; 9 reads rule it out
    assert all(not r[2] for r in b[1]["banner_reads"]) and all(set(r) >= {"t_start", "t_end", "duration_s"} for r in b)

    # the executor was called in every battle (the banner is only recorded)
    assert len(ex.calls) == 24 + 24 + 16
    assert any(b[1]["t_start"] <= t <= b[1]["t_end"] for _, t in ex.calls)

    # every row carries its battle number; None outside a battle
    decisions = rows(tmp_path / "decisions.jsonl")
    assert sorted({r["battle"] for r in decisions}) == [1, 2, 3] and len(decisions) == 64
    assert {r["executor"]["status"] for r in decisions} == {"planned"} and not any("gate" in r["executor"] for r in decisions)
    frames = rows(tmp_path / "frames.jsonl")
    assert {r["battle"] for r in frames} == {None, 1, 2, 3} and frames[0]["battle"] is None
    assert [r["battle"] for r in frames if r["readiness"] == "match"][:1] == [1]
    ts = [r["t"] for r in frames]                                      # the window's own clock restarts at 0: ours does not
    assert ts == sorted(ts) and len(set(ts)) == len(ts)

    # per-battle reset at each battle start; the window loss reset perception once; every source was closed
    assert loop.builder.resets == 3 and loop.policy._acted == {} and len(resets) == 1
    assert src1.closed == 1 and src2.closed == 1 and loop.perception.closed == 1
    # summary.json: the loop's summary plus the demo block
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["demo"]["battles"] == 3 and s["demo"]["training_camp"] == 2 and s["demo"]["not_training_camp"] == 1
    assert s["demo"]["taps"] == 64 and s["demo"]["gated"] == 0 and s["demo"]["stopped_by"] == "Ctrl-C"
    assert s["demo"]["window_losses"] == 1 and s["demo"]["load"] == {"total_s": 1.0}
    assert s["decisions_by_status"] == {"planned": 64} and summary["demo"] == s["demo"]
    assert session.gate() is False and session.gate.reason == "no battle"             # closed again between battles


def test_with_taps_off_a_battle_is_watched_and_planned_not_sent(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path, taps_on=False)

    class DryExecutor(FakeExecutor):
        def execute(self, key, tile, t, hand=None):
            self.calls.append((key, t))
            return {"t": t, "key": key, "tile": tile, "status": "planned", "slot": 0, "clicks": [], "sent": False}

    dry = DryExecutor()
    session.executor_factory = lambda source: dry
    session.watch(lambda: ScriptSource(timeline((True, "Trainer Cheddar", 12), (False, "", 8))))
    session.finish("done")
    assert "battle vs Trainer Cheddar — watch only (taps OFF)" in out
    b = rows(tmp_path / "battles.jsonl")[0]
    assert b["taps"] == 0 and b["planned"] == 12 and b["gated"] == 0 and len(dry.calls) == 12
    assert any("12 decisions, 0 taps (12 planned)" in line for line in out)


def test_any_banner_is_played_and_recorded_the_verdict_gates_nothing(tmp_path):
    cases = [("Trainer Cheddar II", "TRAINING_CAMP", "Trainer Cheddar II"), ("Tyler", "NOT_TRAINING_CAMP", "Tyler"),
             ("", "NOT_TRAINING_CAMP", ""), ("  ", "NOT_TRAINING_CAMP", "  "), ("Tra", "NOT_TRAINING_CAMP", "Tra"),
             ("garbled ###", "NOT_TRAINING_CAMP", "garbled ###"), ("2618", "NOT_TRAINING_CAMP", "2618")]
    for i, (banner, verdict, recorded) in enumerate(cases):
        d = tmp_path / str(i)
        session, out, ex, loop, _ = make_session(d)
        session.watch(lambda: ScriptSource(timeline((True, banner, 14), (False, "", 8))))
        session.finish("done")
        b = rows(d / "battles.jsonl")[0]
        assert len(ex.calls) == 14, banner                              # every frame of the battle reached the executor
        assert b["verdict"] == verdict and b["banner"] == recorded and (b["taps"], b["gated"]) == (14, 0), banner
        assert f"battle vs {recorded.strip() or '(banner not read)'} — playing" in out, banner
        assert not any("not Training Camp" in x or "watching only" in x for x in out)


def test_a_battle_with_no_banner_text_says_so(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path)
    session.watch(lambda: ScriptSource(timeline((True, "", 14), (False, "", 8))))
    session.finish("done")
    assert "battle vs (banner not read) — playing" in out and rows(tmp_path / "battles.jsonl")[0]["banner"] == ""


def test_one_trainer_read_is_not_enough_to_label_it_training_camp_and_it_is_played_anyway(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path)
    items = [(True, "Trainer Cheddar" if k == 0 else "Tyler", k * 0.5) for k in range(14)] + run_of("", 8, 7.0, ready=False)
    session.watch(lambda: ScriptSource(items))
    session.finish("done")
    b = rows(tmp_path / "battles.jsonl")[0]
    assert len(ex.calls) == 14 and b["verdict"] == "NOT_TRAINING_CAMP" and len(b["banner_reads"]) == 10
    assert b["banner"] == "Trainer Cheddar"                              # the text that was read is kept for the log


def test_two_trainer_reads_anywhere_in_the_check_label_it_training_camp(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path)
    names = ["", "Tyler", "Trainer Cheddar", "", "TRAINER Cheddar"] + ["Trainer Cheddar"] * 9
    items = [(True, n, k * 0.5) for k, n in enumerate(names)] + run_of("", 8, 7.0, ready=False)
    session.watch(lambda: ScriptSource(items))
    session.finish("done")
    b = rows(tmp_path / "battles.jsonl")[0]
    assert b["verdict"] == "TRAINING_CAMP" and len(b["banner_reads"]) == 5 and b["banner"] == "Trainer Cheddar"
    assert len(ex.calls) == 14                                          # the decisions of frames 0..4 were not held back


def test_the_first_frame_of_a_battle_is_already_played_and_the_gate_is_closed_between_battles(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path)
    states = []
    real = session.on_frame

    def watch_gate(frame, t, idx, capture_ms=0.0):
        decision = real(frame, t, idx, capture_ms)
        states.append((frame["ready"], session.gate()))
        return decision

    session.on_frame = watch_gate
    session.watch(lambda: ScriptSource(timeline((False, "", 4), (True, "Tyler", 6), (False, "", 10))))
    session.finish("done")
    assert [g for ready, g in states if ready] == [True] * 6                     # open from the first match frame
    assert [g for ready, g in states[:4]] == [False] * 4 and states[-1] == (False, False)    # closed before and after
    d = rows(tmp_path / "decisions.jsonl")
    assert len(d) == 6 and {r["executor"]["status"] for r in d} == {"planned"} and len(ex.calls) == 6


def test_banner_ocr_runs_only_while_checking_and_at_most_every_half_second(tmp_path):
    reads = []
    session, out, ex, loop, _ = make_session(tmp_path / "a")
    session.reader = lambda frame: reads.append(1) or frame["banner"]
    items = [(True, "Trainer Cheddar", k * 0.1) for k in range(60)] + run_of("", 40, 6.0, dt=0.1, ready=False)   # 10 fps
    session.watch(lambda: ScriptSource(items))
    session.finish("done")
    assert len(reads) == 2                                              # two hits settle it; none after PLAYING
    reads.clear()
    session, out, ex, loop, _ = make_session(tmp_path / "b")
    session.reader = lambda frame: reads.append(1) or frame["banner"]
    session.watch(lambda: ScriptSource([(True, "Tyler", k * 0.1) for k in range(80)]))
    session.finish("done")
    assert len(reads) == 9                                              # one per 0.5 s until two hits are out of reach


def test_battle_ends_after_three_seconds_without_a_match_and_a_short_gap_is_the_same_battle(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path)
    items = timeline((False, "", 2), (True, "Trainer X", 10), (False, "", 4), (True, "Trainer X", 10), (False, "", 8))
    session.watch(lambda: ScriptSource(items))                           # a 2 s gap (4 frames) inside one battle
    session.finish("done")
    assert len(rows(tmp_path / "battles.jsonl")) == 1 and loop.builder.resets == 1
    assert sum(1 for x in out if x.startswith("battle over")) == 1
    items = timeline((True, "Trainer X", 10), (False, "", 8), (True, "Trainer X", 10), (False, "", 8))
    session, out, ex, loop, _ = make_session(tmp_path / "two")
    session.watch(lambda: ScriptSource(items))                           # a 4 s gap: two battles
    session.finish("done")
    assert [r["n"] for r in rows(tmp_path / "two" / "battles.jsonl")] == [1, 2] and loop.builder.resets == 2


def test_a_video_that_ends_finishes_the_session_and_so_does_the_time_limit(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path / "a")
    reason = session.watch(lambda: ScriptSource(timeline((True, "Trainer X", 20))), max_seconds=4.0)
    assert reason == "time limit" and len(loop.frames) == 8              # frames at 0.0 .. 3.5 s
    session.finish(reason)
    assert rows(tmp_path / "a" / "battles.jsonl")[0]["ended"] == "time limit"
    session, out, ex, loop, _ = make_session(tmp_path / "b")
    assert session.watch(lambda: ScriptSource(timeline((True, "Trainer X", 6)))) == "source ended"


def test_stop_flag_ends_the_loop_cleanly(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path)

    def frames():
        for k in range(100):
            if k == 5:
                session.stop = True
            yield {"ready": False, "banner": ""}, k * 0.5, k

    src = ScriptSource([])
    src.frames = frames
    assert session.watch(lambda: src) == "stopped" and src.closed == 0
    summary = session.finish("Ctrl-C")
    assert src.closed == 1 and summary["frames"] == 6 and (tmp_path / "summary.json").exists()


def test_a_frame_error_is_reported_once_and_the_loop_continues(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path)
    calls = []
    real_step = loop.step

    def flaky(frame, t, idx, capture_ms):
        calls.append(t)
        if len(calls) in (2, 3):
            raise ValueError("boom")
        return real_step(frame, t, idx, capture_ms)

    loop.step = flaky
    session.watch(lambda: ScriptSource(timeline((False, "", 6))))
    session.finish("done")
    assert [x for x in out if "frame error" in x] == ["  (frame error, continuing: ValueError: boom)"]
    assert session.totals["step_errors"] == 2 and session.totals["frames"] == 6


def test_watching_adds_less_than_a_millisecond_per_frame(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path)
    session.watch(lambda: ScriptSource(timeline((False, "", 400))))
    summary = session.finish("done")
    assert summary["demo"]["watch_overhead_ms"] < 1.0


def test_the_window_closing_mid_battle_ends_it_and_the_next_match_is_checked_again(tmp_path):
    src1 = ScriptSource(timeline((True, "Trainer X", 10)), end=OSError("gone"))
    src2 = ScriptSource(timeline((True, "Tyler", 14), (False, "", 8)))
    queue = [src1, src2]
    session, out, ex, loop, resets = make_session(tmp_path)
    session.watch(lambda: queue.pop(0), sleep=lambda s: None)
    session.finish("done")
    b = rows(tmp_path / "battles.jsonl")
    assert [(r["n"], r["verdict"], r["ended"]) for r in b] == [
        (1, "TRAINING_CAMP", "window closed"), (2, "NOT_TRAINING_CAMP", "battle over")]
    assert any(x.startswith("waiting for the iPhone Mirroring window…") and "OSError: gone" in x for x in out)
    assert len(resets) == 1


def test_reset_perception_makes_a_fresh_gate_and_drains_the_ocr_worker():
    from cr_perception.screen import MatchGate

    class P:
        closed = 0

        def __init__(self):
            self.gate = MatchGate(3, 2.0)
            self.gate.in_match = True
            self.match_started = True

        def close(self):
            P.closed += 1

    p = P()
    demo.reset_perception(p)
    assert p.gate.in_match is False and p.gate.enter_frames == 3 and p.gate.exit_seconds == 2.0
    assert p.match_started is False and P.closed == 1
    demo.reset_perception(object())                                      # anything without those parts is left alone


def test_attach_builds_the_executor_for_the_new_window_and_continues_the_clock(tmp_path):
    session, out, ex, loop, resets = make_session(tmp_path)
    made = []
    session.executor_factory = lambda source: made.append(source) or ex
    s1, s2 = ScriptSource(timeline((False, "", 4)), end=RuntimeError("x")), ScriptSource(timeline((False, "", 4)))
    queue = [s1, s2]
    session.watch(lambda: queue.pop(0), sleep=lambda s: None)
    assert made == [s1, s2] and loop.perception.source is s2 and loop.executor is ex and loop.source is s2
    ts = [r["t"] for r in loop.frames]
    assert len(ts) == 8 and ts == sorted(ts) and ts[4] > ts[3]


def test_dashboard_is_not_imported_and_only_started_when_installed(monkeypatch):
    # Other test modules may already have imported `clashbot.viz`; drop it (monkeypatch restores it afterwards) so the
    # final assertion checks that `start_dashboard` itself does not import the package.
    for name in [n for n in sys.modules if n == "clashbot.viz" or n.startswith("clashbot.viz.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    out = []
    monkeypatch.setattr(demo.importlib.util, "find_spec", lambda name: None)
    assert demo.start_dashboard(Path("runs/demo"), out.append) is None and out == ["dashboard not installed"]
    started = []

    class FakePopen:
        def __init__(self, cmd):
            started.append(cmd)

    monkeypatch.setattr(demo.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(demo.subprocess, "Popen", FakePopen)
    demo.start_dashboard(Path("runs/demo"), out.append)
    assert started == [[sys.executable, "-m", "clashbot.viz", "--follow", "runs/demo", "--open"]]
    assert "clashbot.viz" not in sys.modules


# ------------------------------------------------------------------------------------------------ end-of-battle frames (S4)
class IdSource(ScriptSource):
    """ScriptSource whose frames also carry their index as "id" (to see which frames were kept)."""

    def frames(self):
        for i, (ready, banner, t) in enumerate(self.items):
            yield {"ready": ready, "banner": banner, "id": i}, t, i
        if self.end is not None:
            raise self.end


class RecordingWriter:
    """EndFrameWriter's `writer`: records (relative name, frame id) instead of encoding."""

    def __init__(self):
        self.calls = []

    def __call__(self, path, frame):
        self.calls.append((f"{path.parent.name}/{path.name}", frame["id"]))
        return True


def end_session(tmp_path, **kw):
    writer = RecordingWriter()
    session, out, ex, loop, resets = make_session(tmp_path, **kw)
    session.end_frames = demo.EndFrameWriter(tmp_path, writer=writer)
    return session, writer, loop


def test_a_battle_row_lists_the_last_three_frames_after_the_screen_left_the_match(tmp_path):
    session, writer, loop = end_session(tmp_path)
    # match frames 0..11 (t 0..5.5), then not a match; the battle is over 3 s later, at frame 17 (t 8.5): frames 15, 16, 17
    session.watch(lambda: IdSource(timeline((True, "Trainer X", 12), (False, "", 8))))
    session.finish("done")
    b = rows(tmp_path / "battles.jsonl")[0]
    assert b["end_frames"] == ["battles/1_end_0.jpg", "battles/1_end_1.jpg", "battles/1_end_2.jpg"]
    assert writer.calls == [("battles/1_end_0.jpg", 15), ("battles/1_end_1.jpg", 16), ("battles/1_end_2.jpg", 17)]
    s = json.loads((tmp_path / "summary.json").read_text())["demo"]
    assert s["end_frames_written"] == 3 and s["end_frames_errors"] == 0


def test_a_battle_that_ends_in_the_match_saves_its_last_match_frame(tmp_path):
    session, writer, loop = end_session(tmp_path)
    session.watch(lambda: IdSource(timeline((True, "Trainer X", 12))))              # the video ends mid-match
    session.finish("done")
    assert rows(tmp_path / "battles.jsonl")[0]["end_frames"] == ["battles/1_end_0.jpg"]
    assert writer.calls == [("battles/1_end_0.jpg", 11)]
    # one or two frames after the match: just those, not the match frame
    session, writer, loop = end_session(tmp_path / "two")
    session.watch(lambda: IdSource(timeline((True, "Trainer X", 12), (False, "", 2))))
    session.finish("done")
    assert writer.calls == [("battles/1_end_0.jpg", 12), ("battles/1_end_1.jpg", 13)]


def test_a_short_gap_inside_a_battle_does_not_leak_into_its_end_frames(tmp_path):
    session, writer, loop = end_session(tmp_path)
    items = timeline((True, "Trainer X", 10), (False, "", 4), (True, "Trainer X", 10), (False, "", 8))
    session.watch(lambda: IdSource(items))                    # frames 10..13 are the gap; the battle is over at frame 29 (3 s on)
    session.finish("done")
    assert [i for _, i in writer.calls] == [27, 28, 29]
    assert len(rows(tmp_path / "battles.jsonl")) == 1


def test_every_battle_gets_its_own_frames_gated_ones_and_a_closed_window_included(tmp_path):
    session, writer, loop = end_session(tmp_path)
    src1 = IdSource(timeline((True, "Trainer X", 12), (False, "", 8), (True, "Tyler", 14)), end=OSError("gone"))
    queue_ = [src1, IdSource([], end=KeyboardInterrupt())]                   # the window closes in battle 2, then Ctrl-C
    session.watch(lambda: queue_.pop(0), sleep=lambda s: None)
    session.finish("done")
    b = rows(tmp_path / "battles.jsonl")
    assert [(r["n"], r["verdict"], r["ended"]) for r in b] == [
        (1, "TRAINING_CAMP", "battle over"), (2, "NOT_TRAINING_CAMP", "window closed")]
    assert b[1]["end_frames"] == ["battles/2_end_0.jpg"]                             # the last match frame (id 33)
    assert writer.calls[-1] == ("battles/2_end_0.jpg", 33)
    assert [n for n, _ in writer.calls[:3]] == [f"battles/1_end_{k}.jpg" for k in range(3)]


def test_only_three_frames_and_the_last_match_frame_are_ever_held(tmp_path):
    session, writer, loop = end_session(tmp_path)
    held = []
    real = session.on_frame

    def watch_memory(frame, t, idx, capture_ms=0.0):
        out = real(frame, t, idx, capture_ms)
        held.append(len(session._tail))
        return out

    session.on_frame = watch_memory
    session.watch(lambda: IdSource(timeline((False, "", 100), (True, "Trainer X", 20), (False, "", 2), (True, "Trainer X", 4),
                                            (False, "", 12), (False, "", 100))))
    assert max(held) <= demo.END_FRAMES
    assert len(session._tail) == 0 and session._last_match_frame is None        # nothing is held outside a battle
    session.finish("done")


def test_watching_with_no_battle_holds_no_frame(tmp_path):
    session, writer, loop = end_session(tmp_path)
    session.watch(lambda: IdSource(timeline((False, "", 200))))
    assert len(session._tail) == 0 and session._last_match_frame is None
    session.finish("done")
    assert writer.calls == [] and not (tmp_path / "battles.jsonl").exists()


def test_a_battle_adds_less_than_a_millisecond_per_frame_for_the_end_frames(tmp_path):
    session, out, ex, loop, _ = make_session(tmp_path)
    session.watch(lambda: ScriptSource(timeline((True, "", 300), (False, "", 100))))
    summary = session.finish("done")
    assert summary["demo"]["watch_overhead_ms"] < 1.0


def test_the_loop_does_not_wait_for_the_writer_and_finish_flushes_it(tmp_path):
    import threading
    entered, release, calls = threading.Event(), threading.Event(), []

    def slow_writer(path, frame):
        entered.set()
        release.wait(10)
        calls.append(path.name)
        return True

    session, out, ex, loop, _ = make_session(tmp_path)
    session.end_frames = demo.EndFrameWriter(tmp_path, writer=slow_writer)
    session.watch(lambda: ScriptSource(timeline((True, "Trainer X", 12), (False, "", 8))))     # returns while the writer is stuck
    assert entered.wait(5) and calls == []
    assert rows(tmp_path / "battles.jsonl")[0]["end_frames"] == [f"battles/1_end_{k}.jpg" for k in range(3)]
    release.set()
    session.finish("done")
    assert calls == ["1_end_0.jpg", "1_end_1.jpg", "1_end_2.jpg"]
    assert json.loads((tmp_path / "summary.json").read_text())["demo"]["end_frames_written"] == 3


def test_a_writer_that_fails_is_counted_and_never_stops_the_session(tmp_path):
    def broken(path, frame):
        raise OSError("disk full")

    session, out, ex, loop, _ = make_session(tmp_path)
    session.end_frames = demo.EndFrameWriter(tmp_path, writer=broken)
    session.watch(lambda: ScriptSource(timeline((True, "Trainer X", 12), (False, "", 8), (True, "Trainer X", 12), (False, "", 8))))
    session.finish("done")
    s = json.loads((tmp_path / "summary.json").read_text())["demo"]
    assert s["battles"] == 2 and s["end_frames_written"] == 0 and s["end_frames_errors"] == 6


def test_a_full_queue_drops_the_job_instead_of_blocking(tmp_path):
    import threading
    entered, release = threading.Event(), threading.Event()

    def stuck(path, frame):
        entered.set()
        release.wait(10)
        return True

    w = demo.EndFrameWriter(tmp_path, writer=stuck, queue_size=1)
    assert w.submit(1, ["f"]) == ["battles/1_end_0.jpg"]
    assert entered.wait(5)                                                 # the thread holds job 1, the queue is empty
    assert w.submit(2, ["f", "g"]) == ["battles/2_end_0.jpg", "battles/2_end_1.jpg"]      # queued
    assert w.submit(3, ["f"]) == [] and w.dropped == 1                     # full: dropped, nothing blocks
    assert w.submit(4, []) == []
    release.set()
    w.close()
    assert w.written == 3 and w.errors == 0


def test_end_frames_are_real_jpegs_at_quality_85_and_full_size(tmp_path):
    import cv2
    import numpy as np

    class ImgFrame(np.ndarray):
        """A real image that also answers frame["ready"] / frame["banner"] for the fake perception and reader."""

        def __new__(cls, ready, banner, k):
            obj = np.full((60, 40, 3), 20 + 10 * k, np.uint8).view(cls)
            obj.meta = {"ready": ready, "banner": banner}
            obj[5:25, 5:15] = (255, 0, 0)                                  # something to compress
            return obj

        def __getitem__(self, key):
            return self.meta[key] if isinstance(key, str) else super().__getitem__(key)

    class ImgSource(ScriptSource):
        def frames(self):
            for i, (ready, banner, t) in enumerate(self.items):
                yield ImgFrame(ready, banner, i), t, i

    session, out, ex, loop, _ = make_session(tmp_path)
    session.watch(lambda: ImgSource(timeline((True, "Trainer X", 12), (False, "", 8))))
    session.finish("done")
    b = rows(tmp_path / "battles.jsonl")[0]
    assert b["end_frames"] == [f"battles/1_end_{k}.jpg" for k in range(3)]
    for k, name in enumerate(b["end_frames"]):
        data = (tmp_path / name).read_bytes()
        assert data[:2] == b"\xff\xd8"
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (60, 40, 3)                                    # not scaled
        assert abs(int(img[50, 30, 0]) - (20 + 10 * (15 + k))) <= 4        # frames 15, 16, 17, oldest first
        want = cv2.imencode(".jpg", np.asarray(ImgFrame(False, "", 15 + k)), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        assert data == want                                                # JPEG quality 85
    assert json.loads((tmp_path / "summary.json").read_text())["demo"]["end_frames_written"] == 3


# ------------------------------------------------------------------------------------------------ main (everything faked)
class NoShots:
    def __init__(self, *a, **k):
        pass

    def on_frame(self, *a, **k):
        pass

    def on_decision(self, *a, **k):
        pass

    def summary(self):
        return {}

    def close(self):
        pass


def _fake_main(monkeypatch, tmp_path, src, *argv, environ=None):
    """Run `demo.main` with every model, the window, OCR and the executor faked (nothing is loaded or sent); `src` is a
    frame source or a function(args) that returns one / raises. Returns (rc, printed lines, executor, [args.live of each
    executor built], session dir)."""
    import clashbot.actions as actions_mod
    import clashbot.recorder_shots as shots_mod
    import cr_perception.hud as hud_mod
    import cr_perception.trainer_banner as banner_mod

    ex, built, lines = FakeExecutor(), [], []

    def build_stack(args, **kw):
        assert kw.get("with_source") is False                           # the models load before the window is looked for
        kw["on_stage"]("perception + detector", 0.1)
        return Stack(perception=FakePerception(), builder=FakeBuilder(), cards=load_cards(), kg=None, kg2=None, index=None,
                     generator=FakeGenerator(), tactics=None, ranker=FirstKey(), policy=Policy(), coach=None, plan_text="P",
                     load_s={"perception + detector": 0.1})

    def build_executor(args, source):
        built.append(args.live)
        ex.live = args.live                                             # like the real one: it sends only when built live
        return ex

    monkeypatch.setattr(demo, "build_stack", build_stack)
    monkeypatch.setattr(demo, "warm_up", lambda stack, setup, say=print: say("  warm-up faked") or {"d1_first_rank_s": 0.5})
    monkeypatch.setattr(demo, "_build_executor", build_executor)
    monkeypatch.setattr(demo, "_build_source", src if callable(src) else (lambda args: src))
    monkeypatch.setattr(demo, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr(demo.signal, "signal", lambda *a: None)
    monkeypatch.setattr(hud_mod, "OcrReader", lambda: object())
    monkeypatch.setattr(banner_mod, "read_banner", lambda frame, calib, ocr: frame["banner"])
    monkeypatch.setattr(shots_mod, "ShotRecorder", NoShots)
    monkeypatch.setattr(actions_mod, "load_placement_prior", lambda: PRIOR)
    out = tmp_path / "session"
    rc = demo.main(["--deck", ",".join(DECK), "--evo-cards", "tesla,electro-dragon", "--hero-cards", "musketeer", "--yes",
                    "--out", str(out), "--setup-file", str(tmp_path / "last.json"), *argv], ask=Keys(), say=lines.append,
                   environ=environ)
    return rc, lines, ex, built, out


def test_main_without_the_env_switch_prints_the_d031_message_and_watches_only(monkeypatch, tmp_path):
    monkeypatch.delenv("CLASHBOT_ALLOW_LIVE", raising=False)
    src = ScriptSource(timeline((True, "Trainer Cheddar", 12), (False, "", 8)), end=KeyboardInterrupt())
    rc, lines, ex, built, out = _fake_main(monkeypatch, tmp_path, src, "--live")
    text = "\n".join(lines)
    assert rc == 0
    assert "live taps are disabled; the user enables them with --live and CLASHBOT_ALLOW_LIVE=1" in text
    ready = "READY — open iPhone Mirroring and start a Training Camp battle. Taps: OFF (--live without CLASHBOT_ALLOW_LIVE=1). Ctrl-C to stop."
    assert ready in lines
    assert built == [False]                                              # the executor was built dry: no live switch
    assert "battle vs Trainer Cheddar — watch only (taps OFF)" in lines
    assert lines.index("  warm-up faked") < lines.index(ready)                    # warm-up, then READY
    assert demo.TAPS_ON_NOTE not in lines                                         # no "taps are ON" reminder
    assert ex.calls and not any(r["executor"]["sent"] for r in rows(out / "decisions.jsonl"))     # planned, never sent
    assert rows(out / "battles.jsonl")[0]["taps"] == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["taps_on"] is False and cfg["live"] is False and cfg["evo_cards"] == "tesla,electro-dragon"
    assert cfg["hero_cards"] == "musketeer" and cfg["hero_cards_list"] == ["musketeer"]
    assert cfg["evo_cards_list"] == ["tesla", "electro-dragon"]
    assert cfg["policy"] == "fast" and cfg["ranker"] == "d1" and cfg["d1_model"] == "db/models/d1-ft-joint-tac"
    assert (out / "summary.json").exists() and (out / "battles.jsonl").exists()
    assert json.loads((tmp_path / "last.json").read_text())["evo_cards"] == ["tesla", "electro-dragon"]
    s = json.loads((out / "summary.json").read_text())["demo"]
    assert s["taps_on"] is False and s["stopped_by"] == "Ctrl-C" and s["load"]["d1_first_rank_s"] == 0.5


def test_main_without_live_says_no_live_and_never_prints_the_d031_message(monkeypatch, tmp_path):
    src = ScriptSource(timeline((False, "", 4)), end=KeyboardInterrupt())
    rc, lines, ex, built, out = _fake_main(monkeypatch, tmp_path, src)
    assert rc == 0 and built == [False]
    assert "READY — open iPhone Mirroring and start a Training Camp battle. Taps: OFF (no --live). Ctrl-C to stop." in lines
    assert not any("taps are disabled" in x for x in lines)


def test_main_with_both_switches_builds_the_executor_live_and_says_taps_on(monkeypatch, tmp_path):
    # (the executor is a fake here: the real one would check the process environment again, D-031)
    src = ScriptSource(timeline((True, "Trainer Cheddar", 12), (False, "", 8)), end=KeyboardInterrupt())
    rc, lines, ex, built, out = _fake_main(monkeypatch, tmp_path, src, "--live", environ={"CLASHBOT_ALLOW_LIVE": "1"})
    assert rc == 0 and built == [True]
    ready = "READY — open iPhone Mirroring and start a Training Camp battle. Taps: ON (live). Ctrl-C to stop."
    assert ready in lines and "battle vs Trainer Cheddar — playing" in lines
    assert demo.TAPS_ON_NOTE in lines and lines.index(demo.TAPS_ON_NOTE) == lines.index(ready) - 1
    assert len(ex.calls) == 6                                            # (the real min-gap of 1 s: t = 0 .. 5, from the first frame)


@pytest.mark.parametrize("banner, verdict", [("Trainer Cheddar", "TRAINING_CAMP"), ("Tyler", "NOT_TRAINING_CAMP"),
                                             ("", "NOT_TRAINING_CAMP"), ("2618", "NOT_TRAINING_CAMP")])
def test_main_with_both_switches_any_banner_is_played_and_recorded(monkeypatch, tmp_path, banner, verdict):
    src = ScriptSource(timeline((True, banner, 14), (False, "", 8)), end=KeyboardInterrupt())
    rc, lines, ex, built, out = _fake_main(monkeypatch, tmp_path, src, "--live", environ={"CLASHBOT_ALLOW_LIVE": "1"})
    assert rc == 0 and built == [True] and len(ex.calls) == 7            # (real min-gap of 1 s: t = 0 .. 6) every banner is played
    d = rows(out / "decisions.jsonl")
    assert {r["executor"]["status"] for r in d} == {"planned"} and any(r["executor"]["sent"] for r in d)
    b = rows(out / "battles.jsonl")[0]
    assert (b["banner"], b["verdict"]) == (banner, verdict) and b["taps"] == 7
    assert f"battle vs {banner or '(banner not read)'} — playing" in lines
    assert not any("watching only" in x or "not Training Camp" in x for x in lines)


@pytest.mark.parametrize("flags, environ", [([], {}), ([], {"CLASHBOT_ALLOW_LIVE": "1"}), (["--live"], {}),
                                            (["--live"], {"CLASHBOT_ALLOW_LIVE": "0"})])
@pytest.mark.parametrize("banner", ["Trainer Cheddar", "Tyler", ""])
def test_main_without_both_switches_nothing_is_ever_sent_whatever_the_banner(monkeypatch, tmp_path, flags, environ, banner):
    monkeypatch.delenv("CLASHBOT_ALLOW_LIVE", raising=False)
    src = ScriptSource(timeline((True, banner, 14), (False, "", 8)), end=KeyboardInterrupt())
    rc, lines, ex, built, out = _fake_main(monkeypatch, tmp_path, src, *flags, environ=environ)
    assert rc == 0 and built == [False]                                  # the executor is built dry (D-031 is untouched)
    assert ex.calls and not any(r["executor"]["sent"] for r in rows(out / "decisions.jsonl"))
    b = rows(out / "battles.jsonl")[0]
    assert b["taps"] == 0 and b["banner"] == banner and demo.TAPS_ON_NOTE not in lines
    assert f"battle vs {banner or '(banner not read)'} — watch only (taps OFF)" in lines


def test_main_waits_for_a_missing_window_and_stops_on_the_time_limit(monkeypatch, tmp_path):
    attempts = []

    def window(args):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("iPhone Mirroring window not found: open iPhone Mirroring")
        return ScriptSource(timeline((False, "", 8)))

    monkeypatch.setattr(demo.time, "sleep", lambda s: None)
    rc, lines, ex, built, out = _fake_main(monkeypatch, tmp_path, window, "--seconds", "2")
    assert rc == 0 and len(attempts) == 3 and built == [False]
    assert [t for t in tags(lines) if len(t) == 1] == ["W", "S", "T"]           # one waiting line, however many retries
    assert json.loads((out / "summary.json").read_text())["demo"]["stopped_by"] == "time limit"


# ------------------------------------------------------------------------------------------------ evolutions in the state
def test_evo_cards_ride_in_the_state_next_to_hero_cards_without_touching_the_d1_text():
    from tests.test_live_state import builder, make_gs
    plain, with_evo = builder(hero_cards=["berserker"]), builder(hero_cards=["berserker"], evo_cards=["archers", "tornado"])
    s0, text0 = plain.build(make_gs(100.0))
    s1, text1 = with_evo.build(make_gs(100.0))
    assert "evo_cards" not in s0["own"]                                  # no evolutions: the state is what it always was
    assert s1["own"]["evo_cards"] == ["archers", "tornado"] and s1["own"]["hero_cards"] == ["berserker"]
    assert text1 == text0                                                 # the context text d1 reads is unchanged
    assert {k: v for k, v in s1["own"].items() if k != "evo_cards"} == s0["own"] and s1["units"] == s0["units"]
    with_evo.reset()                                                      # a new battle keeps the config
    assert with_evo.build(make_gs(100.0, match_id=2))[0]["own"]["evo_cards"] == ["archers", "tornado"]
