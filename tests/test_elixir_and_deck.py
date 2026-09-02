import math
from pathlib import Path

from cr_perception.elixir_sim import ElixirSimulator, PHASE_SECONDS_PER_ELIXIR
from cr_perception.decktracker import OpponentDeckTracker, load_kb_decks, DECK_SIZE

KB = Path(__file__).resolve().parents[1] / "knowledge_base"


def test_regen_and_clamp():
    s = ElixirSimulator(elixir=5.0)
    s.advance(0.0)
    s.advance(2.8)
    assert math.isclose(s.elixir, 6.0, abs_tol=1e-6)
    s.advance(2.8 + 1.4, phase="double")
    assert math.isclose(s.elixir, 7.0, abs_tol=1e-6)
    s.advance(100.0)
    assert s.elixir == 10.0
    s.spend(4)
    assert s.elixir == 6.0
    s.spend(9)
    assert s.elixir == 0.0
    assert PHASE_SECONDS_PER_ELIXIR["triple"] < PHASE_SECONDS_PER_ELIXIR["double"] < PHASE_SECONDS_PER_ELIXIR["single"]


def test_drift_measurement_and_estimate():
    s = ElixirSimulator(elixir=5.0)
    s.advance(0.0)
    s.advance(5.6)                    # sim = 7
    assert s.observe(5.6, 7) == 0.0
    s.spend(3)                        # sim = 4 ; suppose we missed a 2-cost play -> observed 2
    d = s.observe(6.0, 2)
    assert d == 2.0
    st = s.drift_stats()
    assert st["n"] == 2 and st["max_abs"] == 2.0
    est, conf, (lo, hi) = s.estimate(st["abs_mean"])
    assert lo <= est <= hi and 0 <= conf <= 1


def test_deck_tracker_cycle_convergence():
    deck = ["hog-rider", "ice-spirit", "skeletons", "the-log", "ice-golem", "musketeer", "cannon", "fireball"]
    t = OpponentDeckTracker(kb_decks=[])
    # opponent plays the deck in cycle order twice
    order = deck * 2
    for i, c in enumerate(order):
        if i >= DECK_SIZE:
            assert t.deck_complete
            assert t.check_cycle(c) is True     # prediction says it is in hand
            assert t.cards_until(c) == 0
        t.observe_play(c)
    assert t.deck_complete and t.known_cards == deck
    assert t.predicted_hand() == deck[:4]      # after 16 plays the first 4 are back in hand
    assert t.predicted_next() == deck[4]
    assert t.cycle_confirmed == 8 and t.cycle_violations == 0
    assert t.cards_until("fireball") == 4


def test_kb_inference_ranks_the_right_deck():
    decks = load_kb_decks(KB)
    assert len(decks) >= 10
    target = next(d for d in decks if "hog-rider" in d["cards"] and "musketeer" in d["cards"])
    t = OpponentDeckTracker(kb_decks=decks)
    for c in target["cards"][:4]:
        t.observe_play(c)
    ranked = t.rank_kb_decks(top=3)
    assert ranked[0]["deck_key"] == target["deck_key"]
    preds = t.deck_predictions()
    assert preds and all(p.source == "kb_inference" for p in preds)
    assert {p.card for p in preds} & set(target["cards"][4:])
    assert not t.deck_complete and t.predicted_hand() is None
