from cr_perception.clock import ClockTracker, segment_by_clock


def feed(ct, seq, t0=0.0):
    out = []
    for i, r in enumerate(seq):
        out.append(ct.update(t0 + i, r))
    return out


def test_misread_and_impossible_values_do_not_split():
    ct = ClockTracker()
    evs = feed(ct, [120, 119, 599, 118, 117, 171, 116, 115])   # 9:59 impossible, 2:51 one-frame glitch
    assert not any(evs) and ct.match_index == 0 and ct.remaining == 115


def test_overtime_then_new_match():
    ct = ClockTracker()
    seq = [3, 2, 1, 0, 0, 119, 118, 117, 60, 30, 5, 0] + [171, 170, 169, 168]
    evs = feed(ct, seq)
    assert "overtime" in evs and ct.match_index == 1
    i_ot = evs.index("overtime"); i_nm = evs.index("new_match")
    assert i_ot < i_nm
    assert ct.overtime is False and ct.phase() == "single_elixir"


def test_phases():
    ct = ClockTracker()
    for r, ph in [(180, "single_elixir"), (100, "double_elixir"), (30, "triple_elixir")]:
        ct.remaining = r
        assert ct.phase() == ph
    ct.overtime = True
    ct.remaining = 100
    assert ct.phase() == "double_elixir_overtime" and ct.regen_key() == "double"


def test_segment_by_clock():
    samples = [(t, 180 - t) for t in range(0, 175)]                       # game 1: 3:00 -> 0:05
    samples += [(175 + k, 119 - k) for k in range(0, 60)]                # overtime, same game
    samples += [(240 + k, 175 - k) for k in range(0, 100)]               # new game
    samples += [(345, 599), (346, 74), (347, 73)]                        # misread + continue
    segs = segment_by_clock(samples, min_seconds=30)
    assert len(segs) == 2 and segs[0][1] == 0 and 235 <= segs[1][1] <= 241
