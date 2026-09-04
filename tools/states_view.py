"""Compact digest of a game's state samples, for the enrichment agents.

  python3 tools/states_view.py <key> [<key> ...] [--kind key,play,periodic]

`<key>` is `<video_id>-m<match_index>` (or a path to a jsonl). Prints a short
game header and then one block per sample, ordered by `t`:

    -- <id> | <kind> | t=<video s> | <clock>
    <state_text, 6 lines, verbatim>
    key: <the Phase 1 key-moment bullet>            (kind=key only)
    say: <transcript cues within +-6 s, joined>     (when there are any)
    out: <tower deltas | towers taken/lost | verdict | elixir trade | result>

Roughly a tenth of the jsonl: everything an enrichment agent needs to write
`situation_read` / `reaction` / `pro_action_rationale`. The jsonl stays the
source of truth for anything the digest leaves out (unit tiles beyond
`state_text`, `recent_plays` fields, `context_refs`, `quality.notes`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATES_DIR = ROOT / "knowledge_base" / "states"

KEY_TEXT_CHARS = 460
COMMENTARY_CHARS = 380
KINDS = ("key", "play", "periodic")


def clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def num(x) -> str:
    if x is None:
        return "?"
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return f"{x:+d}" if isinstance(x, int) and x else (f"{x:+g}" if isinstance(x, float) else str(x))


def deltas(d) -> str:
    if not d:
        return "?"
    return "/".join("0" if d.get(k) == 0 else num(d.get(k)) for k in ("king", "left", "right"))


def outcome_line(sample: dict) -> str:
    o = sample["outcome"]
    parts = [f"own k/l/r {deltas(o.get('own_tower_hp_delta'))}",
             f"enemy {deltas(o.get('enemy_tower_hp_delta'))}"]
    if o.get("towers_taken"):
        parts.append("took " + ",".join(o["towers_taken"]))
    if o.get("towers_lost"):
        parts.append("lost " + ",".join(o["towers_lost"]))
    trade = o.get("elixir_trade")
    parts.append(f"{o.get('verdict') or '?'}")
    parts.append(f"trade {num(trade) if trade else '0'}")
    parts.append(f"game {o.get('game_result') or 'unknown'}")
    return "out: " + " | ".join(parts)


def header(key: str, samples: list[dict]) -> str:
    first = samples[0]
    src, own, q = first["source"], first["state"]["own"], first["quality"]
    counts = {k: sum(1 for s in samples if s["kind"] == k) for k in KINDS}
    sources = sorted({s["quality"].get("state_source") for s in samples})
    lines = [
        f"# {key} — {src.get('video_title') or '?'} (match {src.get('match_index')}, ryleycr1)",
        f"deck: {', '.join(own.get('deck') or []) or '?'}"
        + (f" | hero: {', '.join(own['hero_cards'])}" if own.get("hero_cards") else "")
        + f" | result: {first['outcome'].get('game_result') or 'unknown'}",
        f"quality: hand_conf {q.get('hand_confidence')} | clock_read "
        f"{'yes' if q.get('clock_read') else 'no'} | state from {', '.join(x for x in sources if x)}",
        f"samples: {len(samples)} (" + ", ".join(f"{k} {counts[k]}" for k in KINDS) + ")"
        f" | match file: {src.get('match_file')}",
        "null hand slots / next_card and action 'unknown' are discarded HUD misreads: treat as unknown.",
    ]
    return "\n".join(lines)


def render_sample(sample: dict) -> str:
    state = sample["state"]
    out = [f"-- {sample['id']} | {sample['kind']} | t={sample['t']} | {state.get('clock') or '?'}",
           sample["state_text"]]
    km = sample.get("key_moment") or {}
    if km.get("text"):
        out.append("key: " + clip(km["text"], KEY_TEXT_CHARS))
    cues = " ".join(c.get("text", "") for c in sample.get("commentary") or [])
    if cues.strip():
        out.append("say: " + clip(cues, COMMENTARY_CHARS))
    out.append(outcome_line(sample))
    return "\n".join(out)


def load(key: str) -> list[dict]:
    path = Path(key) if key.endswith(".jsonl") else STATES_DIR / f"{key}.jsonl"
    if not path.exists():
        raise SystemExit(f"no such state file: {path}")
    samples = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    return sorted(samples, key=lambda s: s["t"])


def render(key: str, samples: list[dict], kinds: tuple[str, ...]) -> str:
    kept = [s for s in samples if s["kind"] in kinds]
    if not kept:
        return header(key, samples) + "\n\n(no samples of that kind)"
    return "\n\n".join([header(key, samples)] + [render_sample(s) for s in kept])


def cli_argv(argv: list[str]) -> list[str]:
    """Keep video ids that start with '-' (e.g. -V4H_YeMGGk-m0.0) out of
    argparse's option parsing by moving positionals behind a '--'."""
    if "--" in argv:
        return argv
    opts: list[str] = []
    pos: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--kind" and i + 1 < len(argv):
            opts += [a, argv[i + 1]]
            i += 2
        elif a.startswith("--kind=") or a in ("-h", "--help"):
            opts.append(a)
            i += 1
        else:
            pos.append(a)
            i += 1
    return opts + (["--"] + pos if pos else [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="+", help="<video_id>-m<match_index> (or a .jsonl path)")
    ap.add_argument("--kind", default=",".join(KINDS), help="comma-separated kinds to show")
    a = ap.parse_args(cli_argv(sys.argv[1:]))
    kinds = tuple(k.strip() for k in a.kind.split(",") if k.strip())
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise SystemExit(f"unknown kind(s): {', '.join(bad)}")
    try:
        for i, key in enumerate(a.keys):
            if i:
                print()
            print(render(Path(key).stem if key.endswith(".jsonl") else key, load(key), kinds))
        sys.stdout.flush()
    except BrokenPipeError:  # piping into head/less
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
