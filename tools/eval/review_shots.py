"""Review the decision screenshots of a run (MAC-AF): did each tap land, contact sheets, review.md.

  python3 tools/eval/review_shots.py RUN_DIR [--only failed|all] [--out RUN_DIR/shots/sheets] [--include-dry]
         [--per-sheet 6] [--cell-width 320]

Reads RUN_DIR/shots/index.jsonl (clashbot.recorder_shots, `--record-shots`). Landing verdict per sent tap:
  landed   the elixir dropped by >= cost - 0.6 within 1.5 s (the +0.5 s or +1.5 s reading), or the tapped slot reads
           another card at +0.5 s / +1.5 s;
  failed   neither, with a reading (elixir or slot) at +1.5 s;
  unknown  no elixir reading and no slot reading, or nothing read at +1.5 s (the run ended first).
A hero ability tap (key `ability:musketeer`, status `planned_ability`) has no card slot: it is judged by the elixir drop
alone (FIX #40), landed when it fell by >= cost - 1.5 = 1.5 at the +0.5 s or +1.5 s reading (MAC-AL T3: at 2x / 3x
elixir the bar refills 0.7-1.1 elixir/s, so a tap that worked often shows less than the full 3). The index does not
record the ability button after the tap, so the button state is not used.
Elixir and hand are the raw perception readings when the index has them (`elixir_*`, `hand_raw_*`), else the tracked ones.
A tap that was not sent (dry run) has no verdict unless `--include-dry` (a video dry run: the elixir then follows the
human who played the video). Writes sheet_NNN.png (3 images a / b / c per decision, 6 decisions per sheet, a caption per
row) and review.md (landing rate by card, slot, target row band and mode; how often the tapped card is still in its
slot at +1.5 s) into --out (default RUN_DIR/shots/sheets). `--only failed` limits the sheets to failed taps; the tables
always cover every tap.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from clashbot.actions import ABILITY_COST, is_ability_key  # noqa: E402
from clashbot.config import HOLD_KEY  # noqa: E402
from clashbot.game_rules import load_cards  # noqa: E402

DROP_TOL = 0.6                 # a tap landed when the elixir fell by at least cost - DROP_TOL
ABILITY_DROP_TOL = 1.5         # ... for a hero ability tap cost - 1.5: the bar refills 0.7-1.1 elixir/s at 2x / 3x (MAC-AL T3)
TAP_STATUSES = ("planned", "dry_run", "planned_ability")     # (the ability tap is judged by its elixir drop, FIX #40)
VERDICTS = ("landed", "failed", "unknown")
ROW_BANDS = ((0, 8, "rows 0-7 (own back)"), (8, 16, "rows 8-15 (own front)"),
             (16, 24, "rows 16-23 (enemy front)"), (24, 32, "rows 24-31 (enemy back)"))
TAGS = ("a", "b", "c")


# ----------------------------------------------------------------------------- rows and verdicts

def read_index(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "shots" / "index.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: (r.get("idx", 0), r.get("t", 0.0)))


def row_card(row: dict) -> str:
    return str(row["key"]).split("|")[0]


def is_tap(row: dict) -> bool:
    """A decision that tapped (or, in a dry run, would have): it has a card and was not held or skipped."""
    return row.get("key") != HOLD_KEY and row.get("status") in TAP_STATUSES


def tap_cost(row: dict, cards: dict) -> float | None:
    """Elixir the tap costs: the card's cost, or ABILITY_COST (3) for the hero ability button; None for an unknown card."""
    if is_ability_key(row.get("key")):
        return float(ABILITY_COST)
    return (cards.get(row_card(row)) or {}).get("elixir_cost")


def slot_reads(row: dict) -> list[str]:
    """The cards read in the tapped slot after the tap (+0.5 s, +1.5 s), unread slots left out. Raw reads when the index
    has them, else the tracked hand at +1.5 s."""
    slot = row.get("slot")
    if slot is None:
        return []
    raw = [row.get("hand_raw_after_0.5"), row.get("hand_raw_after_1.5")]
    hands = raw if any(isinstance(h, list) for h in raw) else [row.get("hand_after_1.5")]
    return [h[slot] for h in hands if isinstance(h, list) and slot < len(h) and h[slot]]


def slot_card_at_1_5(row: dict) -> str | None:
    """The card read in the tapped slot at +1.5 s (raw read first), None when unread."""
    slot = row.get("slot")
    h = row.get("hand_raw_after_1.5")
    if not isinstance(h, list):
        h = row.get("hand_after_1.5")
    if slot is None or not isinstance(h, list) or slot >= len(h):
        return None
    return h[slot] or None


def elixir_drop(row: dict) -> float | None:
    """Elixir before minus the lowest reading at +0.5 s / +1.5 s; None without both."""
    before = row.get("elixir_before")
    afters = [v for v in (row.get("elixir_after_0.5"), row.get("elixir_after_1.5")) if isinstance(v, (int, float))]
    if not isinstance(before, (int, float)) or not afters:
        return None
    return float(before) - min(afters)


def landing_verdict(row: dict, cost: float | None) -> str:
    drop = elixir_drop(row)
    reads = slot_reads(row)
    tol = ABILITY_DROP_TOL if is_ability_key(row.get("key")) else DROP_TOL
    if drop is not None and cost is not None and drop >= cost - tol:
        return "landed"
    card = row_card(row)
    if any(c != card for c in reads):
        return "landed"                     # the card left the slot
    if (drop is None or cost is None) and not reads:
        return "unknown"
    if not isinstance(row.get("elixir_after_1.5"), (int, float)) and slot_card_at_1_5(row) is None:
        return "unknown"                    # the 1.5 s window is not read (the run ended, or the HUD was unreadable)
    return "failed"


def classify(rows: list[dict], cards: dict, include_dry: bool = False) -> list[dict]:
    """Rows plus `kind` (hold / tap / dry / skipped) and, for the taps judged, `verdict`, `cost` and `drop`."""
    out = []
    for r in rows:
        r = dict(r)
        if r.get("key") == HOLD_KEY:
            r["kind"] = "hold"
        elif not is_tap(r):
            r["kind"] = "skipped"
        elif r.get("sent") or include_dry:
            r["kind"] = "tap" if r.get("sent") else "dry"
            cost = tap_cost(r, cards)
            r["cost"] = cost
            r["drop"] = elixir_drop(r)
            r["verdict"] = landing_verdict(r, cost)
        else:
            r["kind"] = "dry"
        out.append(r)
    return out


# ----------------------------------------------------------------------------- tables

def row_band(row: dict) -> str:
    tile = row.get("tile")
    if not tile:
        return "no tile"
    y = float(tile[1])
    for lo, hi, label in ROW_BANDS:
        if lo <= y < hi:
            return label
    return ROW_BANDS[-1][2] if y >= 32 else ROW_BANDS[0][2]


def rate(landed: int, failed: int) -> str:
    return "-" if landed + failed == 0 else f"{100.0 * landed / (landed + failed):.0f}%"


def group_table(rows: list[dict], keyfn, title: str, label: str) -> list[str]:
    groups: dict = defaultdict(lambda: dict.fromkeys(VERDICTS, 0))
    for r in rows:
        groups[keyfn(r)][r["verdict"]] += 1
    lines = [f"### {title}", "", f"| {label} | taps | landed | failed | unknown | landing rate |", "|---|---|---|---|---|---|"]
    for k in sorted(groups, key=lambda x: (str(x) if not isinstance(x, (int, float)) else f"{x:08.2f}")):
        g = groups[k]
        lines.append(f"| {k} | {sum(g.values())} | {g['landed']} | {g['failed']} | {g['unknown']} | "
                     f"{rate(g['landed'], g['failed'])} |")
    return lines + [""]


def still_in_hand(rows: list[dict]) -> dict:
    """How often the tapped slot still reads the tapped card at +1.5 s, overall and per verdict."""
    out = {"all": [0, 0]}
    for v in VERDICTS:
        out[v] = [0, 0]
    for r in rows:
        card = slot_card_at_1_5(r)
        if card is None:
            continue
        same = int(card == row_card(r))
        for k in ("all", r["verdict"]):
            out[k][0] += same
            out[k][1] += 1
    return out


def fmt_hand(hand) -> str:
    return "[" + ", ".join((c or "?")[:9] for c in hand) + "]" if isinstance(hand, list) else "[]"


def fmt_e(v) -> str:
    return "?" if not isinstance(v, (int, float)) else f"{v:.1f}"


def build_review(run_dir: Path, rows: list[dict], include_dry: bool, only: str, sheets: list[str]) -> str:
    taps = [r for r in rows if r["kind"] in ("tap", "dry") and "verdict" in r]
    sent = [r for r in rows if r["kind"] == "tap"]
    counts = {v: sum(1 for r in taps if r["verdict"] == v) for v in VERDICTS}
    kinds = {k: sum(1 for r in rows if r["kind"] == k) for k in ("tap", "dry", "hold", "skipped")}
    lines = [f"# Shots review: {run_dir}", "",
             f"- decisions recorded: {len(rows)} (sent taps {kinds['tap']}, dry-run taps {kinds['dry']}, "
             f"holds {kinds['hold']}, skipped {kinds['skipped']})",
             f"- verdicts over {'sent and dry-run taps' if include_dry else 'sent taps'} ({len(taps)}): "
             f"landed {counts['landed']}, failed {counts['failed']}, unknown {counts['unknown']}; "
             f"landing rate {rate(counts['landed'], counts['failed'])} (unknown left out)",
             f"- landed = elixir fell by >= cost - {DROP_TOL:g} at +0.5 s or +1.5 s (hero ability: cost - {ABILITY_DROP_TOL:g}), "
             "or the tapped slot reads another card; elixir and hand are the raw perception readings when recorded",
             f"- sheets: {len(sheets)} written ({only} taps) in {sheets[0].rsplit('/', 1)[0] if sheets else '-'}", ""]
    if not taps:
        lines += [f"No judged taps ({len(sent)} sent). A dry run needs `--include-dry`.", ""]
        return "\n".join(lines)
    lines += group_table(taps, row_card, "Landing rate by card", "card")
    lines += group_table(taps, lambda r: "?" if r.get("slot") is None else r["slot"], "Landing rate by slot", "slot")
    lines += group_table(taps, row_band, "Landing rate by target row band (tile row, own side = low rows)", "band")
    lines += group_table(taps, lambda r: r.get("mode", "?"), "Landing rate by mode", "mode")
    sih = still_in_hand(taps)
    lines += ["### Tapped card still in its slot at +1.5 s", ""]
    for k in ("all",) + VERDICTS:
        same, n = sih[k]
        lines.append(f"- {k}: {same}/{n}" + (f" ({100.0 * same / n:.0f}%)" if n else ""))
    lines += ["", "### Taps", "",
              "| idx | t | card | slot | tile | mode | verdict | elixir before > after | drop / cost | slot read +1.5s |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in taps:
        after = r.get("elixir_after_1.5") if r.get("elixir_after_1.5") is not None else r.get("elixir_after_0.5")
        tile = r.get("tile")
        drop = r.get("drop")
        lines.append(f"| {r['idx']} | {r['t']:.1f} | {row_card(r)} | {r.get('slot')} | "
                     f"{'-' if not tile else f'{tile[0]:g},{tile[1]:g}'} | {r.get('mode')} | {r['verdict']} | "
                     f"{fmt_e(r.get('elixir_before'))} > {fmt_e(after)} | "
                     f"{fmt_e(drop)} / {fmt_e(r.get('cost'))} | {slot_card_at_1_5(r) or '?'} |")
    return "\n".join(lines + [""])


# ----------------------------------------------------------------------------- contact sheets

def caption_lines(r: dict, width: int) -> list[str]:
    verdict = r.get("verdict") or ("hold" if r["kind"] == "hold" else r["kind"])
    after = r.get("elixir_after_1.5") if r.get("elixir_after_1.5") is not None else r.get("elixir_after_0.5")
    line1 = f"d{r['idx']:04d} t={r['t']:.1f}  {r['key']}  {r.get('mode')}  {verdict.upper()}  slot {r.get('slot')}"
    hand_after = r.get("hand_raw_after_1.5") if isinstance(r.get("hand_raw_after_1.5"), list) else r.get("hand_after_1.5")
    before_hand = r.get("hand_raw_before") if isinstance(r.get("hand_raw_before"), list) else r.get("hand_before")
    line2 = (f"elixir {fmt_e(r.get('elixir_before'))} > {fmt_e(after)}   hand {fmt_hand(before_hand)} > "
             f"{fmt_hand(hand_after)}   {r.get('reason') or ''}")
    cap = max(20, width // 7)                     # about 7 px per character at the caption font
    return [line1[:cap], line2[:cap]]


def _cell(path: Path, cell_w: int, cell_h: int) -> np.ndarray:
    cell = np.full((cell_h, cell_w, 3), 40, np.uint8)
    img = cv2.imread(str(path)) if path.is_file() else None
    if img is None:
        cv2.putText(cell, "no image", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        return cell
    return cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)


def make_sheet(rows: list[dict], shots_dir: Path, cell_w: int = 320) -> np.ndarray:
    """One sheet: a row per decision, three cells (a / b / c) under a two-line caption."""
    first = next((cv2.imread(str(shots_dir / f)) for r in rows for f in r.get("files") or [] if f
                  and (shots_dir / f).is_file()), None)
    aspect = first.shape[0] / first.shape[1] if first is not None else 2.0
    cell_h = int(round(cell_w * aspect))
    cap_h, gap = 40, 4
    width = 3 * cell_w + 4 * gap
    sheet = np.full((len(rows) * (cap_h + cell_h + gap) + gap, width, 3), 20, np.uint8)
    for k, r in enumerate(rows):
        y = gap + k * (cap_h + cell_h + gap)
        color = {"landed": (0, 200, 0), "failed": (60, 60, 255), "unknown": (0, 200, 255)}.get(r.get("verdict"), (200, 200, 200))
        for j, text in enumerate(caption_lines(r, width)):
            cv2.putText(sheet, text, (gap + 2, y + 15 + 17 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        files = list(r.get("files") or [None, None, None]) + [None] * 3
        for j in range(3):
            x = gap + j * (cell_w + gap)
            f = files[j]
            sheet[y + cap_h:y + cap_h + cell_h, x:x + cell_w] = _cell(shots_dir / f if f else Path("/nonexistent"),
                                                                       cell_w, cell_h)
    return sheet


def write_sheets(rows: list[dict], shots_dir: Path, out_dir: Path, per_sheet: int = 6, cell_w: int = 320) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, i in enumerate(range(0, len(rows), per_sheet), start=1):
        path = out_dir / f"sheet_{n:03d}.png"
        cv2.imwrite(str(path), make_sheet(rows[i:i + per_sheet], shots_dir, cell_w), [cv2.IMWRITE_PNG_COMPRESSION, 6])
        paths.append(str(path))
    return paths


# ----------------------------------------------------------------------------- main

def review(run_dir: Path, only: str = "all", out_dir: Path | None = None, include_dry: bool = False,
           per_sheet: int = 6, cell_w: int = 320, cards: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "shots" / "sheets"
    rows = classify(read_index(run_dir), load_cards() if cards is None else cards, include_dry)
    if only == "failed":
        shown = [r for r in rows if r.get("verdict") == "failed"]
    else:
        shown = [r for r in rows if r["kind"] != "skipped"]
    sheets = write_sheets(shown, run_dir / "shots", out_dir, per_sheet, cell_w)
    text = build_review(run_dir, rows, include_dry, only, sheets)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review.md").write_text(text)
    return {"rows": rows, "sheets": sheets, "review": str(out_dir / "review.md"), "text": text}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--only", choices=("failed", "all"), default="all", help="which decisions go on the sheets")
    ap.add_argument("--out", default=None, help="output dir (default RUN_DIR/shots/sheets)")
    ap.add_argument("--include-dry", action="store_true", help="also judge taps that were not sent (video dry run)")
    ap.add_argument("--per-sheet", type=int, default=6)
    ap.add_argument("--cell-width", type=int, default=320)
    args = ap.parse_args(argv)
    if not (Path(args.run_dir) / "shots" / "index.jsonl").is_file():
        print(f"error: {args.run_dir}/shots/index.jsonl not found (run with --record-shots)", file=sys.stderr)
        return 2
    res = review(Path(args.run_dir), args.only, Path(args.out) if args.out else None, args.include_dry,
                 args.per_sheet, args.cell_width)
    print(res["text"])
    print(f"{len(res['sheets'])} sheet(s) and review.md in {Path(res['review']).parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
