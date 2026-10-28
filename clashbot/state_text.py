"""Deterministic text views of a clean game state (D-015). Same code for dataset building and the live loop."""
from __future__ import annotations


def fmt_num(x) -> str:
    if x is None:
        return "?"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


def unit_token(u: dict) -> str:
    if u.get("card"):
        return u["card"] + ("*" if u.get("is_evo") else "")
    return u["unit_class"]


def render_threats(units: list[dict]) -> list[str]:
    out = []
    for u in units:
        eta = u.get("eta_tower")
        if u.get("side") != "enemy" or not isinstance(eta, dict):
            continue
        secs = eta.get("s")
        if not str(eta.get("tower", "")).startswith("own") or secs is None or secs > 8.0:
            continue
        head = u.get("heading") or "approaching"
        out.append(f"{unit_token(u)}(e) {head}, tower in {secs}s")
    return out


def render_context(state: dict, t: float | None = None) -> str:
    own, opp = state["own"], state["opponent"]
    hand = ", ".join(x or "?" for x in own["hand"])
    status = "" if own.get("deck_status", "valid") == "valid" else f" [deck {own['deck_status']}]"
    opp_deck = ", ".join(opp["deck_known"]) if opp.get("deck_known") else "unknown"
    line1 = (f"[{state['phase'] or 'unknown'}|{state['clock'] or '?'}] "
             f"elixir {fmt_num(own['elixir'])} (opp ~{fmt_num(opp['elixir_estimate'])}). "
             f"hand: {hand} (next {own.get('next_card') or '?'}). "
             f"deck: {', '.join(own['deck']) if own['deck'] else 'unknown'}{status}. "
             f"opp deck: {opp_deck}")
    field = []
    for u in state["units"]:
        s = f"{unit_token(u)}({'a' if u['side'] == 'ally' else 'e'})@[{fmt_num(u['tile'][0])},{fmt_num(u['tile'][1])}]"
        if u.get("heading"):
            s += f" {u['heading']}"
        eta = u.get("eta_tower")
        if isinstance(eta, dict) and eta.get("tower") is not None:
            s += f" eta {eta['tower']} {fmt_num(eta.get('s'))}s"
        field.append(s)
    line2 = "field: " + ("; ".join(field) if field else "none")
    line3 = "threats: " + ("; ".join(state["threats"]) if state["threats"] else "none")
    line4_items = []
    for tag, plays in (("own", own["recent_plays"]), ("opp", opp["recent_plays"])):
        for p in plays:
            tile = f"@[{fmt_num(p['tile'][0])},{fmt_num(p['tile'][1])}]" if p.get("tile") else ""
            ago = f" {round(t - p['t'], 1)}s ago" if t is not None else ""
            line4_items.append(f"{tag} {p['card'] or '?'}{tile}{ago}")
    line4 = "recent: " + ("; ".join(line4_items) if line4_items else "none")
    ot, et = own["towers"], opp["towers"]
    line5 = (f"towers own {fmt_num(ot['left'])}/{fmt_num(ot['right'])}/{fmt_num(ot['king'])}, "
             f"enemy {fmt_num(et['left'])}/{fmt_num(et['right'])}/{fmt_num(et['king'])}")
    return "\n".join([line1, line2, line3, line4, line5])


def render_action(action: dict) -> str:
    if action["type"] == "play":
        tile = action.get("tile")
        where = f" at [{fmt_num(tile[0])},{fmt_num(tile[1])}]" if tile else ""
        placement = f" ({action.get('zone') or '?'} {action.get('lane') or '?'})" if tile else ""
        delay = f" after {fmt_num(action.get('delay_s'))}s" if action.get("delay_s") is not None else ""
        return f"action: play {action.get('card') or '?'}{where}{placement}{delay}"
    return f"action: {action['type']}"


def render_state_text(state: dict, action: dict, t: float | None = None) -> str:
    return render_context(state, t) + "\n" + render_action(action)
