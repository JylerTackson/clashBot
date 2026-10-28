"""Shared markdown helpers for the knowledge-graph and document builders (WP2)."""
from __future__ import annotations

import re

_FM_LINE = re.compile(r"^([A-Za-z0-9_]+):\s*(.*)$")
_CREATOR = re.compile(
    r"<!-- creator-insights:ryleycr1:(?P<game>[^:]+):start -->\n?(?P<body>.*?)"
    r"<!-- creator-insights:ryleycr1:(?P=game):end -->",
    re.S,
)

FIXED_ALIASES = {
    "pekka": "p-e-k-k-a",
    "mini pekka": "mini-p-e-k-k-a",
    "e-spirit": "electro-spirit",
    "e spirit": "electro-spirit",
    "ewiz": "electro-wizard",
    "e-wiz": "electro-wizard",
    "x bow": "x-bow",
}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    lines = text.split("\n")
    end = None
    for i in range(1, len(lines)):
        if lines[i] == "---":
            end = i
            break
    if end is None:
        return {}, text
    fm: dict = {}
    for line in lines[1:end]:
        m = _FM_LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if value.startswith("[") and value.endswith("]"):
            fm[key] = [x.strip().strip('"') for x in value[1:-1].split(",") if x.strip()]
        else:
            if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            fm[key] = value
    return fm, "\n".join(lines[end + 1:])


def section(body: str, heading: str) -> str | None:
    lines = body.split("\n")
    target = "## " + heading
    start = None
    for i, line in enumerate(lines):
        if line == target:
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end])


def bullets(section_text: str) -> list[str]:
    out: list[str] = []
    current: str | None = None
    for line in section_text.split("\n"):
        if line.startswith("- "):
            if current is not None:
                out.append(current.strip())
            current = line[2:]
        elif current is not None and line.startswith("  ") and line.strip():
            current += " " + line.strip()
        else:
            if current is not None:
                out.append(current.strip())
            current = None
    if current is not None:
        out.append(current.strip())
    return out


def _cells(row: str) -> list[str]:
    parts = row.split("|")
    return [c.strip() for c in parts[1:-1]]


def first_table(section_text: str) -> list[dict]:
    run: list[str] = []
    for line in section_text.split("\n"):
        if line.startswith("|"):
            run.append(line)
        elif run:
            break
    if len(run) < 2:
        return []
    header = _cells(run[0])
    rows = []
    for line in run[2:]:
        cells = _cells(line)
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def creator_blocks(text: str) -> list[tuple[str, str]]:
    return [(m.group("game"), m.group("body").strip()) for m in _CREATOR.finditer(text)]


def build_aliases(cards: dict) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for slug, card in cards.items():
        name = card.get("name") or ""
        pairs.add((name.lower(), slug))
        pairs.add((slug.replace("-", " "), slug))
        if "." in name:
            pairs.add((name.lower().replace(".", "").strip(), slug))
    for alias, slug in FIXED_ALIASES.items():
        if slug in cards:
            pairs.add((alias, slug))
    kept = [(a, s) for a, s in pairs if len(a) >= 3]
    return sorted(kept, key=lambda p: (-len(p[0]), p[0]))


def mentions(text: str, aliases: list[tuple[str, str]], exclude: str | None = None) -> list[str]:
    low = text.lower()
    consumed = [False] * len(low)
    found: list[tuple[int, str]] = []
    for alias, slug in aliases:
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", low):
            if not any(consumed[m.start():m.end()]):
                for k in range(m.start(), m.end()):
                    consumed[k] = True
                found.append((m.start(), slug))
    found.sort(key=lambda x: x[0])
    out: list[str] = []
    for _, slug in found:
        if slug not in out and slug != exclude:
            out.append(slug)
    return out
