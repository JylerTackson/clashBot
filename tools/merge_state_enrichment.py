"""Fold an enrichment sidecar into a game-state jsonl.

  python3 tools/merge_state_enrichment.py <key> [<key> ...] [--keep-sidecar]

For each key the tool reads `knowledge_base/states/<key>.enrich.json`

  {"<sample id>": {"situation_read": "...", "reaction": "...", ...}, ...}

writes the `enrichment` object (and nothing else) onto the matching lines of
`knowledge_base/states/<key>.jsonl`, revalidates the file against the schema
and, on success, deletes the sidecar. Unknown ids and enrichment fields the
schema does not allow are errors: the jsonl is left untouched and the sidecar
kept. Re-running the extractor afterwards preserves what was merged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATES_DIR = ROOT / "knowledge_base" / "states"

sys.path.insert(0, str(ROOT / "tools"))
from validate_states import load_schema, validate, check_sample  # noqa: E402


def enrichment_schema(schema: dict) -> dict:
    return schema["properties"]["enrichment"]


def merge_key(key: str, schema: dict, keep_sidecar: bool) -> tuple[int, list[str]]:
    jsonl = STATES_DIR / f"{key}.jsonl"
    sidecar = STATES_DIR / f"{key}.enrich.json"
    if not jsonl.exists():
        return 0, [f"{jsonl}: missing"]
    if not sidecar.exists():
        return 0, [f"{sidecar}: missing"]
    try:
        payload = json.loads(sidecar.read_text())
    except ValueError as exc:
        return 0, [f"{sidecar}: not JSON ({exc})"]
    if not isinstance(payload, dict):
        return 0, [f"{sidecar}: expected an object keyed by sample id"]

    errors: list[str] = []
    sub = enrichment_schema(schema)
    for sid, enr in payload.items():
        errors += [f"{sidecar}[{sid}]: {e}" for e in validate(enr, sub, schema)]

    lines = [ln for ln in jsonl.read_text().splitlines() if ln.strip()]
    samples = [json.loads(ln) for ln in lines]
    by_id = {s["id"]: s for s in samples}
    for sid in payload:
        if sid not in by_id:
            errors.append(f"{sidecar}: id {sid!r} is not in {jsonl.name}")
    if errors:
        return 0, errors

    merged = 0
    for sid, enr in payload.items():
        by_id[sid]["enrichment"] = enr
        merged += 1
    for s in samples:
        errors += [f"{jsonl.name} [{s.get('id')}]: {e}" for e in check_sample(s, schema)]
    if errors:
        return 0, errors

    jsonl.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n")
    if not keep_sidecar:
        sidecar.unlink()
    return merged, []


def cli_argv(argv: list[str]) -> list[str]:
    """Keep keys that start with '-' (video ids like -V4H_YeMGGk) positional."""
    if "--" in argv:
        return argv
    opts = [a for a in argv if a in ("--keep-sidecar", "-h", "--help")]
    pos = [a for a in argv if a not in opts]
    return opts + (["--"] + pos if pos else [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="+", help="<video_id>-m<match_index> keys")
    ap.add_argument("--keep-sidecar", action="store_true", help="do not delete the sidecar on success")
    a = ap.parse_args(cli_argv(sys.argv[1:]))

    failed = False
    for key in a.keys:
        merged, errors = merge_key(key, load_schema(), a.keep_sidecar)
        for e in errors:
            print(e, file=sys.stderr)
        failed = failed or bool(errors)
        print(json.dumps({"key": key, "merged": merged, "errors": len(errors)}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
