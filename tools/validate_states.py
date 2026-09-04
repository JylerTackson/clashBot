"""Validate game-state sample files against knowledge_base/meta/game_state_schema.json.

  python3 tools/validate_states.py [paths ...] [--summary]

No third-party packages: the validator below covers exactly the JSON Schema
subset the state schema uses (type, enum, const, required, properties,
additionalProperties, items, minItems/maxItems, minimum/maximum, pattern,
$ref/$defs). Beyond the schema it checks that sample ids are unique across the
files given and that `state_text` starts with the template's first line.
Exits non-zero if anything fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "knowledge_base" / "meta" / "game_state_schema.json"
STATES_DIR = ROOT / "knowledge_base" / "states"

# "[single_elixir|2:31] elixir 5 (opp ~10). hand: a, b, c, d (next e). deck: ..."
STATE_TEXT_HEAD = re.compile(
    r"^\[[a-z_]+\|[^\]]*\] elixir \S+ \(opp ~[^)]*\)\. hand: .+ \(next [^)]*\)\. deck: ")

TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is_type(value, name: str) -> bool:
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "null":
        return value is None
    raise ValueError(f"unsupported type {name!r}")


def resolve(schema: dict, root: dict) -> dict:
    seen = 0
    while "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise ValueError(f"unsupported $ref {ref!r}")
        node = root
        for part in ref[2:].split("/"):
            node = node[part]
        schema = node
        seen += 1
        if seen > 10:
            raise ValueError("$ref loop")
    return schema


def validate(instance, schema: dict, root: dict | None = None, path: str = "") -> list[str]:
    """Errors as 'path: message'; empty list means valid."""
    root = root if root is not None else schema
    schema = resolve(schema, root)
    err: list[str] = []
    p = path or "$"

    if "const" in schema and instance != schema["const"]:
        err.append(f"{p}: expected const {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        err.append(f"{p}: {instance!r} not in enum {schema['enum']}")
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(instance, n) for n in names):
            err.append(f"{p}: expected type {'|'.join(names)}, got {type(instance).__name__}")
            return err

    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                err.append(f"{p}: missing required property {key!r}")
        extra = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                err += validate(value, props[key], root, f"{p}.{key}")
            elif extra is False:
                err.append(f"{p}: additional property {key!r} not allowed")
            elif isinstance(extra, dict):
                err += validate(value, extra, root, f"{p}.{key}")
    elif isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, value in enumerate(instance):
                err += validate(value, items, root, f"{p}[{i}]")
        if "minItems" in schema and len(instance) < schema["minItems"]:
            err.append(f"{p}: {len(instance)} items < minItems {schema['minItems']}")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            err.append(f"{p}: {len(instance)} items > maxItems {schema['maxItems']}")
    elif isinstance(instance, str):
        pat = schema.get("pattern")
        if pat and not re.search(pat, instance):
            err.append(f"{p}: {instance!r} does not match pattern {pat!r}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            err.append(f"{p}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            err.append(f"{p}: {instance} > maximum {schema['maximum']}")
    return err


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def check_sample(sample: dict, schema: dict) -> list[str]:
    err = validate(sample, schema)
    text = sample.get("state_text")
    if isinstance(text, str) and not STATE_TEXT_HEAD.match(text.splitlines()[0] if text else ""):
        err.append("$.state_text: first line does not match the template")
    if sample.get("kind") == "key" and not sample.get("key_moment"):
        err.append("$.key_moment: required for kind=key")
    return err


def validate_file(path: Path, schema: dict, seen_ids: dict[str, str]) -> tuple[list[str], list[dict]]:
    errors: list[str] = []
    samples: list[dict] = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except ValueError as exc:
            errors.append(f"{path}:{n}: not JSON ({exc})")
            continue
        samples.append(sample)
        sid = sample.get("id")
        if sid in seen_ids:
            errors.append(f"{path}:{n}: duplicate id {sid!r} (also in {seen_ids[sid]})")
        elif isinstance(sid, str):
            seen_ids[sid] = f"{path}:{n}"
        errors += [f"{path}:{n}: {e}" for e in check_sample(sample, schema)]
    return errors, samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="jsonl files (default: knowledge_base/states/*.jsonl)")
    ap.add_argument("--summary", action="store_true", help="print counts by kind and enrichment coverage")
    a = ap.parse_args()

    paths = [Path(p) for p in a.paths] or sorted(STATES_DIR.glob("*.jsonl"))
    if not paths:
        print("no state files found", file=sys.stderr)
        return 1
    schema = load_schema()
    seen_ids: dict[str, str] = {}
    errors: list[str] = []
    counts = {"key": 0, "play": 0, "periodic": 0}
    enriched = {"key": 0, "play": 0, "periodic": 0}
    total = 0
    for path in paths:
        if not path.exists():
            errors.append(f"{path}: missing")
            continue
        errs, samples = validate_file(path, schema, seen_ids)
        errors += errs
        for s in samples:
            total += 1
            k = s.get("kind")
            if k in counts:
                counts[k] += 1
                if s.get("enrichment"):
                    enriched[k] += 1
    for e in errors[:50]:
        print(e, file=sys.stderr)
    if len(errors) > 50:
        print(f"... and {len(errors) - 50} more errors", file=sys.stderr)
    if a.summary:
        print(json.dumps({"files": len(paths), "samples": total, "by_kind": counts,
                          "enriched": enriched,
                          "enriched_pct": round(100.0 * sum(enriched.values()) / total, 1) if total else 0.0,
                          "errors": len(errors)}))
    elif not errors:
        print(f"ok: {total} samples in {len(paths)} file(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
