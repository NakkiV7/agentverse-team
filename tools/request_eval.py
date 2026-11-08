"""Score local Gemma paper routines exactly, without moving the arm.

    .venv/bin/python tools/request_eval.py --output /tmp/request-eval.json
    .venv/bin/python tools/request_eval.py --only alphabet

Exit 0: >=85% exact structures. Exit 1: below target. Exit 2: invalid cases.
Latency and machine load are reported, never acceptance gates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buddy.agent import requests  # noqa: E402


def action_structure(action: dict) -> dict:
    result = {"op": action["op"]}
    if "place" in action:
        result["place"] = action["place"]
    fields = {"text": action["text"]} if "text" in action else {}
    if "chars" in action.get("target", {}):
        fields["chars"] = action["target"]["chars"]
    result["templates"] = {key: sorted(set(requests.TEMPLATE_RE.findall(value)))
                           for key, value in fields.items()}
    return result


def structure(result: requests.Routine | requests.Refusal) -> dict:
    if isinstance(result, requests.Refusal):
        return {"refusal": True}
    return {"now": [action_structure(a) for a in result.now],
            "rules": [{"when": rule["when"], "do": [action_structure(a) for a in rule["do"]]}
                      for rule in result.rules], "cancel": result.cancel}


def check(case: dict, result: requests.Routine | requests.Refusal, raw: str | None) -> list[str]:
    if isinstance(result, requests.Refusal):
        # A timeout, malformed JSON or validator rejection is never credited as
        # a correctly understood impossible request. Require a model refusal.
        try:
            value = json.loads(raw) if raw is not None else None
        except (ValueError, TypeError):
            value = None
        if (not isinstance(value, dict) or set(value) != {"reason", "say"}
                or value != asdict(result) or requests.parse(raw) != result):
            return [f"compiler refusal: {result.reason}"]
    actual = structure(result)
    if json.dumps(actual, sort_keys=True) != json.dumps(case["expect"], sort_keys=True):
        return ["expected " + json.dumps(case["expect"], sort_keys=True),
                "got " + json.dumps(actual, sort_keys=True)]
    # Optional literal checks make current-letter selection and supplied names
    # testable too, without requiring exact spoken phrasing or generated IDs.
    if isinstance(result, requests.Routine):
        for item in case.get("literals", []):
            action = result.now[item["action"]]
            actual_text = action.get("text") if item["field"] == "text" else action.get("target", {}).get("chars")
            if actual_text != item["value"]:
                return [f"now[{item['action']}].{item['field']}: expected {item['value']!r}, got {actual_text!r}"]
    return []


def load_cases(path: Path) -> list[dict]:
    cases, ids = [], set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise ValueError(f"line {number}: expected an object")
        for key in ("id", "category", "request", "facts", "state"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"line {number}: missing {key}")
        if case["id"] in ids:
            raise ValueError(f"duplicate id: {case['id']}")
        ids.add(case["id"])
        expected = case.get("expect")
        if not isinstance(expected, dict) or (expected != {"refusal": True}
                and set(expected) != {"now", "rules", "cancel"}):
            raise ValueError(f"line {number}: invalid expected structure")
        cases.append(case)
    if not cases:
        raise ValueError("empty cases")
    return cases


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def run_case(case: dict) -> dict:
    answers = []
    chat = requests._chat

    def capture(*args):
        answer = chat(*args)
        answers.append(answer)
        return answer

    start = time.perf_counter()
    # Observe the answer without altering the production request or deadline.
    with patch.object(requests, "_chat", capture):
        result = requests.compile(case["request"], case["facts"], case["state"])
    seconds = time.perf_counter() - start
    raw = answers[0] if answers else None
    errors = check(case, result, raw)
    return {"case": case, "ok": not errors, "errors": errors, "seconds": seconds,
            "result_type": type(result).__name__, "result": asdict(result), "structure": structure(result), "raw": raw}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "tests/data/requests.jsonl")
    parser.add_argument("--only", help="Run one category (not a full-suite measurement)")
    parser.add_argument("--output", type=Path, help="Save raw answers, results, timings and input hashes")
    args = parser.parse_args(argv)
    try:
        cases = load_cases(args.cases)
        if args.only:
            cases = [case for case in cases if case["category"] == args.only]
            if not cases:
                raise ValueError("unknown category")
    except (OSError, ValueError) as exc:
        print(f"Cannot evaluate: {exc}", file=sys.stderr)
        return 2
    load_start = os.getloadavg()
    print(f"{len(cases)} cases; {requests.C.TEACH_MODEL}; deadline {requests.TIMEOUT:g}s; load {load_start}", flush=True)
    print("Sequential, no warm-up excluded; shared Ollama latency is informational.", flush=True)
    results = []
    for case in cases:
        result = run_case(case)
        results.append(result)
        print(f"{case['id']:<24} {'PASS' if result['ok'] else 'FAIL'} {result['seconds']:7.3f}s", flush=True)
        for error in result["errors"]:
            print("  " + error, flush=True)
    grouped = defaultdict(list)
    for result in results:
        grouped[result["case"]["category"]].append(result)
    for category, group in sorted(grouped.items()):
        print(f"{category:<18} {sum(r['ok'] for r in group)}/{len(group)}", flush=True)
    passed = sum(r["ok"] for r in results)
    accuracy = passed / len(results)
    seconds = [r["seconds"] for r in results]
    p50, p95 = percentile(seconds, .5), percentile(seconds, .95)
    load_end = os.getloadavg()
    print(f"Overall: {passed}/{len(results)} = {accuracy:.2%}; target >=85%", flush=True)
    print(f"Latency (not gated): p50 {p50:.3f}s; p95 {p95:.3f}s; max {max(seconds):.3f}s; load {load_end}", flush=True)
    if args.output:
        report = {"measured_at": datetime.now(timezone.utc).isoformat(), "model": requests.C.TEACH_MODEL,
                  "url": requests.C.OLLAMA_URL, "only": args.only, "accuracy": accuracy, "passed": passed,
                  "total": len(results), "p50_seconds": p50, "p95_seconds": p95, "max_seconds": max(seconds),
                  "wall_seconds": sum(seconds), "load_start": load_start, "load_end": load_end,
                  "latency_gated": False, "deadline_seconds": requests.TIMEOUT,
                  "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                  "prompt_sha256": hashlib.sha256(requests.SYSTEM.encode()).hexdigest(), "results": results}
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"Report: {args.output}")
    return 0 if accuracy >= .85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
