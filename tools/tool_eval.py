"""Measure local Gemma 4 tool selection. No tool is executed and no game is started.

    .venv/bin/python tools/tool_eval.py --output /tmp/tool-eval.json
    .venv/bin/python tools/tool_eval.py --only say_letter

Each case has a fresh two-message conversation. Only the tool descriptions are tuned;
the system prompt and cases stay fixed between the before and after measurements.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buddy.agent.tools import schemas  # noqa: E402

SYSTEM = """You are Buddy, a friendly robot arm playing paper games with an older adult.
Choose the tools that match what the player asks in the situation below. Use native tool calls,
including say for a spoken reply. If there are several requests, call their tools in spoken order.
Do not add an acknowledgement tool to an action. If the request is unclear, use say to ask one short question.
Never choose a game move or guess a letter for the player. The solver chooses moves and hints.
The camera decides what is on the board. A player's report does not prove a mark or sheet is valid.
The executor checks state, legality, reach and safety before doing anything. Never bypass those checks.
Use short, plain British English, without emoji.
"""
CRITICAL = ("stop", "switch_game", "say_letter")


def load_cases(path: Path) -> list[dict]:
    cases, ids = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        for key in ("id", "category", "state", "situation", "said"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Line {line_number}: missing or empty {key}")
        if case["id"] in ids:
            raise ValueError(f"Duplicate case: {case['id']}")
        ids.add(case["id"])
        offered = {t["function"]["name"]: t["function"] for t in schemas(case["state"])}
        expected = case.get("expect")
        if not isinstance(expected, list) or not expected or any(
            not isinstance(name, str) or name not in offered for name in expected
        ):
            raise ValueError(f"{case['id']}: expected tools must be offered in {case['state']}")
        args = case.get("args", {})
        if not isinstance(args, dict) or any(
            name not in expected or not isinstance(values, dict)
            or not set(values) <= set(offered[name]["parameters"]["properties"])
            for name, values in args.items()
        ):
            raise ValueError(f"{case['id']}: args must map expected tool names to parameter values")
        cases.append(case)
    if not cases:
        raise ValueError("The case file is empty")
    return cases


def request_body(case: dict) -> dict:
    return {
        "model": "gemma4", "think": False, "stream": False, "keep_alive": "60m",
        "options": {"temperature": 0, "num_predict": 256},
        "tools": schemas(case["state"]),
        "messages": [
            {"role": "system", "content": SYSTEM + "\nSituation: " + case["situation"]},
            {"role": "user", "content": case["said"]},
        ],
    }


def chat(url: str, case: dict, timeout: float) -> tuple[dict, float]:
    request = Request(url + "/api/chat", data=json.dumps(request_body(case)).encode(),
                      headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, time.perf_counter() - start


def check(case: dict, payload: dict) -> tuple[list[dict], list[str]]:
    """Strict tool order and types; optional expected arguments are exact subsets.

    Do not salvage JSON from prose, rename tools, coerce booleans, or repair calls.
    Native calls must already satisfy the offered schemas.
    """
    errors, calls = [], []
    if not isinstance(payload, dict):
        return [], ["response is not an object"]
    message = payload.get("message")
    if not isinstance(message, dict):
        return [], ["missing message"]
    raw = message.get("tool_calls")
    if not isinstance(raw, list) or not raw:
        return [], ["no native tool_calls"]
    offered = {t["function"]["name"]: t["function"] for t in schemas(case["state"])}
    for index, call in enumerate(raw):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            errors.append(f"call {index + 1}: missing function")
            continue
        name, args = function.get("name"), function.get("arguments")
        calls.append({"name": name, "arguments": args})
        if not isinstance(name, str) or name not in offered:
            errors.append(f"call {index + 1}: tool unavailable: {name!r}")
            continue
        if not isinstance(args, dict):
            errors.append(f"{name}: arguments must be an object")
            continue
        params = offered[name]["parameters"]
        if not set(params["required"]) <= args.keys():
            errors.append(f"{name}: missing required arguments")
        if not args.keys() <= params["properties"].keys():
            errors.append(f"{name}: unexpected arguments")
        for key, value in args.items():
            spec = params["properties"].get(key, {})
            kind = spec.get("type")
            if (kind == "string" and (not isinstance(value, str) or not value.strip())
                    or kind == "boolean" and type(value) is not bool):
                errors.append(f"{name}.{key}: invalid {kind}")
            if "enum" in spec and value not in spec["enum"]:
                errors.append(f"{name}.{key}: outside enum")
        for key, value in case.get("args", {}).get(name, {}).items():
            if type(args.get(key)) is not type(value) or args.get(key) != value:
                errors.append(f"{name}.{key}: expected {value!r}, got {args.get(key)!r}")
    names = [call["name"] for call in calls]
    if names != case["expect"]:
        errors.insert(0, f"expected {case['expect']}, got {names}")
    if payload.get("done") is not True or payload.get("done_reason") == "length":
        errors.append("response did not finish normally")
    return calls, errors


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def local_url(value: str) -> str:
    parsed = urlparse(value)
    host = parsed.hostname or ""
    try:
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if not local or parsed.scheme != "http" or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("OLLAMA_URL must be a local HTTP origin, such as http://127.0.0.1:11434")
    return value.rstrip("/")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "tests/data/tool_eval.jsonl")
    parser.add_argument("--only", metavar="CATEGORY", help="Run just one category")
    parser.add_argument("--output", type=Path, help="Write full results and raw responses as JSON")
    parser.add_argument("--timeout", type=float, default=60, help="Seconds per request (default: 60)")
    opts = parser.parse_args(argv)
    try:
        cases = load_cases(opts.cases)
        categories = sorted({case["category"] for case in cases})
        if opts.only:
            if opts.only not in categories:
                raise ValueError("Unknown category. Choose: " + ", ".join(categories))
            cases = [case for case in cases if case["category"] == opts.only]
        if opts.timeout <= 0:
            raise ValueError("Timeout must be positive")
        url = local_url(os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
        # A separate warm-up request excludes model loading from steady-state latency.
        # It is never appended to any evaluated conversation.
        print(f"Warming gemma4 on {url} (not scored)...", flush=True)
        _, warm_seconds = chat(url, cases[0], opts.timeout)
    except (OSError, URLError, ValueError) as exc:
        print(f"Cannot start evaluation: {exc}", file=sys.stderr)
        return 2
    print(f"Warm-up: {warm_seconds:.3f}s. {len(cases)} stateless cases.", flush=True)
    print(f"{'case':<20} {'expected':<32} {'got':<38} {'seconds':>8}  ok", flush=True)
    results = []
    for case in cases:
        start = time.perf_counter()
        try:
            payload, seconds = chat(url, case, opts.timeout)
            calls, errors = check(case, payload)
        except (OSError, URLError, ValueError) as exc:
            seconds, payload, calls, errors = time.perf_counter() - start, {}, [], [str(exc)]
        result = {"case": case, "calls": calls, "seconds": seconds,
                  "ok": not errors, "errors": errors, "response": payload}
        results.append(result)
        got = ",".join(str(call["name"]) for call in calls) or "(none)"
        print(f"{case['id']:<20} {','.join(case['expect']):<32} {got:<38} {seconds:8.3f}  "
              f"{'yes' if not errors else 'NO'}", flush=True)
        for error in errors:
            print("  " + error, flush=True)
    grouped = defaultdict(list)
    for result in results:
        grouped[result["case"]["category"]].append(result)
    summary = {}
    print("\nAccuracy by category (ordered tools and expected arguments):")
    for category, group in sorted(grouped.items()):
        passed = sum(result["ok"] for result in group)
        summary[category] = {"passed": passed, "total": len(group), "accuracy": passed / len(group)}
        print(f"  {category:<20} {passed:3}/{len(group):<3} {passed / len(group):7.1%}")
    accuracy = sum(result["ok"] for result in results) / len(results)
    seconds = [result["seconds"] for result in results]
    p50, p95 = percentile(seconds, .5), percentile(seconds, .95)
    critical = {}
    for name in CRITICAL:
        group = [result for result in results if name in result["case"]["expect"]]
        if group:
            critical[name] = {"passed": sum(result["ok"] for result in group), "total": len(group)}
    passed = accuracy >= .90 and p95 < 2.5 and all(g["passed"] == g["total"] for g in critical.values())
    print(f"\nOverall: {sum(r['ok'] for r in results)}/{len(results)} = {accuracy:.2%}")
    print(f"Latency: p50 {p50:.3f}s; p95 {p95:.3f}s; max {max(seconds):.3f}s")
    for name, group in critical.items():
        print(f"Required perfect: {name} {group['passed']}/{group['total']}")
    print(f"{'PASS' if passed else 'FAIL'}: requires >=90% overall, 100% on critical cases, p95 <2.5s.")
    if opts.output:
        schema_text = json.dumps({c["state"]: schemas(c["state"]) for c in cases}, sort_keys=True)
        report = {"measured_at": datetime.now(timezone.utc).isoformat(), "model": "gemma4",
                  "url": url, "system_prompt": SYSTEM, "only": opts.only,
                  "cases_sha256": hashlib.sha256(opts.cases.read_bytes()).hexdigest(),
                  "schemas_sha256": hashlib.sha256(schema_text.encode()).hexdigest(),
                  "warmup_seconds": warm_seconds, "accuracy": accuracy, "categories": summary,
                  "p50_seconds": p50, "p95_seconds": p95, "critical": critical,
                  "passed": passed, "results": results}
        opts.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"Report: {opts.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
