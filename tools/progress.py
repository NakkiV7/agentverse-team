"""Progress reporting for long setup jobs (model downloads, batch runs).

Each job writes runtime/progress.json (machine-readable) and runtime/progress.txt (one-line bar). Open
http://localhost:8850/progress.html while `python tools/serve_progress.py` runs: the page refreshes by itself.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent.parent / "runtime"


def _fmt_s(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _fmt_n(n: float, unit: str) -> str:
    if unit == "B":
        for u in ("B", "KB", "MB", "GB"):
            if n < 1024 or u == "GB":
                return f"{n:.1f} {u}" if u != "B" else f"{int(n)} B"
            n /= 1024
    return f"{n:g} {unit}"


class Progress:
    def __init__(self, job: str, unit: str = ""):
        RUNTIME.mkdir(exist_ok=True)
        self.job, self.unit, self.t0 = job, unit, time.time()
        self.update(0, 0, "starting")

    def update(self, done: float, total: float, stage: str = "") -> str:
        el = time.time() - self.t0
        frac = (done / total) if total else 0.0
        eta = (el / frac - el) if frac > 0.01 else None
        n = int(frac * 30)
        bar = (f"{self.job}: [{'#' * n}{'-' * (30 - n)}] {100 * frac:5.1f}% | "
               f"{_fmt_n(done, self.unit)} / {_fmt_n(total, self.unit) if total else '?'} | "
               f"elapsed {_fmt_s(el)} | ETA {_fmt_s(eta) if eta is not None else '?'} | {stage}")
        (RUNTIME / "progress.txt").write_text(bar + "\n")
        (RUNTIME / "progress.json").write_text(json.dumps(
            {"job": self.job, "percent": round(100 * frac, 1), "bar": bar, "stage": stage, "done": False,
             "updated": time.time()}))
        return bar

    def finish(self, stage: str = "done") -> None:
        bar = f"{self.job}: [{'#' * 30}] 100.0% | elapsed {_fmt_s(time.time() - self.t0)} | {stage}"
        (RUNTIME / "progress.txt").write_text(bar + "\n")
        (RUNTIME / "progress.json").write_text(json.dumps(
            {"job": self.job, "percent": 100.0, "bar": bar, "stage": stage, "done": True, "updated": time.time()}))
