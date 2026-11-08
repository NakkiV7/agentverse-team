"""Run validated paper actions and standing rules through the brain's worker.

``run`` is synchronous: call it from a separate orchestration thread, never
from a busy brain drawing worker or its input loop. ``on_event`` only enqueues
and is safe inside ``brain.emit``. Each row (at most six characters per job)
releases the drawing worker so game moves and switches can run between jobs. Install a
routine's rules before starting ``run`` to capture guesses made during writing.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import queue
import string
import threading
import time

import cv2
import numpy as np

from .. import config as C
from ..game import marks as M
from ..perception.board import mm_to_rect_px, rectify
from ..perception.sheet import check_lines
from . import layout as L

OPS = {"write_text", "append_text", "cross_out", "underline", "circle", "tick"}
KINDS = {"heard", "said", "game_start", "move", "guess", "line", "box", "game_over", "switch", "state"}
TEMPLATES = {"letter", "correct", "misses", "pattern", "wrong", "guessed", "score_player",
             "score_buddy", "result", "game", "player_name", "number", "cells", "name"}
# The narrow diagonal corridor excludes neighbouring glyph ink. Missing ink
# gets one retry; acceptance scripts still require LINE_OK_COVERAGE (90%).
MARK_OK_COVERAGE = .8
MARK_HALF_WIDTH_MM = .65
GLYPH_MIN_INK = .025
MAX_WRITE_GROUP = 6


def writing_groups(characters):
    """Keep reading order, row boundaries and a six-character response bound."""
    group = []
    for cell in characters:
        if group and (cell.box.y0 != group[-1].box.y0 or len(group) == MAX_WRITE_GROUP):
            yield group
            group = []
        group.append(cell)
    if group:
        yield group


def context_from(brain):
    """Only public game facts; never expose the secret hangman answer."""
    out = {"game": getattr(brain, "game_name", "the game"), "number": getattr(brain, "games", 0)}
    if getattr(brain, "player_name", None):
        out["player_name"] = brain.player_name
    hangman = getattr(brain, "hangman", None) if L.game_kind(brain) == "hangman" else None
    if hangman is not None:
        out.update(pattern=hangman.pattern, misses=hangman.misses, guessed=sorted(hangman.guessed),
                   wrong=sorted(hangman.guessed - set(hangman.word)))
        if getattr(hangman, "last_guess", None):
            out.update(letter=hangman.last_guess, correct=hangman.last_guess in hangman.word)
        if hangman.result() is not None:
            out["result"] = hangman.result()
    dots = getattr(brain, "dots", None) if L.game_kind(brain) == "dots" else None
    if dots is not None and dots.state is not None:
        out.update(score_player=dots.state.score("P"), score_buddy=dots.state.score("B"))
    # Event history is part of brain state too. Supply the last move/line and
    # a completed result without carrying facts across a game boundary.
    for ev in reversed(list(getattr(brain, "events", ()))):
        if ev.kind in ("switch", "game_start"):
            break
        if ev.kind == "game_over":
            if getattr(brain, "state", None) == "GAME_OVER" and ev.data.get("game") == out["game"]:
                if "result" in ev.data:
                    out.setdefault("result", ev.data["result"])
            break
        if ev.kind == "move" and dots is None and hangman is None and "cells" in ev.data:
            out.setdefault("cells", ev.data["cells"])
        if ev.kind == "line" and dots is not None and "name" in ev.data:
            out.setdefault("name", ev.data["name"])
    return out


def _text(value):
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(x) for x in (sorted(value) if isinstance(value, set) else value))
    return str(value)


def fill_template(text, context):
    if not isinstance(text, str):
        raise ValueError("text and character selections must be strings")
    pieces = []
    for literal, name, spec, conversion in string.Formatter().parse(text):
        pieces.append(literal)
        if name is not None:
            if name not in TEMPLATES or name not in context or context[name] is None or spec or conversion:
                raise ValueError(f"I don't know the template {{{name}}}")
            pieces.append(_text(context[name]))
    return "".join(pieces)


def matches(when, ev):
    if not isinstance(when, dict) or when.get("kind") != ev.kind:
        return False
    actual = {**ev.data, "kind": ev.kind, "who": ev.who}
    # JSON true is not the number 1, even though Python compares them equal.
    return all(k in actual and type(actual[k]) is type(v) and actual[k] == v for k, v in when.items())


@dataclass
class PaperItem:
    id: str
    text: str
    place: str
    size_mm: float
    region: L.Rect
    characters: list[L.Character]
    marks: set[str] = field(default_factory=set)
    complete: bool = False

    @property
    def boxes(self):
        return [c.box for c in self.characters]


@dataclass
class _Request:
    actions: list
    context: dict
    generation: int
    rule: dict | None = None
    done: threading.Event = field(default_factory=threading.Event)
    error: str | None = None


class Paper:
    def __init__(self, brain):
        self.brain, self.app = brain, brain.app
        self.items: dict[str, PaperItem] = {}
        self.rules: list[dict] = []
        self._recipes: dict[str, dict] = {}
        self._queue = queue.Queue()
        self._lock = threading.RLock()
        self._generation = 0
        self._missing: set[str] = set()
        self._active = False
        self.drawing = False
        self._guesses = deque()
        self._closed = False
        self.last_error = None
        self._thread = threading.Thread(target=self._loop, name="paper-routines", daemon=True)
        self._thread.start()

    @property
    def pending(self):
        return self._active or self._queue.unfinished_tasks > 0

    def _trace(self, message):
        self.app.trace("paper", message)

    def run(self, actions, context=None):
        if not isinstance(actions, list):
            return "I need a list of paper actions."
        if self._closed:
            return "The paper worker has stopped."
        request = _Request(deepcopy(actions), {**context_from(self.brain), **(context or {})}, self._generation)
        self._queue.put(request)
        request.done.wait()
        return request.error

    def add_rule(self, rule):
        if (not isinstance(rule, dict) or not isinstance(rule.get("name"), str) or not rule["name"].strip()
                or not isinstance(rule.get("when"), dict) or rule["when"].get("kind") not in KINDS
                or not isinstance(rule.get("do"), list) or not rule["do"]):
            raise ValueError("a rule needs a name, an event kind and a list of actions")
        if any(not isinstance(a, dict) or a.get("op") not in OPS for a in rule["do"]):
            raise ValueError("the rule contains an unsupported paper action")
        with self._lock:
            self.rules[:] = [r for r in self.rules if r["name"] != rule["name"]]
            self.rules.append(deepcopy(rule))

    def cancel(self, name):
        """Cancel by exact name, operation or event kind (also selector dicts)."""
        def selected(rule):
            if isinstance(name, dict):
                return (("name" not in name or rule["name"] == name["name"])
                        and ("kind" not in name or rule["when"]["kind"] == name["kind"])
                        and ("op" not in name or any(a["op"] == name["op"] for a in rule["do"])))
            return (name == rule["name"] or name == rule["when"]["kind"]
                    or any(name == a["op"] for a in rule["do"]))
        with self._lock:
            before = len(self.rules)
            self.rules[:] = [r for r in self.rules if not selected(r)]
            return before - len(self.rules)

    def new_sheet(self):
        """Coordinator calls this on fresh-sheet setup/restore; rules survive."""
        with self._lock:
            self._generation += 1
            self._missing.update(self.items)
            self.items.clear()
            self._guesses.clear()
        self._trace("new sheet: cleared written items, kept standing rules")

    def needs_rewrite(self, item_id):
        with self._lock:
            return item_id in self._missing

    def rewrite(self, item_id, context=None):
        """Restore saved text on a fresh sheet, with new layout and no marks.

        Like run(), call outside the brain drawing worker/input loop. Reapply
        any wanted marks from the current game with run() after this returns.
        """
        with self._lock:
            if item_id in self.items:
                return f"{item_id} is already on this sheet."
            recipe = deepcopy(self._recipes.get(item_id))
        if recipe is None:
            return f"I don't have saved writing for {item_id}; please tell me what to write."
        return self.run([recipe], context)

    def on_event(self, ev):
        if self._closed:
            return
        # Our own progress announcement must not recursively trigger a
        # standing "when Penny says something, write ..." instruction.
        if ev.kind == "said" and threading.current_thread() is self._thread:
            return
        if ev.kind in ("switch", "game_start"):
            self.new_sheet()
        with self._lock:
            for rule in self.rules:
                if matches(rule["when"], ev):
                    context = {**context_from(self.brain), **ev.data}
                    if isinstance(ev.data.get("score"), dict):
                        context.update({f"score_{k}": v for k, v in ev.data["score"].items()})
                    self._queue.put(_Request(deepcopy(rule["do"]), context, self._generation, rule))

    def defer_guess(self, understanding):
        """Call before Brain._hangman_act's busy guard; True means queued.

        The coordinator adds this one guard, so spoken and typed tool/fallback
        guesses share the same behaviour without replacing brain methods.
        """
        if self.drawing and getattr(understanding, "intent", None) == "say_letter":
            with self._lock:
                self._guesses.append(understanding)
            self._trace(f"queued guess {understanding.letter} until this writing job is checked")
            return True
        return False

    def _wait_idle(self):
        def game_pending():
            state = getattr(self.brain, "state", "")
            if state in ("ROBOT_TURN", "DOTS_ROBOT_TURN"):
                return True
            dots = getattr(self.brain, "dots", None)
            # Dots commits its state and emits a line before its last camera
            # look and the robot worker launch. Respect that reserved turn.
            return (state == "DOTS_PLAYER_TURN" and dots is not None and dots.state is not None
                    and dots.state.to_move == "B")

        while (self.brain.busy or getattr(getattr(self.app, "ctl", None), "busy", False)
               or game_pending()):
            if self._closed or not getattr(self.app, "running", True):
                raise ValueError("the drawing session has stopped")
            time.sleep(.02)

    def _boundary(self, generation):
        self._wait_idle()
        if getattr(self.brain, "switch_to", None) is not None:
            self.brain._maybe_switch()
        if generation != self._generation or self._closed:
            raise ValueError("the sheet or game changed before the writing finished")
        while self._guesses:
            with self._lock:
                guess = self._guesses.popleft()
            self.brain._hangman_act(guess)
            self._wait_idle()

    def _loop(self):
        while True:
            request = self._queue.get()
            if request is None:
                self._queue.task_done()
                return
            self._active = True
            failures = []
            try:
                for action in request.actions:
                    with self._lock:
                        cancelled = request.rule is not None and not any(r is request.rule for r in self.rules)
                    if cancelled or request.generation != self._generation:
                        break
                    try:
                        self._boundary(request.generation)
                        with self._lock:
                            if request.rule is not None and not any(r is request.rule for r in self.rules):
                                break
                        self._action(action, request.context, request.generation, request.rule is not None)
                    except (ValueError, KeyError, TypeError) as exc:
                        failures.append(str(exc).rstrip("."))
                        self._trace(f"skipped {action!r}: {exc}")
                if failures:
                    request.error = failures[0][0].upper() + failures[0][1:] + "."
                    self.last_error = request.error
            except Exception as exc:
                request.error = f"I could not finish the paper routine: {exc}."
                self.last_error = request.error
                self._trace(request.error)
            finally:
                self._active = False
                self._queue.task_done()
                request.done.set()

    def _look(self):
        look = self.app.look(timeout=20)
        if (look is None or look.sheet is None or look.sheet.cut_off or look.sheet.confidence < .6
                or look.track is None or not look.track.visible or not look.track.stable
                or not self.app.view_clear()):
            raise ValueError("I cannot see a clear, steady sheet to check the writing")
        return look

    def _reachable(self, pose, strokes):
        points = [tuple(p) for stroke in strokes for p in stroke]
        ok, why = self.app.points_reachable(pose, points)
        if not ok:
            raise ValueError(f"I cannot draw there because {why}")

    def _regions(self, pose, exclude=None):
        with self._lock:
            occupied = [c.box.padded(.5) for key, item in self.items.items() if key != exclude
                        for c in item.characters if c.char != " "]
        cache = {}

        def reachable(points):
            # Candidate rectangles share many corners; avoid repeating IK.
            for p in points:
                key = tuple(round(x, 4) for x in p)
                if key not in cache:
                    cache[key] = self.app.points_reachable(pose, [p])
                if not cache[key][0]:
                    return cache[key]
            return True, "reachable"
        return L.places(self.brain, pose, reachable, occupied)

    def _choose(self, text, place, size, pose):
        choices = self._regions(pose)
        if place not in choices:
            raise ValueError(f"I don't know the place {place!r}; try the top right corner")

        def fit(where):
            for rect in choices[where]:
                try:
                    cells = L.layout_text(text, rect, size)
                    self._reachable(pose, [s for c in cells for s in c.strokes()])
                    return rect, cells
                except ValueError:
                    pass
            return None
        selected = fit(place)
        if selected is not None:
            return selected
        other = next((p for p in L.PLACES if p != place and fit(p) is not None), None)
        suggestion = f"the {other} would work" if other else "please move the sheet closer or ask for less text"
        raise ValueError(f"There isn't enough free space within reach in the {place}; {suggestion}")

    def _append_refusal(self, item, text, pose, why):
        for place, regions in self._regions(pose).items():
            for region in regions:
                try:
                    cells = L.layout_text(text, region, item.size_mm)
                    self._reachable(pose, [s for cell in cells for s in cell.strokes()])
                except ValueError:
                    continue
                return ValueError(f"I cannot append to {item.id} because {why}; "
                                  f"the {place} would work for a new note")
        return ValueError(f"I cannot append to {item.id} because {why}; "
                          "please move the sheet closer or use a fresh sheet")

    @staticmethod
    def glyph_ink(look, app, cell):
        rect = rectify(look.rgb, app.cam, look.track.pose)
        a, b = mm_to_rect_px([(cell.glyph.x0 - .5, cell.glyph.y1 + .5),
                              (cell.glyph.x1 + .5, cell.glyph.y0 - .5)])
        crop = rect[max(0, int(a[1])):int(np.ceil(b[1])), max(0, int(a[0])):int(np.ceil(b[0]))]
        return float((cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) < 140).mean()) if crop.size else 0.0

    def _draw_job(self, name, strokes, generation, check):
        """One arm job and one fresh verification frame, then yield to guesses."""
        self._boundary(generation)
        done = threading.Event()
        errors, results = [], []

        def draw():
            try:
                look = self._look()
                self._reachable(look.track.pose, strokes)
                job = self.app.draw(name, strokes)
                while not job.done.wait(.2):
                    if self._closed or not getattr(self.app, "running", True):
                        raise ValueError("the drawing session has stopped")
                if not job.ok:
                    raise ValueError(f"I could not finish that stroke: {job.error}")
                results.append(check(self._look()))
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        # Mirror the existing brain entry points: reserve busy before spawning.
        self.brain.busy = True
        self.drawing = True
        try:
            self.brain._worker(draw)
            done.wait()
            self._wait_idle()
        finally:
            self.drawing = False
        # Also drain a guess made during the final job, even if there is no
        # following action. A faint character's retry is a separate boundary.
        self._boundary(generation)
        if errors:
            raise ValueError(str(errors[0]))
        return results[0]

    def _check_characters(self, name, look, cells):
        failed = []
        for cell in cells:
            coverage = check_lines(look.rgb, self.app.cam, look.track.pose, M.stroke_segments(cell.strokes()),
                                   half_width_mm=1.5)
            ink = self.glyph_ink(look, self.app, cell)
            self._trace(f"{name}: {cell.char} [{cell.index}], {len(coverage)} segments, "
                        f"minimum coverage {min(coverage, default=1):.1%}, glyph ink {ink:.1%}")
            # Preserve the tiny-glyph criterion: segment evidence is measured
            # for every character, while its ink crop decides faintness.
            if ink >= GLYPH_MIN_INK:
                cell.written = True
            else:
                failed.append(cell)
        return failed

    def _draw_checked(self, name, strokes, generation, cell=None, cells=None):
        if not strokes:
            return
        characters = cells if cells is not None else ([cell] if cell is not None else None)
        if characters is not None:
            characters = [c for c in characters if c.char != " "]
            failed = self._draw_job(name, strokes, generation,
                                    lambda look: self._check_characters(name, look, characters))
            missing = []
            for character in failed:
                self._trace(f"faint ink: retrying {character.char} [{character.index}] once")
                retry_name = f"{name}: retry {character.char} [{character.index}]"
                missing.extend(self._draw_job(
                    retry_name, character.strokes(), generation,
                    lambda look, c=character, n=retry_name: self._check_characters(n, look, [c])))
            if missing:
                raise ValueError("I could not see the ink clearly after one retry")
            return

        def check_mark(look):
            coverage = check_lines(look.rgb, self.app.cam, look.track.pose, M.stroke_segments(strokes),
                                   half_width_mm=MARK_HALF_WIDTH_MM)
            self._trace(f"{name}: {len(coverage)} segments, minimum coverage {min(coverage, default=1):.1%}")
            return min(coverage, default=1) >= MARK_OK_COVERAGE

        for attempt in range(2):
            if self._draw_job(name, strokes, generation, check_mark):
                return
            if not attempt:
                self._trace("faint ink: retrying once")
        raise ValueError("I could not see the ink clearly after one retry")

    def _action(self, action, context, generation, from_rule):
        if not isinstance(action, dict) or action.get("op") not in OPS:
            raise ValueError("I cannot erase ink or do that paper operation")
        op = action["op"]
        if op in ("write_text", "append_text"):
            text = fill_template(action.get("text"), context)
            # Keep raw literal input strict; values such as player names and
            # game names are printed in the capital font after substitution.
            text = text.upper()
            if not text or any(c not in M.CAPITAL_STROKES and c not in " \n" for c in text):
                raise ValueError("I can write capitals, digits, spaces and . : - / only")
            item_id = action.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("the writing needs an item id")
            look = self._look()
            if op == "write_text":
                if item_id in self.items:
                    raise ValueError(f"{item_id} is already on the sheet; I can append to it")
                size = action.get("size_mm", L.DEFAULT_SIZE_MM)
                # Validate size/text independently so a bad size is not
                # misreported as an unavailable place.
                L.layout_text(text[:1], L.PAGE, size)
                place = action.get("place")
                region, cells = self._choose(text, place, size, look.track.pose)
                item = PaperItem(item_id, text, place, size, region, cells)
                with self._lock:
                    if generation != self._generation:
                        raise ValueError("the sheet changed before I could write")
                    self.items[item_id] = item  # reserve even partially drawn ink
                    self._recipes[item_id] = {"op": "write_text", "id": item_id, "text": text,
                                              "place": place, "size_mm": size}
                start = 0
            else:
                item = self._item(item_id, from_rule)
                if item is None:
                    return
                if not item.complete:
                    raise ValueError(f"{item_id} was not completely written, so I cannot append safely")
                try:
                    cells = L.layout_text(item.text + text, item.region, item.size_mm)
                except ValueError as exc:
                    raise self._append_refusal(item, text, look.track.pose, str(exc)) from exc
                start = len(item.characters)
                if any(a.box != b.box for a, b in zip(item.characters, cells)):
                    raise self._append_refusal(item, text, look.track.pose, "that would move earlier writing")
                obstacles = L.footprint(self.brain) + [c.box for key, other in self.items.items() if key != item_id
                                                    for c in other.characters]
                if any(c.box.overlaps(r) for c in cells[start:] for r in obstacles):
                    raise self._append_refusal(item, text, look.track.pose, "that would touch another drawing")
                try:
                    self._reachable(look.track.pose, [s for c in cells[start:] for s in c.strokes()])
                except ValueError as exc:
                    raise self._append_refusal(item, text, look.track.pose, str(exc)) from exc
                with self._lock:
                    if generation != self._generation:
                        raise ValueError("the sheet changed before I could append")
                    item.text += text
                    item.characters += cells[start:]
                    item.complete = False
                    self._recipes[item_id]["text"] = item.text
            if len(item.characters) - start >= 20:
                self.brain.say("Writing all those letters will take a few minutes; you can keep guessing as I write.")
            for group in writing_groups(item.characters[start:]):
                strokes = [s for cell in group for s in cell.strokes()]
                self._draw_checked(f"paper {item.id}: {''.join(c.char for c in group)}", strokes,
                                   generation, cells=group)
                for cell in group:
                    cell.written = True
            item.complete = True
            with self._lock:
                if generation == self._generation:
                    self._missing.discard(item.id)
            return
        target = action.get("target")
        if not isinstance(target, dict):
            raise ValueError("a paper mark needs a target item")
        item = self._item(target.get("id"), from_rule)
        if item is None:
            return
        chars = fill_template(target["chars"], context).upper() if "chars" in target else None
        selected = [c for c in item.characters if c.char != " " and (chars is None or c.char in chars)]
        if any(not c.written for c in selected):
            raise ValueError("I cannot mark letters that have not been checked on the paper")
        for c in selected:
            if op in c.marks:
                continue
            self._draw_checked(f"paper {op}: {c.char}", L.mark_strokes(op, c.box), generation)
            c.marks.add(op)
        if chars is None:
            item.marks.add(op)

    def _item(self, item_id, from_rule):
        item = self.items.get(item_id)
        if item is None:
            if isinstance(item_id, str):
                with self._lock:
                    self._missing.add(item_id)
            if from_rule:
                self._trace(f"target {item_id!r} is missing; needs_rewrite")
                return None
            raise ValueError(f"I haven't written {item_id} on this sheet")
        return item

    def describe(self):
        with self._lock:
            parts = []
            for item in self.items.values():
                text = item.text.replace("\n", "")
                label = "the alphabet" if text == "ABCDEFGHIJKLMNOPQRSTUVWXYZ" else repr(item.text.replace("\n", " "))
                marked = []
                for op, words in (("cross_out", "crossed out"), ("underline", "underlined"),
                                  ("circle", "circled"), ("tick", "ticked")):
                    chars = sorted({c.char for c in item.characters if op in c.marks})
                    if chars:
                        marked.append(f"{', '.join(chars)} {words}")
                parts.append(f"{label} in the {item.place}"
                             + (", with " + "; ".join(marked) if marked else "")
                             + (" (writing incomplete)" if not item.complete else ""))
            result = "Written on the paper: " + ("; ".join(parts) if parts else "nothing") + "."
            if self.rules:
                result += " Standing rules: " + "; ".join(r["name"] for r in self.rules) + "."
            if self._missing:
                result += " Needs rewriting: " + ", ".join(sorted(self._missing)) + "."
            return result

    def close(self):
        self._closed = True
        self._queue.put(None)
