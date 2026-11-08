"""The glue between three round-3 pieces (docs/round3.md): the planner's `custom_request` tool, the request
compiler (buddy/agent/requests.py: the player's words -> a validated routine) and the paper routines
(buddy/paper/routines.py: the routine -> ink on a free part of the sheet, and standing rules on events).

install(brain) is called once by the brain. It is safe when either package is missing (an older checkout, or a
worker branch not merged yet): the tool then explains kindly that it cannot do extra things on the paper yet.

Compiling takes the language model several seconds and writing can take minutes, so neither runs inside
brain._worker: the player can keep talking, guessing and switching games while Buddy writes. The paper
package queues its drawing behind game strokes itself.
"""
from __future__ import annotations

import threading

try:
    from . import requests as _requests
except ImportError:                                   # task C3 not merged
    _requests = None
try:
    from ..paper import Paper, context_from           # task B3 not merged
except ImportError:
    Paper = None

    def context_from(brain):                          # the templates' fallback values
        return {}


def install(brain):
    """Give the brain a Paper when the package exists, listen for events, attach the tool handler."""
    brain.paper = Paper(brain) if Paper is not None else None
    brain.pending_rewrite = []
    if brain.paper is not None:
        brain.listeners.append(brain.paper.on_event)
    brain.listeners.append(lambda ev: _on_event(brain, ev))
    brain._tool_custom_request = lambda args, text: custom_request(brain, args, text)


def _on_event(brain, ev):
    """A new game on a fresh sheet: anything a standing rule still needs is offered again ("from now on")."""
    if ev.kind != "game_start" or brain.paper is None:
        return
    missing = [i for i in list(getattr(brain.paper, "_recipes", {})) if brain.paper.needs_rewrite(i)]
    if missing:
        brain.pending_rewrite = missing
        brain.say(f"Shall I write the {' and the '.join(missing)} on this sheet again?")


def answer_rewrite(brain, yes: bool) -> bool:
    """Called by the answer tool when a rewrite offer is pending. True when the answer was consumed."""
    ids, brain.pending_rewrite = brain.pending_rewrite, []
    if not ids:
        return False
    if yes and brain.paper is not None:
        brain.say("Right you are. This will take me a little while.")
        threading.Thread(target=_rewrite, args=(brain, ids), daemon=True).start()
    else:
        brain.say("Alright, I'll leave the paper as it is.")
    return True


def _rewrite(brain, ids):
    """Write the saved items again on the fresh sheet, then put back the marks the current game calls for."""
    for item in ids:
        problem = _run(brain, brain.paper.rewrite, item, context_from(brain))
        if problem:
            continue
        ctx = context_from(brain)
        if ctx.get("wrong") and any(a.get("op") == "cross_out" for r in brain.paper.rules for a in r.get("do", [])):
            _run(brain, brain.paper.run, [{"op": "cross_out", "target": {"id": item, "chars": "{wrong}"}}], ctx)


def custom_request(brain, args, text):
    request = (text or "").strip() or str(args.get("request") or "").strip()   # the player's whole sentence, verbatim
    if brain.paper is None or _requests is None:
        brain.say("I can't write extra things on the paper yet, I'm afraid. I can play, keep score and explain.")
        return
    brain.app.trace("decide", f"a request about the paper: {request!r}: asking the model for a routine")
    brain.say("Let me think about how to do that.")
    threading.Thread(target=_compile_and_run, args=(brain, request), daemon=True).start()


def _compile_and_run(brain, request: str):
    try:
        routine = _requests.compile(request, brain.facts(), brain.state)
    except Exception as e:                                   # the model or the server, never the arm
        brain.app.trace("decide", f"request compile failed: {e}")
        brain.say("I'm sorry, I couldn't work out how to do that just now.")
        return
    if getattr(routine, "reason", None) is not None and not hasattr(routine, "now"):   # a Refusal
        brain.app.trace("decide", f"request refused: {routine.reason}")
        brain.say(routine.say)
        return
    brain.app.trace("plan", f"routine: now {[a.get('op') for a in routine.now]}, rules "
                            f"{[r.get('name') for r in routine.rules]}, cancel {routine.cancel}")
    if routine.say:
        brain.say(routine.say)
    for item in routine.cancel:
        brain.paper.cancel(item)
    for rule in routine.rules:
        brain.paper.add_rule(rule)
    if routine.now:
        _run(brain, brain.paper.run, routine.now, context_from(brain))


def _run(brain, fn, *args):
    try:
        problem = fn(*args)
    except Exception as e:
        brain.app.trace("decide", f"paper routine error: {e}")
        problem = "I'm sorry, that didn't work on the paper."
    if problem:
        brain.say(problem)
    return problem
