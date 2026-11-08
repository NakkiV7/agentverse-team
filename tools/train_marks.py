"""Train the mark classifier on images generated in our own simulator.

    .venv/bin/python tools/train_marks.py            # ~2 minutes; live progress: runtime/progress.txt and
                                                      # http://localhost:8850/progress.html (tools/serve_progress.py)

1. Generate. For each of N boards: a blank sheet at a random position and rotation, the grid drawn with a little
   jitter, and every cell filled at random with nothing, an X or an O in a random style (clean like Penny's, or
   messy, small, big, slanted, off-centre, unclosed, blue or black ink). Lighting and camera noise are randomised.
   The board goes through the SAME pipeline as at run time: the simulated overhead camera, find_sheet, rectify,
   cell crops. Labels are known because we drew the marks.
2. Train. A small CNN (buddy/perception/marknet.py) on 85 % of the boards, validated on the other 15 % (whole
   boards held out, so no near-duplicates leak between the sets).
3. Report. Accuracy, confusion matrix, and how often a validation cell would fall below the confidence threshold
   (Penny would ask the player). Saved next to the weights in assets/models/.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from progress import Progress                                         # noqa: E402

from buddy import config as C                                          # noqa: E402
from buddy.game.marks import random_mark                               # noqa: E402
from buddy.motion.drawing import cell_centres_mm, grid_lines_mm       # noqa: E402
from buddy.perception.board import CLASSES, MIN_CONFIDENCE, MODEL_PATH, cell_crops, rectify   # noqa: E402
from buddy.perception.sheet import CameraModel, find_sheet            # noqa: E402
from buddy.sim.world import World                                     # noqa: E402

INKS = [C.INK_RGB, C.PLAYER_INK_RGB, (20, 20, 120), (60, 60, 60)]


def paint_strokes(world, strokes, rgb, radius):
    for st in strokes:
        st = np.asarray(st, float)
        for a, b in zip(st[:-1], st[1:]):
            world.paint(a, b, rgb=rgb, radius_mm=radius)


def generate(n_boards: int, seed: int, prog: Progress):
    rng = np.random.default_rng(seed)
    w = World()
    cam = CameraModel(*w.camera_model())
    light0 = w.model.light_diffuse.copy()
    X, Y, G = [], [], []
    for b in range(n_boards):
        w.clear_ink()
        w.move_sheet(rng.uniform(0.19, 0.235), rng.uniform(-0.035, 0.035), math.pi / 2 + rng.uniform(-0.35, 0.35))
        w.model.light_diffuse[:] = light0 * rng.uniform(0.75, 1.2)
        w.noise = rng.uniform(1.0, 6.0)
        jitter = rng.uniform(-1.5, 1.5, 2)
        paint_strokes(w, [np.asarray(l) + jitter for l in grid_lines_mm()], INKS[rng.integers(2)], rng.uniform(0.5, 0.9))
        labels = rng.integers(0, 3, 9)
        for cell, lab in enumerate(labels):
            if lab:
                paint_strokes(w, random_mark(CLASSES[lab], cell_centres_mm()[cell], rng),
                              INKS[rng.integers(len(INKS))], rng.uniform(0.45, 1.0))
        rgb = w.render_overhead()
        sheet = find_sheet(rgb, cam)
        if sheet is None or sheet.confidence < 0.6:
            continue
        X.append(cell_crops(rectify(rgb, cam, sheet.as_tuple())))
        Y.append(labels)
        G.append(np.full(9, b))
        if b % 10 == 0:
            print("\r" + prog.update(b + 1, n_boards, "generating boards in the simulator"), end="", flush=True)
    w.close()
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def train(X, Y, G, epochs: int, seed: int, prog: Progress, n_boards: int):
    import torch
    from torch import nn
    from buddy.perception.marknet import MarkNet
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    boards = np.unique(G)
    val_b = set(rng.choice(boards, int(0.15 * len(boards)), replace=False))
    val = np.isin(G, list(val_b))
    Xt, Yt, Xv, Yv = X[~val], Y[~val], X[val], Y[val]
    net = MarkNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * math.ceil(len(Xt) / 128))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.02)
    xt, yt = torch.from_numpy(Xt[:, None]), torch.from_numpy(Yt.astype(np.int64))
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            idx = perm[i:i + 128]
            xb = xt[idx]
            # augmentation: small shifts, contrast changes, extra noise
            sh = rng.integers(-3, 4, 2)
            xb = torch.roll(xb, shifts=(int(sh[0]), int(sh[1])), dims=(2, 3))
            xb = (xb * torch.empty(len(xb), 1, 1, 1).uniform_(0.7, 1.3) + 0.03 * torch.randn_like(xb)).clamp(0, 1)
            loss = lossf(net(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print("\r" + prog.update(n_boards + ep + 1, n_boards + epochs, f"training: epoch {ep + 1}/{epochs}, "
              f"loss {loss.item():.3f}"), end="", flush=True)
    net.eval()
    with torch.no_grad():
        p = torch.softmax(net(torch.from_numpy(Xv[:, None])), 1).numpy()
    pred, conf = p.argmax(1), p.max(1)
    cm = np.zeros((3, 3), int)
    for t, q in zip(Yv, pred):
        cm[t, q] += 1
    sure = conf >= MIN_CONFIDENCE
    report = {
        "boards": int(len(boards)), "train_cells": int(len(Xt)), "val_cells": int(len(Xv)),
        "val_accuracy": float((pred == Yv).mean()),
        "val_accuracy_when_confident": float((pred[sure] == Yv[sure]).mean()),
        "val_unclear_fraction": float(1 - sure.mean()),
        "confusion_rows_true_cols_pred": {"classes": CLASSES, "matrix": cm.tolist()},
        "min_confidence": MIN_CONFIDENCE, "epochs": epochs,
        "params": int(sum(q.numel() for q in net.parameters())),
    }
    return net, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=1600)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    prog = Progress("train mark classifier")
    t0 = time.time()
    X, Y, G = generate(args.boards, args.seed, prog)
    print(f"\n{len(X)} cell images from {len(np.unique(G))} boards in {time.time() - t0:.0f} s")
    net, report = train(X, Y, G, args.epochs, args.seed, prog, args.boards)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), MODEL_PATH)
    report["seconds"] = round(time.time() - t0, 1)
    MODEL_PATH.with_suffix(".json").write_text(json.dumps(report, indent=1))
    prog.finish(f"val accuracy {100 * report['val_accuracy']:.2f} %")
    print("\n" + json.dumps(report, indent=1))
