"""Pretrain the in-context agent's transformer on the pretraining distribution.

    python scripts/pretrain_icl.py --run-dir runs/icl --steps 20000

Resumable: rerunning with the same --run-dir continues from the latest
checkpoint. Batches are a pure function of the step number, so a resumed run
sees exactly the data an uninterrupted one would have.

Writes to --run-dir:
  latest.pt   full training state, rewritten every --ckpt-every steps
  best.pt     model with the lowest validation loss so far (checkpoint selection per AGENT_SPECS.md)
  log.jsonl   one line per eval
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from regime.agents.in_context import ModelConfig, MoveTransformer, save_model
from regime.pretrain import PRETRAIN_SEED_BASE, VALIDATION_SEED_BASE, pretraining_batch


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def losses(model, x, y, p) -> tuple[torch.Tensor, float]:
    """Cross-entropy, and its excess over an oracle that knows the true distribution."""
    logits, _ = model(x)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    # The oracle's expected CE is the entropy of the true distribution.
    floor = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()
    return ce, (ce - floor).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--episodes-per-batch", type=int, default=8)
    ap.add_argument("--windows-per-episode", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig()
    model = MoveTransformer(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps))),
    )
    step, best_val = 0, float("inf")

    latest = args.run_dir / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu")
        if ck["args"]["steps"] != args.steps or ck["args"]["lr"] != args.lr:
            raise SystemExit("run dir was started with different --steps/--lr; use a new --run-dir")
        model.load_state_dict(ck["model"]["state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"resumed at step {step} (best val {best_val:.4f})")

    batch_kw = dict(n_episodes=args.episodes_per_batch, windows_per_episode=args.windows_per_episode, context=cfg.context)
    val = [
        tuple(torch.as_tensor(a, device=args.device) for a in pretraining_batch(VALIDATION_SEED_BASE + i, **batch_kw))
        for i in range(args.val_batches)
    ]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} parameters; training to step {args.steps}")

    t0 = time.time()
    while step < args.steps:
        model.train()
        x, y, p = (torch.as_tensor(a, device=args.device) for a in pretraining_batch(PRETRAIN_SEED_BASE + step, **batch_kw))
        ce, _ = losses(model, x, y, p)
        opt.zero_grad()
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vals = [losses(model, *b) for b in val]
            val_ce = float(np.mean([v[0].item() for v in vals]))
            val_excess = float(np.mean([v[1] for v in vals]))
            improved = val_ce < best_val
            if improved:
                best_val = val_ce
                save_model(model, args.run_dir / "best.pt")  # load_model maps to CPU
            rec = dict(step=step, train_ce=ce.item(), val_ce=val_ce, val_excess=val_excess,
                       lr=sched.get_last_lr()[0], sec=round(time.time() - t0, 1))
            with open(args.run_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step:>6}  train {ce.item():.4f}  val {val_ce:.4f}  excess {val_excess:.4f}"
                  f"{'  *' if improved else ''}  ({rec['sec']}s)", flush=True)

        if step % args.ckpt_every == 0 or step == args.steps:
            atomic_save(
                dict(model=dict(config=asdict(cfg), state_dict=model.state_dict()), opt=opt.state_dict(),
                     sched=sched.state_dict(), step=step, best_val=best_val, args=vars(args) | {"run_dir": str(args.run_dir)}),
                latest,
            )


if __name__ == "__main__":
    main()
