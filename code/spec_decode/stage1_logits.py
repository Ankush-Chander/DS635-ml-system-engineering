"""Stage 1 — the black box. One forward pass, logits at every position.

Speculative decoding is built on a property of transformer inference that
ordinary generation never exercises: feeding T tokens yields the next-token
distribution for *every* prefix, not just the longest one. That is what lets a
block of k draft tokens be checked in one pass instead of k.

Checkpoint: `logits[i]` from a single pass over T tokens equals the logits from
a separate pass over `tokens[:i+1]`, to 1e-4.

Run:  python3 stage1_logits.py [--model Qwen/Qwen2.5-1.5B-Instruct]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "common"))
from blackbox import LM  # noqa: E402

PROMPT = "The capital of France is Paris. The capital of Japan is"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    lm = LM(args.model, device=args.device)
    tokens = lm.tokenizer.encode(PROMPT)
    print(f"model   {args.model} on {lm.device}")
    print(f"prompt  {len(tokens)} tokens\n")

    logits = lm.forward(tokens)
    print(f"one pass over {len(tokens)} tokens -> logits {tuple(logits.shape)}")

    print(f"\n{'i':>3} {'token':>14} {'argmax next':>16}")
    for i, tok in enumerate(tokens):
        nxt = lm.tokenizer.decode([int(logits[i].argmax())])
        print(f"{i:>3} {lm.tokenizer.decode([tok])!r:>14} {nxt!r:>16}")

    worst = 0.0
    for i in range(len(tokens)):
        alone = lm.forward(tokens[: i + 1])
        worst = max(worst, float((logits[i] - alone[-1]).abs().max()))

    print(f"\nmax |batched - individual| = {worst:.2e}  (atol {args.atol:.0e})")
    ok = worst < args.atol
    print(f"logits at every position   {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
