"""Stage 2 — KV cache with rollback. The real gate.

The sampler appends draft tokens to the cache, then throws some of them away.
If `truncate` leaves a single stale key behind, every later logit is quietly
wrong — no crash, no NaN, just a model that is no longer the model you loaded.

Two checkpoints:
  1. cached vs uncached logits agree to 1e-4
  2. after `truncate(n)` and a different continuation, logits agree with a
     fresh uncached pass over the new sequence

Run:  python3 stage2_kvcache.py [--model Qwen/Qwen2.5-1.5B-Instruct]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "common"))
from blackbox import LM, KVCache  # noqa: E402

PROMPT = "Speculative decoding verifies a block of draft tokens in"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    lm = LM(args.model, device=args.device)
    tokens = lm.tokenizer.encode(PROMPT)
    split = len(tokens) // 3  # not half: truncate(n) and "remove n" must not coincide
    print(f"model   {args.model} on {lm.device}")
    print(f"prompt  {len(tokens)} tokens, cached in two chunks at {split}\n")

    reference = lm.forward(tokens)

    cache = KVCache()
    lm.forward(tokens[:split], cache)
    print(f"after chunk 1  cache holds {len(cache)} positions")
    incremental = lm.forward(tokens[split:], cache)
    print(f"after chunk 2  cache holds {len(cache)} positions")

    drift = float((reference[split:] - incremental).abs().max())
    print(f"\nmax |uncached - cached|    = {drift:.2e}")
    cached_ok = drift < args.atol

    # Rollback: drop everything after `split`, then commit a different continuation.
    detour = lm.tokenizer.encode(" a completely different direction entirely")
    cache.truncate(split)
    print(f"\ntruncate({split})           cache holds {len(cache)} positions")
    rolled = lm.forward(detour, cache)
    fresh = lm.forward(tokens[:split] + detour)

    rollback_drift = float((fresh[split:] - rolled).abs().max())
    print(f"max |fresh - rolled back|  = {rollback_drift:.2e}")
    rollback_ok = rollback_drift < args.atol

    print(f"\nincremental cache          {'PASS' if cached_ok else 'FAIL'}")
    print(f"rollback                   {'PASS' if rollback_ok else 'FAIL'}")
    return 0 if cached_ok and rollback_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
