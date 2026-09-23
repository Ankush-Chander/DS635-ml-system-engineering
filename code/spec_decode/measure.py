"""Measure — sweep k and watch acceptance length and throughput part ways.

Acceptance *rate* is not the number that predicts a speedup. Acceptance
*length* — mean tokens committed per verify pass, between 1 and k+1 — is, and
even that only up to a point: drafting cost grows linearly in k while the payoff
decays geometrically, because the drafts form a chain,

    E[tokens] = 1 + a1 + a1*a2 + a1*a2*a3 + ...

So acceptance length rises monotonically with k and throughput does not. The
sweep prints both; the k where they disagree is the lab's punchline.

Run:  python3 measure.py --drafter model --kmax 8
      python3 measure.py --drafter lookup --kmax 8 --prompt code
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "common"))
from blackbox import LM  # noqa: E402
from drafters import ModelDrafter, PromptLookupDrafter  # noqa: E402
from stage3_greedy_spec import greedy_baseline, greedy_spec_decode  # noqa: E402

PROMPTS = {
    "code": (
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
        "\n"
        "def factorial(n):\n"
    ),
    "prose": "The three laws of thermodynamics are often summarised as follows:",
}
RESULTS = Path(__file__).parent / "results"
HEADER = ["drafter", "k", "tokens", "target_passes", "drafter_passes", "acceptance_length", "seconds", "tokens_per_s", "speedup"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--drafter", choices=["model", "lookup", "both"], default="both")
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--prompt", choices=list(PROMPTS), default="code")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(RESULTS / "sweep.csv"))
    args = ap.parse_args()

    target = LM(args.target, device=args.device)
    prompt = target.tokenizer.encode(PROMPTS[args.prompt])

    greedy_baseline(target, prompt, 8)  # warm the kernels
    target.reset_counters()
    t0 = time.perf_counter()
    baseline = greedy_baseline(target, prompt, args.max_new)
    base_time, base_passes = time.perf_counter() - t0, target.passes
    print(f"target   {args.target} on {target.device}")
    print(f"prompt   {args.prompt}, {len(prompt)} tokens -> {args.max_new} new\n")
    print(f"{'plain greedy':<14}{'':>4}{base_passes:>8} passes{base_time:>9.2f}s{len(baseline) / base_time:>9.1f} tok/s\n")

    kinds = ["lookup", "model"] if args.drafter == "both" else [args.drafter]
    rows = []
    print(f"{'drafter':<14}{'k':>3}{'passes':>8}{'draft':>7}{'accept len':>12}{'tok/s':>9}{'speedup':>9}{'exact':>7}")
    for kind in kinds:
        draft_lm = LM(args.draft, device=args.device) if kind == "model" else None
        for k in range(1, args.kmax + 1):
            drafter = (
                PromptLookupDrafter(vocab_size=target.model.config.vocab_size, device=target.device)
                if kind == "lookup"
                else ModelDrafter(draft_lm, greedy=True)
            )
            drafter.reset()
            target.reset_counters()
            t0 = time.perf_counter()
            out, blocks, committed = greedy_spec_decode(target, drafter, prompt, args.max_new, k)
            elapsed = time.perf_counter() - t0
            accept_len = sum(committed) / blocks
            tps = len(out) / elapsed
            rows.append(dict(zip(HEADER, [
                kind, k, len(out), target.passes, drafter.passes,
                round(accept_len, 3), round(elapsed, 3), round(tps, 2),
                round(base_time / elapsed, 3),
            ])))
            exact = "ok" if out == baseline else "DIFF"
            print(f"{kind:<14}{k:>3}{target.passes:>8}{drafter.passes:>7}{accept_len:>12.2f}{tps:>9.1f}{base_time / elapsed:>9.2f}x{exact:>7}")

    RESULTS.mkdir(exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")

    best = max(rows, key=lambda r: r["tokens_per_s"])
    longest = max(rows, key=lambda r: r["acceptance_length"])
    print(f"fastest            {best['drafter']} k={best['k']}  {best['tokens_per_s']} tok/s  (accept len {best['acceptance_length']})")
    print(f"longest acceptance {longest['drafter']} k={longest['k']}  accept len {longest['acceptance_length']}  ({longest['tokens_per_s']} tok/s)")
    if (best["drafter"], best["k"]) != (longest["drafter"], longest["k"]):
        print("\nlongest acceptance is not the fastest — acceptance length alone does not predict throughput")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
