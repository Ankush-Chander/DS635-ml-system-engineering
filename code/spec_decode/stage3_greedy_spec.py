"""Stage 3 — greedy speculative decoding. ~30 lines, and a hard correctness test.

Greedy collapses the accept/reject rule to `accept while argmax(p_i) == x_i`,
which makes the output *deterministic* — and therefore checkable. The test is
not statistical: speculative output must equal plain greedy output token for
token. If it differs anywhere, the cache rollback or the position bookkeeping
is wrong.

Verification is ONE forward pass over k+1 positions. Running the target once per
draft token would pass this test and delete the entire speedup, so the pass
count is reported alongside.

Run:  python3 stage3_greedy_spec.py --k 4 [--drafter lookup|model]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "common"))
from blackbox import LM, KVCache  # noqa: E402
from drafters import ModelDrafter, PromptLookupDrafter  # noqa: E402

PROMPT = (
    "def fibonacci(n):\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
    "\n"
    "def factorial(n):\n"
)


def greedy_baseline(target: LM, prompt: list[int], max_new: int) -> list[int]:
    """Plain autoregressive greedy: one forward pass per token."""
    cache = KVCache()
    seq = list(prompt)
    logits = target.forward(seq, cache)
    for _ in range(max_new):
        token = int(logits[-1].argmax())
        seq.append(token)
        if token == target.tokenizer.eos_token_id:
            break
        logits = target.forward([token], cache)
    return seq[len(prompt) :]


def greedy_spec_decode(target: LM, drafter, prompt: list[int], max_new: int, k: int):
    """Returns (generated tokens, blocks run, tokens committed per block)."""
    cache = KVCache()
    seq = list(prompt)
    target.forward(seq[:-1], cache)  # cache holds every committed token but the last
    committed: list[int] = []
    blocks = 0

    while len(seq) - len(prompt) < max_new:
        drafts, _ = drafter.draft(seq, k)

        # ONE pass: the uncached tail of the committed sequence, then the drafts.
        held = len(cache)
        logits = target.forward(seq[held:] + drafts, cache)
        preds = logits[len(seq) - held - 1 :].argmax(-1)  # p_1 .. p_{k+1}

        n_acc = 0
        for i, token in enumerate(drafts):
            if int(preds[i]) != token:
                break
            n_acc += 1

        # One token always follows: the correction if a draft was rejected,
        # the bonus token if every draft survived.
        new = drafts[:n_acc] + [int(preds[n_acc])]
        seq.extend(new)
        committed.append(len(new))
        blocks += 1
        cache.truncate(len(seq) - 1)
        if target.tokenizer.eos_token_id in new:
            break

    return seq[len(prompt) : len(prompt) + max_new], blocks, committed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--drafter", choices=["lookup", "model"], default="lookup")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    target = LM(args.target, device=args.device)
    prompt = target.tokenizer.encode(PROMPT)

    if args.drafter == "lookup":
        drafter = PromptLookupDrafter(vocab_size=target.model.config.vocab_size, device=target.device)
    else:
        drafter = ModelDrafter(LM(args.draft, device=args.device), greedy=True)

    print(f"target   {args.target}")
    print(f"drafter  {drafter.name}   k={args.k}\n")

    greedy_baseline(target, prompt, 8)  # warm the kernels so the first timed run is not the slow one

    target.reset_counters()
    t0 = time.perf_counter()
    baseline = greedy_baseline(target, prompt, args.max_new)
    base_time, base_passes = time.perf_counter() - t0, target.passes

    target.reset_counters()
    drafter.reset()
    t0 = time.perf_counter()
    spec, blocks, committed = greedy_spec_decode(target, drafter, prompt, args.max_new, args.k)
    spec_time, spec_passes = time.perf_counter() - t0, target.passes

    print(target.tokenizer.decode(spec))
    print()
    print(f"{'':<18}{'tokens':>8}{'target passes':>15}{'seconds':>10}")
    print(f"{'plain greedy':<18}{len(baseline):>8}{base_passes:>15}{base_time:>10.2f}")
    print(f"{'speculative':<18}{len(spec):>8}{spec_passes:>15}{spec_time:>10.2f}")
    print()
    print(f"blocks             {blocks}")
    print(f"acceptance length  {sum(committed) / blocks:.2f}  (1 to k+1 = {args.k + 1})")
    print(f"drafter passes     {drafter.passes}")
    print(f"speedup            {base_time / spec_time:.2f}x")

    identical = spec == baseline
    fewer = spec_passes < base_passes
    print(f"\nidentical to greedy   {'PASS' if identical else 'FAIL'}")
    print(f"fewer target passes   {'PASS' if fewer else 'FAIL'}")
    if not identical:
        print(f"  baseline {baseline}\n  spec     {spec}")
    return 0 if identical and fewer else 1


if __name__ == "__main__":
    raise SystemExit(main())
