"""Stage 4 — sampled speculative decoding. The output distribution is exactly p.

    for i in 1..k:
        r ~ U(0,1)
        if r < min(1, p_i(x_i) / q_i(x_i)):   accept x_i
        else:                                 x_i' ~ norm(max(0, p_i - q_i));  STOP
    if all k accepted:                        bonus ~ p_{k+1}

Dropping the residual and simply resampling from `p` on rejection looks
harmless and biases the model: tokens the drafter over-weights stay
over-represented. `--verify` is the test that catches it — it holds one real
(p, q) pair fixed and draws from the accept/reject rule often enough for a
chi-square to see the difference.

Run:  python3 stage4_sampled_spec.py --k 4 --verify
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "common"))
from blackbox import LM, KVCache, probs_from  # noqa: E402
from drafters import ModelDrafter, PromptLookupDrafter  # noqa: E402
from stats import chi_square_test  # noqa: E402

PROMPT = "The three laws of thermodynamics are often summarised as follows:"


def residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """norm(max(0, p - q)) — what makes the committed token distributed as p."""
    r = torch.clamp(p - q, min=0.0)
    total = r.sum()
    return p.clone() if total <= 0 else r / total


def accept_reject(p: torch.Tensor, q: torch.Tensor, drafts: list[int]) -> tuple[int, int]:
    """Returns (tokens accepted, the one token that follows them)."""
    for i, x in enumerate(drafts):
        ratio = (p[i, x] / q[i, x]).item() if q[i, x] > 0 else float("inf")
        if torch.rand(()).item() < min(1.0, ratio):
            continue
        return i, int(torch.multinomial(residual(p[i], q[i]), 1))
    return len(drafts), int(torch.multinomial(p[len(drafts)], 1))  # bonus token


def sampled_spec_decode(target, drafter, prompt, max_new, k, temperature):
    cache = KVCache()
    seq = list(prompt)
    target.forward(seq[:-1], cache)
    committed: list[int] = []
    blocks = 0

    while len(seq) - len(prompt) < max_new:
        drafts, q = drafter.draft(seq, k)

        held = len(cache)
        logits = target.forward(seq[held:] + drafts, cache)
        p = probs_from(logits[len(seq) - held - 1 :], temperature)  # p_1 .. p_{k+1}

        n_acc, follow = accept_reject(p, q, drafts) if drafts else (0, int(torch.multinomial(p[0], 1)))
        new = drafts[:n_acc] + [follow]
        seq.extend(new)
        committed.append(len(new))
        blocks += 1
        cache.truncate(len(seq) - 1)
        if target.tokenizer.eos_token_id in new:
            break

    return seq[len(prompt) : len(prompt) + max_new], blocks, committed


def verify_distribution(target: LM, draft_lm: LM, prompt: list[int], temperature: float, n: int) -> bool:
    """Hold one real (p, q) pair fixed; check the committed token is distributed as p.

    With p and q fixed the whole experiment is vectorised: draw n tokens from q,
    decide acceptance against n uniforms, and draw the rejections from a residual
    that only has to be built once.
    """
    p = probs_from(target.forward(prompt)[-1], temperature).cpu()
    q = probs_from(draft_lm.forward(prompt)[-1], temperature).cpu()

    draws = torch.multinomial(q, n, replacement=True)
    ratio = torch.where(q[draws] > 0, p[draws] / q[draws], torch.full_like(p[draws], float("inf")))
    keep = torch.rand(n) < ratio.clamp(max=1.0)

    counts = torch.zeros(len(p), dtype=torch.long)
    counts.index_add_(0, draws[keep], torch.ones(int(keep.sum()), dtype=torch.long))
    rejected = n - int(keep.sum())
    if rejected:
        replacements = torch.multinomial(residual(p, q), rejected, replacement=True)
        counts.index_add_(0, replacements, torch.ones(rejected, dtype=torch.long))

    stat, dof, pvalue, kept = chi_square_test(counts.numpy(), p.numpy())
    rate = int(keep.sum()) / n
    predicted = float(torch.minimum(p, q).sum())

    print(f"\n--- distribution check ({n} draws, {len(kept)} cells + pooled tail) ---")
    print(f"chi-square      stat={stat:.1f}  dof={dof}  p={pvalue:.4f}")
    print(f"acceptance      observed={rate:.4f}  predicted={predicted:.4f}  (1 - TV)")
    ok = pvalue > 0.01
    print(f"output ~ p      {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--drafter", choices=["model", "lookup"], default="model")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="run the chi-square check")
    ap.add_argument("--draws", type=int, default=200_000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    target = LM(args.target, device=args.device)
    prompt = target.tokenizer.encode(PROMPT)

    needs_draft_model = args.drafter == "model" or args.verify
    draft_lm = LM(args.draft, device=args.device) if needs_draft_model else None

    if args.drafter == "lookup":
        drafter = PromptLookupDrafter(vocab_size=target.model.config.vocab_size, device=target.device)
    else:
        drafter = ModelDrafter(draft_lm, temperature=args.temperature)

    print(f"target   {args.target}")
    print(f"drafter  {drafter.name}   k={args.k}  T={args.temperature}\n")

    sampled_spec_decode(target, drafter, prompt, 8, args.k, args.temperature)  # warm the kernels

    target.reset_counters()
    drafter.reset()
    t0 = time.perf_counter()
    out, blocks, committed = sampled_spec_decode(
        target, drafter, prompt, args.max_new, args.k, args.temperature
    )
    elapsed = time.perf_counter() - t0

    print(target.tokenizer.decode(out))
    print()
    print(f"tokens             {len(out)}")
    print(f"blocks             {blocks}")
    print(f"target passes      {target.passes}")
    print(f"acceptance length  {sum(committed) / blocks:.2f}  (1 to k+1 = {args.k + 1})")
    print(f"seconds            {elapsed:.2f}")

    length_ok = sum(committed) / blocks > 1.0
    print(f"\nacceptance length > 1  {'PASS' if length_ok else 'FAIL'}")

    dist_ok = verify_distribution(target, draft_lm, prompt, args.temperature, args.draws) if args.verify else True
    return 0 if length_ok and dist_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
