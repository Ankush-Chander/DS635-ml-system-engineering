# Speculative Decoding — Route A

## TL;DR

- **The model is a black box:** `(tokens, cache) -> logits`. Forward passes only — no backprop, no training, no optimizer.
- **Start with `stage0_rejection.py`.** No model, no GPU, ~5 seconds. It holds the entire correctness argument.
- **The gate is the KV cache rollback**, not the sampler. Stage 2 is where builds actually break.
- **Every stage exits non-zero when its checkpoint fails.** `make all` is a test suite, not a demo.
- **Route B swaps one file** — `common/blackbox.py`. The sampling loops never change.

---

## Run this first

```bash
source ../../.venv/bin/activate        # or your own env
pip install -r requirements.txt

python stage0_rejection.py             # no model, no GPU, ~5 s
python stage0_rejection.py --broken    # watch the same test fail
```

If `--broken` does not fail, nothing downstream is trustworthy.

The `Makefile` uses `python3` — point it elsewhere with `make PY=../../.venv/bin/python stage0`.

---

## Stage 0 — why it exists

**Purpose: prove the sampler is exact, before anything can hide that it isn't.**

Everything after stage 0 is engineering — caches, batching, pass counts. Stage 0 is the only place the **statistical claim** is on trial, and it is the one claim no benchmark can check.

### Four takeaways

| # | Takeaway | The evidence in the file |
| --- | --- | --- |
| 1 | **It's a sampling rule, not a model trick** | 5 symbols, 93 lines, no torch. `P` and `Q` are hardcoded |
| 2 | **Output is *exactly* `p`** — not approximately | χ²=4.5 on dof=4 over 1M samples |
| 3 | **`norm(max(0, p - q))` is load-bearing, and fails silently** | `--broken` — see below |
| 4 | **Acceptance rate belongs to the *pair*, not the drafter** | observed 0.6993 vs `1 - TV(p, q)` = 0.7000 |

### Takeaway 3 is the payload

`--broken` resamples from `p` on rejection — a fix that looks obviously fine:

| | Acceptance rate | Chi-square |
| --- | --- | --- |
| correct | 0.6993 | 4.6 |
| broken | **0.6993** | **20191** |

Both rows `--n 200000 --seed 0`. **Same rate to four decimals. Same speed. Different model.**

No latency dashboard, no throughput graph, no eyeball test finds this. Only a distribution test does.

### Read the drafter `Q` closely

```python
P = [0.40, 0.25, 0.20, 0.10, 0.05]
Q = [0.40, 0.15, 0.10, 0.00, 0.35]
```

`Q` is wrong in **three distinct ways** on purpose:

- **Symbol 0** — correct.
- **Symbol 4** — overconfident.
- **Symbol 3** — blind. `Q[3] = 0`, so **symbol 3 can only enter the output through the residual**. Break the residual and its frequency is the tell.

### Run it without running it

Two browser pages, no Python needed. Both ship with the course site under **Demos**:

| Page | Published at |
| --- | --- |
| The acceptance rule | <https://DS635.ankushchander.com/viz/acceptance-rule.html> |
| Convergence bench | <https://DS635.ankushchander.com/viz/rejection-sampling.html> |

**`docs/viz/acceptance-rule.html`** — one proposal at a time. Use this to teach.

- **Propose token** draws one symbol from `q` and shows the accept probability as `min(1, p/q)`.
- **Where the mass goes** is the point: the residual chart shows what `p` wants that `q` never supplies.
- The `q` card annotates **each symbol with how often a proposal there survives** — 100%, or 14%, or never.

**`docs/viz/rejection-sampling.html`** — the convergence bench. Use this to prove it.

- **Drag `p` and `q`** and watch the acceptance rate track `1 - TV(p, q)` live.
- **Toggle `--broken`** and watch the chi-square blow up while the acceptance rate sits still.

### One sentence

Stage 0 exists so that when stage 3 says PASS, you know it is testing the **plumbing** — the math was settled somewhere a GPU could not obscure it.

---

## What each file is for

| File | What it teaches | Model? | Time |
| --- | --- | --- | --- |
| `stage0_rejection.py` | **The accept/reject rule**, on 5 symbols instead of 150k | no | 5 s |
| `stage1_logits.py` | **One pass gives logits at every position** — the property the whole trick rests on | yes | 30 s |
| `stage2_kvcache.py` | **`append` + `truncate(n)`** and why a stale key is silent | yes | 30 s |
| `stage3_greedy_spec.py` | **Greedy spec decode** — deterministic, so exactly checkable | yes | 1 min |
| `stage4_sampled_spec.py` | **Sampled spec decode** — residual + bonus token | yes | 2 min |
| `measure.py` | **k sweep** → `results/sweep.csv` | yes | 5 min |

### Supporting files

| File | What's in it |
| --- | --- |
| `common/blackbox.py` | `LM.forward` and `KVCache`. **The only file Route B replaces** |
| `common/drafters.py` | `PromptLookupDrafter` (no weights) and `ModelDrafter` |
| `common/stats.py` | Chi-square goodness of fit, no SciPy |
| `Makefile` | `make stage0`, `make stages`, `make sweep` |
| `docs/viz/acceptance-rule.html` | **One proposal at a time** — q proposes, the ratio test rules, the residual corrects. Start here in a lecture |
| `docs/viz/rejection-sampling.html` | **Convergence bench** — run millions of draws, watch the histogram settle onto `p` |

---

## The ladder — and what proves each rung

| # | Stage | Checkpoint |
| --- | --- | --- |
| 0 | Rejection sampling | 1M samples match `p` (chi-square); acceptance rate = `1 - TV(p, q)` |
| 1 | Logits everywhere | Batched logits match per-position passes to **1e-4** |
| 2 | Cache + rollback | Cached = uncached to 1e-4, **and** survives `truncate(n)` + a new continuation |
| 3 | Greedy spec | Output **identical to plain greedy**, token for token, fewer target passes |
| 4 | Sampled spec | Committed token distributed as `p` (chi-square, 300k draws); acceptance length > 1 |

Stage 4 on real models, 300k draws: `stat=986.7, dof=994, p=0.5590`. **A chi-square landing on its degrees of freedom is what exact looks like.**

---

## The correctness core

```
draft  x1..xk  autoregressively from q     (k forward passes, small model)
verify ONE forward pass of p over the k drafts -> p1..p_{k+1}

for i in 1..k:
    r ~ U(0,1)
    if r < min(1, p_i(x_i) / q_i(x_i)):   accept x_i
    else:                                 x_i' ~ norm(max(0, p_i - q_i));  STOP
if all k accepted:                        bonus ~ p_{k+1}
```

- **ONE pass over k+1 positions.** Looping the target k times passes every correctness test here and deletes the speedup — so each stage prints its pass count.
- **`max(0, p - q)` renormalised** is what makes the output exactly `p`.
- **The bonus token** is why a block yields `k+1`, not `k`. `p_{k+1}` is already computed.

Greedy collapses all of it to `accept while argmax(p_i) == x_i`.

---

## The cache invariant

**The target cache holds every committed token but the last.**

- A verify pass feeds `[last committed token] + drafts`, reads `p_1 .. p_{k+1}`.
- Afterwards it truncates back to `len(sequence) - 1`.
- `DynamicCache.crop` takes a **negative** count to remove tokens; positive is the legacy "keep n". `KVCache.truncate` takes the length to **keep** and asserts the result.

---

## Drafters

| Drafter | Extra weights | Use it when |
| --- | --- | --- |
| `PromptLookupDrafter` | **none** | Code, or any repeating text. **Reaches stage 3 with no second model** |
| `ModelDrafter` | one checkpoint | 0.5B drafting for 1.5B. **Must share a tokenizer** |

Prompt-lookup drafts deterministically, so its `q` is one-hot — a real distribution. `min(1, p/1) = p(x)`, and `norm(max(0, p - e_x))` is `p` with `x` removed. The sampled loop runs on it unchanged.

---

## Two traps

**1. Acceptance rate is not speedup.** Measured, 1.5B target + 0.5B draft, fp32, k=4, CPU:

| | Target passes | Acceptance length | Wall clock |
| --- | --- | --- | --- |
| plain greedy | 33 | — | 47.8 s |
| speculative | **8** | **4.86** of 5 | 53.7 s → **0.89x** |

Near-perfect drafting, 4x fewer target passes, **and still slower** — 28 drafter passes cost more than the 25 they saved. Speculative decoding pays only where the target's forward pass is latency-bound, which is what a GPU makes it.

**2. Per-position rates do not average.** Drafting is a chain:

```
E[tokens] = 1 + a1 + a1*a2 + a1*a2*a3 + ...
```

Cost grows linearly in k; payoff decays geometrically.

---

## Gotchas

| Thing | Why it bites |
| --- | --- |
| **Temperature must shape `p` and `q` identically** | Shape one only and the ratio compares two different models. The guarantee is void |
| **Tokenizers must match** | `p(x)` and `q(x)` must mean the same token `x` |
| **fp32 is the default here** | So the distribution checks are not fighting numerical noise |
| **`--device cpu` works for stages 0-3** | Keep `--max-new` small; the wall-clock numbers invert |

---

## Next

```bash
make stage0                 # no model, no GPU
make stages                 # stages 1-4, Qwen2.5 1.5B / 0.5B
make sweep                  # k sweep -> results/sweep.csv
```

Run the sweep **on the machine you intend to serve from**. A drafter that is right almost every time can still be the wrong drafter.

## Open questions

- Which k wins on your GPU? The sweep answers it; the CPU numbers above do not transfer.
- Does prompt-lookup beat the 0.5B drafter on your actual prompts? `measure.py --prompt code` vs `--prompt prose` splits sharply.
