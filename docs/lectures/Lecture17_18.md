# Choosing a drafter

> **Module thesis:** [Lecture 15/16](Lecture15_16.md) showed that a speculative decoder is correct for **any** drafter — a bad one costs speed, never accuracy. That freedom moves the entire engineering problem into one question: what is the cheapest predictor that agrees with this target often enough to pay for itself? The families below answer it differently, and the right answer changes with the workload, the checkpoint and the VRAM you have left.

**Prerequisite:** [Speculative decoding](Lecture15_16.md)

---

## 1. What the job actually is

A drafter must be **much cheaper than the target** and **agree with it often**. Those pull against each other, and every family below is a different point on that trade.

Two constraints bound the whole space:

- **The tokenizer must match the target's exactly.** The accept test compares `p(x)` against `q(x)` for the same integer `x`.
- **The drafter's cost is paid whether or not its tokens are accepted.** A rejected block still cost its draft passes.

One number summarises the result, and it is worth carrying through the whole lecture: **acceptance length**, the mean tokens committed per verification pass, between 1 and k+1.

---

## 2. Training-free: predict from the text, not from a model

Do not run a model at all. Search the context for a matching prefix and propose whatever followed it.

- **N-gram / prompt lookup** — match the last *n* tokens against earlier context, copy the continuation.
- **Suffix decoding** — the same idea with better data structures and a cache of past generations.

| | |
| --- | --- |
| **Cost** | No weights, no VRAM, no training, no extra forward passes |
| **Wins on** | Code editing and refactoring, RAG quoting, JSON echoing — anything where the output repeats the input |
| **Useless for** | Genuinely novel prose |

**Always the first thing to try**, because evaluating it is nearly free — there is nothing to download and nothing to fit in memory. It is also the honest baseline: if prompt lookup already captures most of the available speedup on your traffic, a trained drafter has to beat *that*, not beat nothing.

---

## 3. Self-drafting: the target, run cheaply

Use the target as its own drafter by running a reduced version of it — exit early from a subset of layers, or skip layers.

No extra parameters, and no acquisition cost. But the draft passes are not *that* cheap, and acceptance is mediocre for a structural reason: **a truncated model is a different model.** It does not approximate the target's distribution so much as replace it with a worse one.

Rarely the right answer when any of the alternatives below are available.

---

## 4. Learned heads: read the target's own hidden state

Small trained modules attached to the target that predict several positions ahead from its hidden states.

| Head | How it drafts | Trade |
| --- | --- | --- |
| **Medusa** | Multiple independent heads, one per future position | Simple and fast to train; the heads do not condition on each other, so later positions degrade |
| **EAGLE** | Autoregressively in *feature* space, each guess conditioned on the previous | Substantially higher acceptance than Medusa at similar size |
| **MTP** | Trained *jointly with the base model* as a pre-training objective, shipped inside the checkpoint | Best alignment with the target; **zero acquisition cost when present** |

The reason this family wins against a separate model of equal size:

> A head consumes the target's own hidden state. It predicts what **this** model would say, not what **a** model would say.

**The single-layer caveat.** A head with one layer reused across k positions conditions on its own unverified output. Acceptance at position 2 and beyond falls away faster than the chain alone predicts — which shows up in the diagnosis table of Lecture 15/16 as *"position 2+ acceptance far below position 1"*, and is fixed by lowering `k`, not by drafting harder.

---

## 5. Block drafters: the whole window in one pass

A small model that proposes the **entire draft window in a single pass** rather than one token at a time. It keeps candidates at every position and runs a light selector to trace a coherent path through them.

**This breaks the cost model every other family obeys.**

| | Autoregressive drafter | Block drafter |
| --- | --- | --- |
| Passes to draft `k` | `k` sequential | **1** |
| Cost in `k` | linear | **flat** |
| Optimal `k` | early, 3–4 | its trained block size |

Because cost is flat, the usual "stop at k≈3–4" advice does not apply — that rule is a statement about *autoregressive* drafters, where each extra position costs another pass while contributing a geometrically smaller return. With a flat cost, you run the window the drafter was trained at.

### Anatomy of one

`z-lab/Qwen3.8-27B-DFlash2`, the block drafter for `Qwen3.8-27B`:

| Field | Value |
| --- | --- |
| Size | 3.85 GB, bf16 |
| Layers | 5, all sliding attention, `sliding_window: 2048` |
| `block_size` | **8** → `num_speculative_tokens: 7` |
| Selector | `selector_rank: 256`, `selector_top_k: 16` |
| Target taps | layers `[5, 19, 33, 47, 61]` of 64 |
| `is_causal` | **False** |

Two fields carry most of the design. **`is_causal: False`** is what lets it fill a whole block at once rather than extending a sequence — the same property that makes [bidirectional models](Lecture13_14.md) able to emit many positions in one pass. **The target taps** are how it reads the target's hidden state at five depths, which is the learned-head trick from §4 applied to a block.

**It must run its trained window.** Measured on that drafter, shorter windows are *slower*, not safer — the model was fitted to produce 8 positions and produces them whether you ask for 8 or 3.

---

## 6. A separate draft model

A small model from the same family — a 0.5B drafting for a 7B, say.

Works, and is sometimes the only option. But it is **strictly worse than a head of equal size**, because it re-derives the context from scratch instead of reading the target's hidden state. It also costs 1–2 GB of VRAM and demands an exactly matching tokenizer, which rules out most cross-family pairings.

Useful mainly as a teaching device and as a fallback when nothing else exists for your checkpoint. The DS635 lab uses one for exactly that reason — `code/spec_decode/` pairs Qwen2.5-0.5B with Qwen2.5-1.5B, because the pair is small enough to run anywhere and shares a tokenizer by construction.

---

## 7. Linear drafts and trees

Everything so far assumes a **linear** draft: one candidate sequence. The generalisation proposes a *tree* — several alternatives at each position — and verifies the whole tree in one pass, using an attention mask that stops branches from seeing each other.

- **Trees raise acceptance**, because the target only has to agree with *some* branch rather than one specific guess.
- **Trees cost positions.** More positions per verification pass eventually reintroduces the compute pressure that speculation was exploiting in the first place — the pass stops being nearly free and starts being a real prefill.

Medusa and EAGLE both use tree verification; simple MTP setups usually do not. Block drafters sit between the two: they hold candidates at every position like a tree, then collapse them to one linear draft before verification.

---

## 8. The decision

```text
block drafter published for this model?   -> usually fastest, if VRAM allows
checkpoint ships an MTP head?             -> the zero-acquisition default
output largely copies the input?          -> n-gram / suffix decoding
trained EAGLE head available?             -> use it
VRAM plentiful and a sibling exists?      -> small draft model
else                                      -> skip speculation
```

Read top to bottom and stop at the first yes. The ordering is by **acquisition cost against expected acceptance**, not by sophistication — prompt lookup sits third because when it applies it beats trained drafters that cost gigabytes.

!!! question "💬 Your checkpoint ships an MTP head and a block drafter exists for it. You are serving 128k-token agentic sessions on a single card. Which do you choose?"

    ??? hint "Answer"
        Probably the **MTP head**, despite the block drafter being faster in tokens/sec. The block drafter's weights come out of the KV pool, and at long context that is the binding constraint — Lecture 15/16 records 3.3 GiB of drafter cutting the KV pool from 309,329 to 168,340 tokens and halving concurrency at full context. A 22% decode gain is a poor trade for halving how many 128k sessions fit. **The answer inverts at short context**, where KV pressure is slack and the throughput win is free. This is why the decision list ends at "if VRAM allows" rather than at "fastest".

---

## 9. What to measure before choosing

Drafter choice is one of the few places where the published number is almost never yours: acceptance is a property of the **pair and the workload**, so a vendor's benchmark on their traffic predicts little about yours.

1. **Baseline with speculation off**, fixed output length.
2. **Try prompt lookup first** — it costs nothing to evaluate and sets the bar.
3. **Check acceptance length, not throughput** — near 1.0 means the drafter is pure overhead.
4. **Measure the KV pool** with the drafter loaded and without.
5. **Re-measure at your real concurrency** — Lecture 15/16 §5 on why batch 1 flatters speculation.
6. **Sweep `k`**, and read the sweep knowing which cost model your drafter obeys: linear or flat.

**Lab:** `code/spec_decode/` implements the separate-draft-model and prompt-lookup families behind one interface, so the two can be swapped and swept against the same target.

---

## References

1. Cai et al., [*Medusa*](https://arxiv.org/abs/2401.10774) (2024) — multiple decoding heads, tree attention
2. Li et al., [*EAGLE*](https://arxiv.org/abs/2401.15077) (2024) — feature-space autoregressive drafting
3. [Lecture 15/16](Lecture15_16.md) — the mechanism, the guarantee, the measurements
4. [Lecture 9/10](Lecture9_10.md) — the memory wall these drafters are spending
