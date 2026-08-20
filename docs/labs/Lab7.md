# Lab 7 — The concurrency budget

> **Lab thesis:** a GPU is not fast because its threads are fast. It is fast because thousands of them are in flight at once. This lab makes you *derive* how much work your GPU must have in flight to run at full speed — from two numbers you measure yourself — and then find the exact grid size at which it stops being able to.

This is the experimental half of [Lecture 7 — Matmul to silicon](../lectures/Lecture7.md). The lecture argued latency hiding qualitatively: over-subscribe the machine and memory latency disappears behind other warps' arithmetic. It never put a number on *how much* over-subscription is enough. Here you produce that number, and then test it.

[**Open the notebook in Colab**](https://colab.research.google.com/github/Ankush-Chander/DS635-ml-system-engineering/blob/main/docs/labs/Lab7_concurrency_budget.ipynb) · [read it on this site](Lab7_concurrency_budget.ipynb) · [download from GitHub](https://github.com/Ankush-Chander/DS635-ml-system-engineering/blob/main/docs/labs/Lab7_concurrency_budget.ipynb)

---

## What this is not

It is not the lecture notebook with the answers removed. Nothing you run here appears in [Lecture 7](../lectures/Lecture7.md):

| The lecture showed you | This lab makes you produce |
|---|---|
| that a copy kernel speeds up as you add blocks | *why it stops speeding up exactly where it does* — predicted before you look |
| that the GPU holds ~16× more threads than it can execute | your GPU's own concurrency requirement, in bytes, derived from latency × bandwidth |
| that `multi_processor_count` is ambiguous on RDNA | that same count **measured from outside**, with no driver query, from where a staircase steps |
| a warp-width sweep and a divergence cost | neither — Part 4 is about the *block* scheduler, not the warp scheduler |

Three of the four kernels here — a pointer chase, a fixed-work grid sweep, a bandwidth ramp — do not exist anywhere in the course material. You write them.

## What you need

**Google Colab** — open the notebook, then `Runtime > Change runtime type > T4 GPU`. The first cell installs Triton if the runtime does not have it.

**Your own machine** — **NVIDIA (CUDA)** or **AMD (ROCm)**. `torch.cuda` is the correct namespace on ROCm.

!!! warning "Apple silicon cannot run this lab"
    Every experiment is a Triton kernel, and Triton has no Metal backend. Unlike Lab 5/6, there is no degraded path: MPS is not a supported target here. Use Colab.

!!! warning "If you are on a laptop, plug in the charger"
    Part 4 holds the GPU at sustained peak arithmetic for seconds at a time. On battery, a discrete mobile GPU can draw more than the battery and its regulators can supply, and the machine **hard-resets with nothing in the logs**. The notebook's preflight cell detects your power source and scales every sweep down automatically if you are not on mains. Leave `SAFE_MODE` alone unless you are plugged in.

## What you do

Five parts. Each follows the same three steps, and the order matters:

1. **Predict** — write your prediction in the cell provided *before* running anything.
2. **Measure** — run the cell.
3. **Explain** — say which mechanism from the lecture produced the number.

| Part | Question | Marks |
|---|---|---:|
| 0 | What does my GPU claim to be? | 5 |
| 1 | **How long is one memory access?** *(pointer chase)* | 20 |
| 2 | How many bytes per second can I actually move? | 15 |
| 3 | **Little's Law — how much must be in flight?** | 25 |
| 4 | **The staircase — what is my scheduling quantum?** | 25 |
| 5 | Spend the budget: the tail effect, and a design rule | 10 |

**Parts 1 and 2 exist to feed Part 3.** Latency and bandwidth are the two terms of a queueing identity that has nothing to do with GPUs — Little's Law, which is equally true of supermarket checkouts:

```text
bytes in flight  =  bandwidth  x  latency
```

Multiply your own two measurements and you get the amount of data your GPU must have *requested but not yet received*, at every instant, to run at full speed. Divide by the bytes one block has outstanding and you have predicted the knee of your Part 2 curve — before looking at it. On the course's RX 6700M the prediction was 67.7 KB and the measurement 72.0 KB.

**Part 4 is the sharpest result in the lab, and the one that settles an argument from Part 0.** Give every block identical work and no memory traffic, then sweep the block count one at a time. Runtime cannot depend on how much data there is, so it can depend only on how many *rounds* of blocks the scheduler needs — and the plot comes out as a staircase with perfectly flat treads. The width of a tread is a hardware property the vendor does not print on the box, and you get it without asking the driver anything. On the course laptop the steps land at 37, 55, 73, 91, 109 blocks: a period of exactly 18, from a card whose `multi_processor_count` is 18 and whose CU count is 36. Part 4 asks you to decide which of those two numbers you would size a grid with, and defend it.

### Measuring latency is the hard part

The GPU is built specifically to stop you seeing latency. Issue a hundred independent loads and the memory system overlaps them, and you measure throughput instead. So Part 1 makes you build a load the machine *cannot* overlap: a buffer holding one permutation cycle, where each access's **address** is the previous access's **value**. One thread, one block, the rest of the GPU idle — which is the point. You are measuring the unhidden cost of one access, the exact quantity that latency hiding exists to conceal.

Two traps are marked in the notebook and both have bitten this course's own code: a stride of 1 measures the cache line rather than the memory, and an arithmetic chain without a clamp gets closed-formed away by the compiler, leaving you timing an empty loop.

## What you submit

**Two files.** The JSON carries your measurements; the notebook carries your reasoning. A JSON without its notebook cannot be marked.

1. **`submission_lab7_<roll>.json`** — written by the notebook's final cell. Set `ROLL_NUMBER` and `NAME` in the setup cell first, or the export refuses to run.
2. **The executed notebook** — `File > Download > .ipynb`, outputs intact, every 📝 cell filled in.

Run every cell top to bottom before exporting. The export cell reports how many measurements it recorded and names any that are missing.

## How it is marked

**35 marks are automatic**, checked from your JSON. Every check tests a *relationship between your own numbers*, because there is no answer key — your hardware is yours:

| Checked | Why it holds on any GPU |
|---|---|
| Latency rises with footprint and never falls | you cannot get faster by using more memory |
| Slowest latency ≥ 2× the fastest | there is a memory hierarchy, and you found it |
| Memory latency lands in 100–2000 ns | sanity band; outside it, the chase was broken |
| Peak bandwidth ≥ 3× the one-block bandwidth | one block cannot saturate a GPU |
| Little's Law ratio within 8× either way | the identity holds; the residue is what Task 3.2 explains |
| A staircase exists — at least one step ≥ 10% | block scheduling is quantised |
| Treads are flat: spread within a tread < 10% | that is what "quantised" means |
| Step period is within a factor of 2 of the reported unit count | period 18 or 36 both pass; 5 does not |
| Your short answers match your own recorded numbers | you read your own data |

Plus provenance (device, backend, Triton version, roll number) and completeness.

**Three things are deliberately not auto-checked**, because the honest answer varies by machine: how many plateaus your latency staircase has (some GPUs show two, some four, and two levels can share a latency), whether your step period equals `multi_processor_count` or twice it, and how close your Little's Law ratio lands. Reporting "I expected an L2 step and there isn't one" costs you nothing.

**65 marks are rubric-marked** from your written cells. Predictions are marked for being *made*, not for being right. Explanations are marked on whether they name the mechanism — dependent load, prefetch, residency, occupancy, wave, tail. A number with no mechanism scores about half. A mechanism that contradicts your own data scores less than an honest *"I measured X, I expected Y, here is my best explanation for the gap."*

The heaviest single task is **3.2**, which asks you to explain why your predicted and measured bytes-in-flight disagree. It is worth 19 marks on its own, and a ratio of 1.0 does not earn them — the explanation does.

## Academic integrity

Timing measurements carry several digits of run-to-run noise, so two independent runs never produce identical values. Submissions sharing a measurement vector are treated as copied outputs, and the environment fingerprint and run timestamps in each JSON are checked alongside.

Discussing mechanisms with classmates is encouraged. Sharing measurement files is not — and the numbers are the one thing you cannot borrow, because they describe *your* machine.

## Related

- [Lecture 7 — Matmul to silicon](../lectures/Lecture7.md) — the execution model this lab puts numbers on
- [Lab 5/6 — GPU Job Submission](Lab5_6.md) — the submission protocol, one layer above
- `code/gpu_internals/warp_costs.py` — divergence and coalescing microbenchmarks, deliberately *not* reused here
