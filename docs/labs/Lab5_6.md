# Lab 5/6 — GPU Job Submission

> **Lab thesis:** there is no *call the GPU* instruction. Every kernel launch is a command appended to a queue, one doorbell write, and a fence that comes back later. You will measure that protocol from the outside — and watch it manufacture a throughput number that beats the hardware.

This is the experimental half of [Lecture 5/6 — GPU Job Submission](../lectures/Lecture5_6.md). The lecture derived the mechanism from a movie file and an `lspci` listing. Here you observe its consequences on a real device.

[**Open the notebook in Colab**](https://colab.research.google.com/github/Ankush-Chander/DS635-ml-system-engineering/blob/main/docs/labs/Lab5_6_gpu_job_submission.ipynb) · [read it on this site](Lab5_6_gpu_job_submission.ipynb) · [download from GitHub](https://github.com/Ankush-Chander/DS635-ml-system-engineering/blob/main/docs/labs/Lab5_6_gpu_job_submission.ipynb)

---

## What you need

**Google Colab** — open the notebook, then `Runtime > Change runtime type > T4 GPU`. Everything runs unchanged; no installs.

**Your own machine** — **NVIDIA (CUDA)**, **AMD (ROCm)** or **Apple silicon (MPS)**. `torch.cuda` is the correct namespace on ROCm too.

Apple's Metal backend exposes no command streams, no graph capture and no pinned host memory, so three measurements are skipped there automatically. The export records what was skipped and **you are not penalised** — but you are still asked to explain *why* each one does not apply to a unified-memory machine, and those explanations carry the marks.

!!! warning "If you are on a laptop, plug in the charger"
    This lab deliberately drives the GPU to sustained peak power. On battery, a discrete mobile GPU can draw more than the battery and voltage regulators can supply, and the machine **hard-resets with nothing written to the logs** — no error, no traceback, just an instant reboot. It is not a bug you can debug from Python.

    The notebook's preflight cell checks your power source and scales the workload down automatically if you are not on mains. Leave `SAFE_MODE` alone unless you are plugged in. This is a real failure that happened while the lab was being written, not a hypothetical.

## What you do

Five parts, each following the same three steps — and the order matters:

1. **Predict** — write your prediction in the cell provided *before* running anything.
2. **Measure** — run the cell.
3. **Explain** — say which mechanism from the lecture produced the number.

| Part | Question | Marks |
|---|---|---:|
| 0 | What am I measuring on? | 5 |
| 1 | **What is my GPU's peak, and why is that hard to answer?** | 20 |
| 2 | Is submission the same as execution? | 20 |
| 3 | What does one submission cost, and what does a fence cost? | 20 |
| 4 | Why do `pin_memory` and `non_blocking` exist, and do streams really overlap? | 20 |
| 5 | Can submission be amortised away? *(CUDA Graphs)* | 5 |
| 6 | Place every term on the control-plane / data-plane board | 10 |

**Part 1 comes first for a reason.** Everything afterwards is compared against one number — Part 2 hinges on producing a measurement that *exceeds* your GPU's peak — so that number has to mean something before it can be beaten. Establishing it is genuinely awkward: you derive it from what the hardware reports, look up what the vendor advertises, and measure what you can actually reach, and the three disagree.

Each term in `units × lanes × 2 × clock` is a trap. On the course's own RX 6700M, PyTorch reports `multi_processor_count = 18` for a 36-CU card — RDNA counts work-group processors, not compute units, so the naive derivation lands at exactly half. `get_device_capability()` returns `(10, 3)` there too, meaning *gfx1030*, not compute capability 10.3. Vendors quote tensor-core, sparsity and lower-precision numbers in the same breath as the FP32 figure you actually want.

## What you submit

**Two files.** The JSON carries your measurements; the notebook carries your reasoning. A JSON without its notebook cannot be marked.

1. **`submission_<roll>.json`** — written by the notebook's final cell. Set `ROLL_NUMBER` and `NAME` in the Part 0 setup cell first, or the export will refuse to run.
2. **The executed notebook** — `File > Download > .ipynb`, with outputs intact and every 📝 cell filled in.

Run every cell top to bottom before exporting. The export cell reports how many experiments it recorded and warns you if any are missing.

## How it is marked

**40 marks are automatic**, checked from your JSON. Every check tests a *relationship between your own numbers*, because there is no answer key — your hardware is yours:

| Checked | Why it holds on any GPU |
|---|---|
| Unsynchronised timer ≥10× faster than synchronised | submission is not execution |
| Implied no-sync throughput exceeds your GPU's peak | that is the whole point of Part 1 |
| `synchronize()` and CUDA events agree within 15% | two independent mechanisms must agree |
| Launch-bound signature present in the size sweep | per-launch cost is roughly constant |
| Syncing every op costs ≥1.5× syncing once | draining the pipeline is not free |
| Two streams beat one stream | copy engines are separate queues |
| Per-launch cost lands in 1–20 µs | sanity band on the protocol |
| Your short answers match your own recorded numbers | you read your own data |
| Peak established on every route your hardware allows, measured ≤ vendor | Part 1 was done honestly |

Plus provenance (device, backend, declared peak, roll number) and completeness (all eight experiments recorded — anything skipped by your backend still counts as recorded).

**Four things are deliberately not auto-checked**, because the honest answer varies by machine: whether pinning helps at large sizes, whether CUDA Graphs improve GPU-side time, whether overlap is observable when your GPU's copy and compute times are very unbalanced, and anything your backend cannot do at all. Reporting "no benefit" or "unavailable" costs you nothing — a validated Apple-silicon submission with three skipped experiments scores full automatic marks.

**60 marks are rubric-marked** from your written cells. Predictions are marked for being *made*, not for being right. Explanations are marked on whether they name the mechanism — ring, doorbell, fence, DMA, pinning, stream. A number with no mechanism scores about half. A mechanism that contradicts your own data scores less than an honest *"I measured X, I expected Y, here is my best explanation for the gap."*

## Academic integrity

Timing measurements carry several digits of run-to-run noise, so two independent runs never produce identical values. Submissions sharing a measurement vector are treated as copied outputs, and the environment fingerprint and run timestamps in each JSON are checked alongside.

Discussing mechanisms with classmates is encouraged. Sharing measurement files is not — and the numbers are the one thing you cannot borrow, because they describe *your* machine.

## Related

- [Lecture 5/6 — GPU Job Submission](../lectures/Lecture5_6.md) — the mechanism this lab measures
- [Lecture 7 — Inside the GPU: execution and latency hiding](../lectures/Lecture7.md) — what happens after the dispatch packet lands
