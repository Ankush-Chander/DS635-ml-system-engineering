# Matmul to silicon: execution and latency hiding

> **Module thesis:** a GPU is not built to make one thread fast — it is built to
> never have an idle ALU. This lecture is the machinery that achieves that.

This is a guided descent. We start at one line of Python and go down until we
reach the lanes that do the arithmetic, stopping at every layer that makes a
decision on your behalf. It is split into three parts, one session each, with
depth numbering running continuously across them:

| Part | What it covers | Depths |
| ---- | -------------- | ------ |
| [7a — From Python to a kernel launch](Lecture7a.md) | the dispatcher, the doorbell, what a kernel is, and the SPMD inversion | 1–2 |
| [7b — The throughput machine](Lecture7b.md) | the hardware hierarchy, over-subscription, and latency hiding measured directly | 3 |
| [7c — SIMT, tiles, and occupancy](Lecture7c.md) | warps, divergence, one block = one tile of `C`, the residency budget — plus the synthesis and capstone | 4–5 |

```text
"how C = A @ B reaches the GPU and gets executed"
├── ✗ what matmul computes (O(n³) work on O(n²) data)     — assumed
├── ✗ latency machines vs throughput machines             — assumed
├── 1. the software stack: dispatcher → BLAS → launch       ◀ Lecture 7a
├── 2. what a kernel really is (the SPMD inversion)         ◀ Lecture 7a
├── 3. the throughput machine: residency & the free switch  ◀ Lecture 7b
├── 4. SIMT: grids, blocks, warps of 32, divergence         ◀ Lecture 7c
├── 5. one block = one tile of C                            ◀ Lecture 7c
├── 6. coalescing & broadcast                               → Lecture 8
├── 7. the memory hierarchy and the FLOPs/byte wall         → Lecture 8
└── 8. tensor cores & torch.compile                         → Lecture 8
```

Depths 1–5 are the execution model: how work is created, named, scheduled and
kept flowing. Depths 6–8 are about *feeding* that machine, and they are
[Lecture 8](Lecture8.md).

Each part is also a runnable notebook — every measurement on those pages was
produced by a cell you can re-run on your own GPU, or on a free Colab T4 via the
badge at the top of each part.

The NVIDIA/AMD terminology table, this laptop's GPU inventory, and the
references for all three parts are collected at the end of
[Lecture 7c](Lecture7c.md#appendix-a-nvidia-and-amd-terminology).
