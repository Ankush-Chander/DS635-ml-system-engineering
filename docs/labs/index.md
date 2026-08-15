# Labs

Labs are where the course's claims get tested against hardware you control. Each one pairs with a lecture: the lecture derives a mechanism, the lab measures its consequences.

Every lab runs on **Google Colab** (free GPU) or **your own machine**. Nothing requires lab hardware, and nothing requires a particular vendor: **NVIDIA (CUDA)**, **AMD (ROCm)** and **Apple silicon (MPS)** are all supported. Where a backend genuinely cannot run an experiment, it is skipped and recorded rather than failed, and you are marked on explaining why it does not apply.

| Lab | Pairs with | You measure | Marks |
|---|---|---|---:|
| [Lab 5/6 — GPU Job Submission](Lab5_6.md) | [Lecture 5/6](../lectures/Lecture5_6.md) | Establishing theoretical peak · submission vs execution · launch cost · fences · pinned memory · stream overlap · CUDA Graphs | 100 |

## How labs are marked

Two halves, assessed differently:

- **Automatic** — a script checks *relationships between your own numbers*, never their magnitudes against a reference. Your GPU is yours; a Colab T4 and an RX 6700M disagree on every value in these labs and both are correct.
- **Rubric** — your written predictions and explanations, marked on whether they name the mechanism rather than on whether the number was what you expected.

A wrong prediction you then explain earns full marks. A blank prediction earns none.

## A standing note on honest results

Several results in these labs come out differently on different hardware, and some come out *opposite* to the textbook expectation. Those are deliberately not auto-marked, and reporting them costs you nothing:

- pinned memory sometimes gives no speedup at large transfer sizes
- CUDA Graphs sometimes make GPU-side time *worse*
- stream overlap is unobservable when a GPU's copy and compute times are very unbalanced
- a vendor's advertised peak may disagree with the one you derive from the hardware's own reported numbers — and the hardware is not always the one that is right

> **An honest surprising measurement, correctly explained, scores higher than a tidy expected one.**

Report what your machine did.
