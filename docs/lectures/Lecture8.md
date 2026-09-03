# Inside the GPU: memory and the roofline

> **Module thesis:** hiding latency is not the same as having bandwidth. What ultimately limits a GPU is how many bytes it can move per FLOP it performs — which is why the same LLM on the same silicon is memory-bound when it serves and compute-bound when it trains.

This is the second of two lectures on GPU internals. [Lecture 7](Lecture7.md) explained **how a GPU keeps thousands of ALUs busy**. This one asks **what limits performance**: the memory hierarchy, coalescing, the roofline, and what both imply for AI workloads.

---

[Last lecture](Lecture7.md) we built a machine that never waits: thousands of resident threads, and a scheduler that runs somebody else whenever one warp stalls.

!!! question "💬 Three questions before we go on"

    1. Why does a GPU have thousands of ALUs?
    2. Why does it keep thousands of threads resident?
    3. What happens when one warp waits?

    ??? hint "Answers"
        1. **Parallelism.** The workload is wide, regular and independent, so the transistor budget goes to arithmetic instead of control.
        2. **Latency hiding.** Residency is what guarantees the scheduler always has an eligible warp.
        3. **Run another warp.** Nothing stalls, and it costs nothing, because the waiting warp's registers never left the register file.

So: we have succeeded in hiding latency. Thousands of threads are ready to run.

That success creates the next problem.

> **What happens when the GPU simply cannot get the bytes fast enough?**

Latency hiding solves *waiting*. It does nothing about **bandwidth**. If every resident warp wants data and the memory system can only deliver 320 GB/s, no amount of scheduling cleverness invents more bytes.

This lecture follows the data.

---

## 7. Feeding the ALUs

The memory hierarchy is roughly:

```text
             fastest
                │
                ▼
           Registers
                │
               L1
                │
               L2
                │
          Last-level cache
                │
              VRAM
                │
                ▼
             slowest
```

Two things about this hierarchy matter more than the list itself.

---

### Shared memory is not a cache

A cache is controlled by hardware.

Shared memory is controlled by your program.

```text
cache:
    hardware decides

shared memory:
    program decides
```

|                             | L1 cache                               | Shared memory                       |
| --------------------------- | -------------------------------------- | ----------------------------------- |
| Who decides what is stored? | hardware                               | **program**                         |
| How do you use it?          | implicit                               | explicit loads/stores               |
| When does it help?          | when access patterns happen to benefit | when you deliberately arrange reuse |

Consider tiled matrix multiplication.

A CPU may restructure its loops so that a tile remains in cache.

The CPU then *hopes* the cache retains it.

On a GPU, a kernel can explicitly do:

```text
VRAM
  │
  ▼
shared memory
  │
  ├── reuse
  ├── reuse
  └── reuse
```

Shared memory is therefore best thought of as:

> **a programmer-managed scratchpad close to the arithmetic units.**

---

### GPUs cache too

There is a common misconception:

> "GPUs don't have caches."

This GPU has:

```text
96 MB last-level cache
```

That is six times the 16 MB L3 cache of the CPU in the same machine.

The difference is not that GPUs do not cache.

The difference is how they combine:

```text
cache
+
high memory bandwidth
+
massive parallelism
+
latency hiding
```

That 96 MB number is not trivia. It is about to explain a 3× discrepancy in the next benchmark.

---

## 8. Coalescing: how threads access memory

A warp contains many threads.

Suppose all 32 threads need one float:

```text
T0 → A[0]
T1 → A[1]
T2 → A[2]
...
T31 → A[31]
```

These accesses are contiguous.

The memory system can service them efficiently.

A simplified view:

```text
T0 T1 T2 T3 ... T31
 ↓  ↓  ↓  ↓      ↓
┌──────────────────┐
│ contiguous data  │
└──────────────────┘
```

Now consider scattered accesses:

```text
T0  → A[0]
T1  → A[1024]
T2  → A[2048]
T3  → A[3072]
...
```

The same number of useful bytes are requested, but many more memory lines may have to be fetched — the cache line on this GPU is 128 B, so a strided warp can touch 32 separate lines to collect 32 floats it could have got in one.

This is the intuition behind **coalesced memory access**.

---

### Measure it

The benchmark reads the same elements in different orders. The only thing that changes is the access pattern.

!!! question "💬 Predict two things: how much slower is stride-1024 than contiguous — and does that ratio change when the buffer grows from 16 MB to 256 MB?"

    ??? hint "Answer"
        It gets **more than 3× worse** on the larger buffer:

        ```text
        16 MB:
            stride 1024 → 4.54× slower

        256 MB:
            stride 1024 → 15.63× slower
        ```

        Same kernel, same pattern, same useful bytes. Only the working-set size changed — and it crossed the 96 MB last-level cache.

```text
Buffer: 16 MB in + 16 MB out

contiguous :   0.234 ms    143.5 GB/s useful
stride 32  :   0.267 ms    125.7 GB/s useful
stride 128 :   0.601 ms     55.8 GB/s useful
stride 1024:   1.062 ms     31.6 GB/s useful
```

For a larger working set:

```text
Buffer: 256 MB in + 256 MB out

contiguous :   2.360 ms    227.5 GB/s useful
stride 32  :   4.750 ms    113.0 GB/s useful
stride 128 :  24.173 ms     22.2 GB/s useful
stride 1024:  36.882 ms     14.6 GB/s useful
```

The same useful data.

The same kernel.

Only the access pattern changes.

---

### Why does working-set size matter?

This GPU has:

```text
96 MB last-level cache
```

The 16 MB working set fits inside it.

The scattered access pattern still fetches whole cache lines, but later accesses can benefit from lines that remain resident. The cache quietly rescues the bad pattern.

The 256 MB working set is much larger than the cache.

Lines are evicted before the remaining useful data can be reused. Nothing rescues the bad pattern, and it costs what it always should have cost.

The lesson:

> **Benchmark memory behavior at the size you will actually run.**

A memory-access pattern that looks harmless on a toy dataset can become disastrous when the working set exceeds cache capacity.

---

## 9. What actually limits performance?

We have now met two fundamentally different resources:

```text
compute
    │
    └── arithmetic throughput

memory
    │
    └── bytes per second
```

A workload can be limited by either.

This is the basic idea behind the **roofline model**.

---

### Compute roof

From [Lecture 7b](Lecture7b.md):

```text
2,304 FP32 ALUs
× 2 FLOPs per FMA
× 2.3 GHz

≈ 10.6 TFLOP/s
```

That is the theoretical compute roof.

---

### Memory roof

The memory bus is:

```text
160 bits
```

and the effective memory data rate is:

```text
16 Gbps per pin
```

Therefore:

```text
memory bandwidth
=
160 bits × 16 Gbps / 8
=
320 GB/s
```

The memory clock reported by the device is:

```bash
cat /sys/class/drm/card*/device/pp_dpm_mclk
```

It tops out at:

```text
1000 MHz
```

That is the clock frequency, not the effective transfer rate.

GDDR6 transfers multiple bits per clock, so always distinguish:

```text
physical clock
```

from:

```text
effective data rate
```

---

### The ridge point

We now have:

```text
compute roof  = 10.6 TFLOP/s
memory roof   = 320 GB/s
```

Divide one by the other:

```text
10.6e12 FLOP/s
----------------
320e9 bytes/s

≈ 33 FLOP/byte
```

This is the **ridge point**.

It tells us approximately where a workload changes from memory-bound to compute-bound.

```text
performance
    ▲
    │                    compute roof
    │              ─────────────────────
    │            /
    │          /
    │        /
    │      /
    │    /
    │  /
    │/
    └──────────────────────────────────►
       arithmetic intensity

             ↑
          33 FLOP/byte
          ridge point
```

Below roughly 33 FLOP/byte this GPU is memory-bound. Above it, compute-bound.

The exact boundary is an approximation—real performance depends on the kernel, cache behavior, instruction mix, occupancy, and how closely the implementation approaches each roof.

But it gives an extremely useful first diagnosis:

> **Arithmetic intensity tells us which resource we are asking too much of.**

---

## 10. Where does matrix multiplication sit?

Consider FP32 matrix multiplication:

```text
C = A × B
```

for:

```text
N = 2048
```

The work is approximately:

```text
2N³
=
17.2 GFLOP
```

The input and output traffic, under the simple model, is:

```text
3N² × 4 bytes
=
50.3 MB
```

Therefore:

```text
arithmetic intensity
=
17.2e9 / 50.3e6

≈ 341 FLOP/byte
```

Compare:

```text
GPU ridge:
    33 FLOP/byte

matmul:
    341 FLOP/byte
```

Matmul sits far to the right of the ridge.

The roofline therefore predicts:

> **This matmul should be compute-bound.**

Measured on this GPU:

```text
9,047 GFLOP/s
```

compared with:

```text
10,600 GFLOP/s theoretical
```

That is about:

```text
85% of the FP32 compute roof
```

The prediction and the measurement agree — which is what earns the model our trust for the next question.

---

## 11. The payoff: what does this mean for AI?

Now we can finally answer the question that matters for ML systems:

!!! question "💬 Is running an LLM compute-bound or memory-bound?"

    ??? hint "Answer"
        The question is malformed. It depends on **which phase of the workload you mean** — and the two phases of the *same model on the same GPU* land on opposite sides of the ridge.

---

### LLM decode: one token at a time

Consider generating one token with batch size 1.

The model's weights must be read from memory and used to perform relatively little computation per byte loaded.

A rough estimate is:

```text
≈ 2 FLOPs per parameter
```

For a model stored in 16-bit precision:

```text
≈ 2 bytes per parameter
```

giving an arithmetic intensity on the order of:

```text
1–2 FLOP/byte
```

Even allowing for implementation details, this is far below the GPU's:

```text
≈ 33 FLOP/byte
```

ridge point.

Therefore:

> **LLM decode at small batch size is typically memory-bound.**

The GPU spends much of its time moving model weights rather than doing arithmetic.

The mental model is:

```text
VRAM
 │
 │ stream weights
 ▼
ALUs
 │
 │ tiny amount of useful computation
 ▼
next token
```

This is why reducing the number of *bytes per parameter* — quantization — can be more valuable than increasing raw arithmetic throughput.

---

### Prefill and training: many tokens at once

Now consider training, or processing a long prompt.

The same weights can be reused across many pieces of work:

```text
             ┌── token 1
weights ─────┼── token 2
             ├── token 3
             ├── token 4
             └── ...
```

The same bytes loaded from memory now support much more computation.

Arithmetic intensity can move into the hundreds of FLOP/byte — the matmul regime we just measured at 85% of the compute roof.

That puts the workload to the right of the ridge:

```text
memory-bound
     │
     │
     └───────────────────► compute-bound
                    ↑
                batching
```

Therefore:

> **Training and sufficiently batched/large-shape matrix operations are often compute-bound.**

---

### Same model, opposite bottleneck

This is the result worth remembering:

```text
                 LLM
                  │
          ┌───────┴────────┐
          │                │
        decode           prefill
          │              / training
       batch 1         large batch
          │                │
          ▼                ▼
    ~1–2 FLOP/byte    hundreds FLOP/byte
          │                │
          ▼                ▼
    memory-bound       compute-bound
```

The silicon did not change.

The model did not change.

The **workload shape** changed.

> **Same model. Same GPU. Different bottleneck.**

And the lever that moves it is not a hardware purchase:

> **Batching is a software decision that can move a workload across the roofline.**

That is why a GPU can be compute-bound while training a model and memory-bound while serving the very same model.

---

## 12. Both lectures in one picture

```text
GPU architecture
      ↓
many ALUs
      ↓
many resident warps
      ↓
latency hiding
      ↓
SIMT
      ↓
divergence / occupancy
      ↓
memory hierarchy
      ↓
coalescing / bandwidth
      ↓
arithmetic intensity
      ↓
roofline
      ↓
┌───────────────┬────────────────┐
│ LLM decode    │ training       │
│ memory-bound  │ compute-bound  │
└───────────────┴────────────────┘
```

The GPU is a machine designed around one fundamental idea:

> **Keep the arithmetic units busy by having enough independent work available to run while other work waits.**

Everything in this module is either that idea, the hardware that pays for it, or the bill that arrives when the memory system cannot keep up.

---

## Run it yourself

Inspect your GPU:

```bash
rocminfo
# or
nvidia-smi -q
```

Inspect clock levels on Linux/AMD:

```bash
cat /sys/class/drm/card*/device/pp_dpm_sclk
cat /sys/class/drm/card*/device/pp_dpm_mclk
```

Run the benchmark:

```bash
python code/gpu_internals/warp_costs.py
```

The benchmark code:

[`code/gpu_internals/warp_costs.py`](https://github.com/Ankush-Chander/DS635-ml-system-engineering/tree/main/code/gpu_internals)

The Triton benchmark is designed to run across NVIDIA and AMD hardware and can also be run on a free Colab/Kaggle GPU.

---

<!--# Exercises

## 1. Build a roofline

Build the roofline for a different GPU.

Calculate:

```text
compute roof
memory roof
ridge point
```

Then place a:

```text
341 FLOP/byte
```

matmul on the graph.

Is it memory-bound or compute-bound?

---

## 2. Find when cache stops helping

Sweep:

```text
stride
```

and:

```text
working-set size
```

on your own GPU.

Find the point at which increasing the working set causes the cache to stop rescuing the scattered access pattern.

---

## 3. Move a workload across the ridge

Estimate the arithmetic intensity of decoding one token from a model you use, at batch size 1.

Then recompute it at batch size 64.

At what batch size does the workload cross your GPU's ridge point?

----->

## Appendix C — Tensor and matrix units

Modern GPUs may contain dedicated matrix-multiply hardware.

NVIDIA calls these **Tensor Cores**.

AMD uses terms such as **Matrix Cores** for corresponding functionality.

The RX 6700M used for the measurements in this module does not have dedicated matrix cores, so the FP32 matmul result comes from ordinary FP32 arithmetic lanes.

That is why:

```text
9,047 GFLOP/s
```

can reach about:

```text
85% of the 10.6 TFLOP/s FP32 roof
```

without dedicated matrix hardware.

The important distinction is:

> **A workload's performance depends on which computational roof the hardware exposes for the requested precision and operation.**

Dedicated matrix units raise that roof for supported matrix operations and precisions.

---

## Appendix D — NVIDIA and AMD terminology

The terminology table for both lectures is in [Lecture 7c, Appendix A](Lecture7c.md#appendix-a-nvidia-and-amd-terminology).

---

## References

1. Vijay Janapa Reddi, [*Machine Learning Systems*](https://mlsysbook.ai) — Ch. 11: AI Acceleration
2. NVIDIA, [*CUDA C++ Programming Guide*](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — Hardware Implementation, SIMT, divergence, occupancy
3. AMD, [*RDNA 2 Instruction Set Architecture Reference Guide*](https://gpuopen.com/amd-gpu-architecture-programming-documentation/) — compute units, wavefronts, vector register file
4. [Triton documentation](https://triton-lang.org)
5. Course code: [`code/gpu_internals`](https://github.com/Ankush-Chander/DS635-ml-system-engineering/tree/main/code/gpu_internals)
