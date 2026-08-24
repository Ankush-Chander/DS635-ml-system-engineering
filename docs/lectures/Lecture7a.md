---
jupyter:
  jupytext:
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Matmul to silicon 7a: from Python to a kernel launch

> **Module thesis:** a GPU is not built to make one thread fast — it is built to
> never have an idle ALU. Lecture 7 is the machinery that achieves that.

This is a guided descent. We start at one line of Python and go down until we
reach the lanes that do the arithmetic, stopping at every layer that makes a
decision on your behalf. The descent spans three parts, one session each, and
the depth numbering runs continuously across them:

* **7a (this part)** — the software stack: what `C = A @ B` does before any math
  happens, and what a kernel really is *(Depths 1–2)*.
* **[7b — The throughput machine](Lecture7b.md)** — the hardware hierarchy, and
  the latency hiding everything else is built around *(Depth 3)*.
* **[7c — SIMT, tiles, and occupancy](Lecture7c.md)** — warps, divergence, the
  tile of `C` each block owns, and the residency budget *(Depths 4–5)*.

Together they answer **how a GPU keeps thousands of ALUs busy**.
[Lecture 8](Lecture8.md) asks the other half — **what limits performance** once
the ALUs are busy.

Every measurement on this page was produced by a cell you can re-run on your own
GPU. If you do not have one, open the same page as a notebook on a free Colab T4:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ankush-Chander/DS635-ml-system-engineering/blob/main/docs/lectures/Lecture7a.ipynb)

On Colab, pick **Runtime → Change runtime type → T4 GPU** first, then run the
cell below once. It installs only what is missing, so it is a no-op on a machine
that already has everything.

```python
# Setup. Safe to re-run: nothing is installed if it is already importable.
import importlib.util, subprocess, sys

def ensure(module, spec=None):
    """Install `spec` only if `module` cannot already be imported."""
    if importlib.util.find_spec(module) is not None:
        return
    for candidate in ([spec, module] if spec and spec != module else [module]):
        print(f"installing {candidate} ...")
        if subprocess.run([sys.executable, "-m", "pip", "install", "-q", candidate]).returncode == 0:
            return
    raise RuntimeError(f"could not install {module}")

def triton_spec():
    """The exact Triton this torch build was compiled against.

    Torch declares Triton as a dependency, so we can read the pin instead of
    guessing. A mismatched Triton imports fine and then fails at the first
    `@triton.jit` -- much harder to diagnose than a missing package.
    """
    try:
        from importlib.metadata import requires
        for req in requires("torch") or []:
            dep = req.split(";")[0].strip()     # drop the environment marker
            if "triton" in dep:                 # `triton` on CUDA, `triton-rocm` on ROCm
                return dep
    except Exception:
        pass
    return "triton"

ensure("torch")                  # preinstalled on Colab; listed so the set is complete
ensure("matplotlib")
ensure("triton", triton_spec())

import torch, triton
print(f"torch {torch.__version__} | triton {triton.__version__} | cuda {torch.cuda.is_available()}")
```

---

## The bird's-eye view

One line of Python triggers a relay race across five layers, each speaking a
different language:

```text
 YOUR CODE           C = A @ B                             Python
     │
     ▼
 FRAMEWORK           dispatcher asks: "CUDA + fp32 →        C++
 (PyTorch / ATen)     which implementation?" → picks a
                      GEMM routine
     │
     ▼
 GPU LIBRARIES       cuBLAS / rocBLAS picks a PRE-BUILT     CUDA / HIP
 (cuBLAS, rocBLAS)    kernel tuned for this shape,
                      dtype and chip
     │
     ▼
 DRIVER + RUNTIME    "launch kernel X with 512 blocks"      command queues,
 (CUDA / ROCm)        queued, doorbell rung, CPU moves on   MMIO, DMA
     │
     ▼
 SILICON             thousands of threads, in warps of      SASS / GCN ISA
 (the GPU)            32, each compute a tile of C          (GPU machine code)
```

Two ideas are worth having in hand before the descent begins, because everything
below is a consequence of one or the other.

> **Intuition 1 — the CPU never multiplies anything.** It is a *manager*: it
> fills out a work order and drops it in the GPU's queue. Your Python line
> returns **before the math happens**.

> **Intuition 2 — the GPU wins by being wide and by hiding waiting**, not by
> being fast per thread. While one group of threads waits on memory, the
> hardware instantly swaps in another. Matmul is the ideal workload because
> every value loaded gets reused many times.

---

## The descent

```text
"how C = A @ B reaches the GPU and gets executed"
├── ✗ what matmul computes (O(n³) work on O(n²) data)     — assumed
├── ✗ latency machines vs throughput machines             — assumed
├── 1. the software stack: dispatcher → BLAS → launch       ◀ this part (7a)
├── 2. what a kernel really is (the SPMD inversion)         ◀ this part (7a)
├── 3. the throughput machine: residency & the free switch  → Lecture 7b
├── 4. SIMT: grids, blocks, warps of 32, divergence         → Lecture 7c
├── 5. one block = one tile of C                            → Lecture 7c
├── 6. coalescing & broadcast                               → Lecture 8
├── 7. the memory hierarchy and the FLOPs/byte wall         → Lecture 8
└── 8. tensor cores & torch.compile                         → Lecture 8
```

Depths 1–5 are the execution model: how work is created, named, scheduled and
kept flowing — this part covers Depths 1–2, [Lecture 7b](Lecture7b.md) is
Depth 3, and [Lecture 7c](Lecture7c.md) is Depths 4–5. Depths 6–8 are about
*feeding* that machine, and they are [Lecture 8](Lecture8.md).

---

## Depth 1 — The first ten microseconds

Everything in this lecture is a consequence of one line of PyTorch:

```text
C = A @ B
```

You did not say how many threads to make, how to split `C`, or which arithmetic
lane should compute which element. Something made all of those decisions for you
— and it made them **on the CPU, before any math happened**:

```text
C = A @ B                                          Python
  │
  ▼
ATen DISPATCHER                                    ~1–2 µs
  reads the "dispatch keys" stamped on the tensors:
  {CUDA, float32, autograd needed?}
  a routing table: (op="matmul") × (keys) → one
  concrete function. here: matmul_cuda.
  same Python, CPU tensors → different code path.
  │
  ▼
BLAS HEURISTICS                                    ~1 µs
  the vendor ships hundreds of PRE-COMPILED matmul
  kernels, each tuned for (shape, dtype, chip).
  a lookup picks one.
  nothing is compiled at runtime.
  │
  ▼
launch(kernel, grid=(16,32), block=128, args…)
  work order written into a command queue,
  doorbell register rung (an MMIO write), CPU returns.
```

### Proof by stopwatch

The gap between "the line returned" and "the work happened" is not a detail —
it is why a naive `time.time()` around GPU code lies to you, and why every
benchmark in this lecture calls `torch.cuda.synchronize()`.

```python
import time
import torch, triton, triton.language as tl
from torch.profiler import profile, ProfilerActivity

assert torch.cuda.is_available(), "these cells need an NVIDIA or AMD GPU (or a free Colab T4)"
dev  = torch.cuda.get_device_properties(0)
WARP = getattr(dev, "warp_size", 32)
print(f"{dev.name} | {dev.multi_processor_count} units | warp {WARP} | torch {torch.__version__}")

Ah = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
Bh = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
torch.matmul(Ah, Bh); torch.cuda.synchronize()      # warm up, and drain the queue

t0 = time.perf_counter()
Ch = torch.matmul(Ah, Bh)     # dispatcher + BLAS pick + doorbell. that is ALL.
t1 = time.perf_counter()
torch.cuda.synchronize()      # the CPU actually waits for the GPU here
t2 = time.perf_counter()

flop = 2 * 4096**3
print(f"\npython call returned in : {(t1-t0)*1e6:8.0f} us   <- just the paperwork")
print(f"math actually done after: {(t2-t0)*1e6:8.0f} us   <- {flop/1e9:.0f} billion FLOPs")
print(f"                    gap : {(t2-t0)/(t1-t0):8.0f}x")
```

On the course laptop:

```text
python call returned in :      159 us   <- just the paperwork
math actually done after:     8054 us   <- 137 billion FLOPs
                    gap :       51x
```

The Python call returned in 159 microseconds having computed nothing. The 137
billion floating-point operations took fifty times longer, and happened entirely
after your line of code was done. (The first number is host-side bookkeeping, so
it is noisy — re-run the cell and it moves between roughly 70 and 200 µs. The
second barely moves at all. That asymmetry is itself the point.)

**This is what "asynchronous" means in practice**, and it is the first thing the
descent has to establish: the CPU is a manager filling out a work order.

### What did it actually order?

Ask the profiler which kernel the doorbell announced:

```python
def gpu_kernel_name(fn, warmup=3):
    """Name of the GPU kernel that `fn` spends the most DEVICE time in.

    Returns None when the profiler reports no device-side events. That is not a
    bug in your code -- kernel-level tracing is a vendor feature, and on some
    driver/runtime combinations the GPU records come back with timestamps the
    profiler discards. The math still runs; we just cannot read the name here.
    """
    for _ in range(warmup):
        fn()                  # warm up: the first call also selects and loads a kernel
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()

    # Keep only events that really executed ON the GPU. The profiler also records
    # host-side wrappers such as `aten::matmul`, which dispatch GPU work but
    # execute none of it themselves -- filtering on device time is what makes
    # "the largest event" the actual GEMM kernel rather than its wrapper.
    on_device = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    return max(on_device, key=lambda e: e.self_device_time_total).key if on_device else None

A = torch.randn(2048, 2048, device="cuda")
B = torch.randn(2048, 2048, device="cuda")
MATMUL_SHAPE = (A.shape[0], B.shape[1])            # remember it: we decode this later

# Abbreviated, but every field in it is verbatim from a real run on the course
# laptop. Used only as a stand-in so the rest of the lecture still runs on a
# machine whose profiler cannot report kernel names.
COURSE_LAPTOP_KERNEL = "Cijk_Ailk_Bljk_SB_MT128x64x8_SN_1LDSB0_..._TT8_8_..._WS32_WG16_8_1_WGM4"

MATMUL_KERNEL = gpu_kernel_name(lambda: A @ B)
if MATMUL_KERNEL is None:
    MATMUL_KERNEL = COURSE_LAPTOP_KERNEL
    print("this machine's profiler reported no device-side events;")
    print("falling back to the name recorded on the course laptop.\n")

print("the kernel your `@` actually launched:\n", MATMUL_KERNEL)
```

Where the profiler can see the GPU, that prints a single kernel whose name
begins:

```text
Cijk_Ailk_Bljk_SB_MT128x64x8_SN_1LDSB0_..._TT8_8_..._WS32_WG16_8_1_WGM4
```

(If your machine takes the fallback branch, everything downstream still works —
you are reading a decomposition that was really measured, just not on your own
card. A free Colab T4 will show you your own.)

It looks like line noise. It is not. That string is the entire decomposition of
your matmul — how `C` was cut up, how many threads were assigned to each piece,
how wide the hardware's thread groups are, and how much of `C` each individual
thread computed. Every field in it is vocabulary from this lecture.

> **By the end of Depth 5 (in [Lecture 7c](Lecture7c.md)) you will read that
> name and be able to state exactly how your `A @ B` was executed.**

### What a *kernel* is

Nothing to do with an operating-system kernel. Here a **kernel is software**: one
compiled function that runs *on the GPU*, executed by many threads at once.

```text
software                                    hardware
────────                                    ────────
C = A @ B            ← the line you wrote
     ↓
PyTorch dispatcher
     ↓
rocBLAS / cuBLAS     ← a library of pre-built kernels
     ↓
one chosen kernel ──────────────► warps ──► CUs / SMs ──► ALU lanes
```

Keep three things separate:

| | |
| --- | --- |
| `A @ B` | the operation you asked for |
| the kernel | GPU **software** that implements it |
| CU/SM, warp, ALU lane | GPU **hardware** that executes that software |

PyTorch did not write this kernel when you called `@`. Vendor BLAS libraries ship
**many** pre-compiled variants of the same operation — different tile shapes,
dtypes, transpositions, architectures — and select one per call from your shapes
and your device. That is why the name is so specific: it names *one variant*, not
"matmul". Being compiled for one architecture, it will not run on another.

Two consequences this lecture uses:

* the choice depends on your problem, so a different matmul gets a different kernel — shown in [Depth 5](Lecture7c.md#depth-5-one-block-one-tile-of-c)
* a kernel is just a program, so you can write your own — which is what Triton is for. Every `@triton.jit` function below compiles to a kernel of exactly this kind, and runs beside the vendor's.

---

## Depth 2 — The kernel inversion

We now know a kernel is a program the GPU runs. But *what does that program say*?

The tempting answer — "it is the recipe for breaking up the task" — is subtly
wrong, and getting it wrong makes every kernel you read afterwards confusing.
The launch configuration (the grid) does the breaking up. The kernel is the
opposite:

> **The SPMD inversion.** A kernel is an ordinary *sequential* function, written
> from the point of view of **one worker**. The parallelism comes from launching
> a million copies of it. You never write "split the work" — you write "here is
> what worker *(x, y)* does", and each worker computes its own identity to find
> its slice.

Here is the whole idea in a naive CUDA matmul. Read it looking for the loop over
the matrix, or for anything that says "in parallel". Neither is there:

```text
// One thread's job. No visible parallelism anywhere.
__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // "who am I?"
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int k = 0; k < N; k++)
        sum += A[row*N + k] * B[k*N + col];            // one dot product
    C[row*N + col] = sum;                              // one output element
}
```

Those first two lines are the signature move of GPU programming: **the program
asks the hardware who it is**, and derives its work from the answer. Every kernel
in this lecture starts the same way.

### A detour: what Triton is, and why we use it here

To watch the hierarchy we need to *write* a kernel, not just call one. The
traditional way is CUDA C++, which only compiles for NVIDIA. **Triton** is a
Python-embedded language for writing GPU kernels that compiles for both NVIDIA and
AMD, so the same cell runs on a Colab T4 and on the course laptop. That is the only
reason it appears in this lecture — we are not learning Triton, we are using it as
a window onto the execution model.

Three things to know before reading the code:

* `@triton.jit` marks a function as a kernel. It is not ordinary Python: it is
  compiled to GPU machine code the first time you launch it.
* You launch it with `kernel[grid](args...)`. The `[grid]` says **how many
  independent instances to run**.
* Triton is **block-level, not thread-level**. In CUDA you write what *one thread*
  does and the hardware runs thousands of copies. In Triton you write what *one
  chunk of data* has done to it, using whole-array expressions, and the compiler
  decides how many hardware threads to use. This difference is the one thing worth
  keeping straight, and the code below is annotated to make it visible.

One vocabulary note, because it trips people up: Triton calls each instance a
**program**, not a thread block. They correspond closely enough that we will read
`tl.program_id` as "which block am I", but they are not the same abstraction, and
in a moment we will see exactly where they come apart.

### The hierarchy in one runnable kernel

It adds two vectors — the simplest possible thing — but every level is visible as
an actual line of code:

```python
@triton.jit
def vector_add(a_ptr, b_ptr, c_ptr, n_elements, ELEMS_PER_PROG: tl.constexpr):
    """One execution of this body = ONE PROGRAM INSTANCE (≈ one CUDA block).

    `grid` of them exist. They are independent and unordered: no instance may
    assume another has run, is running, or will run. That independence is why
    the same source scales unchanged from an 18-unit laptop to a 132-SM server part.

    Everything below is a VECTOR of width ELEMS_PER_PROG. Triton is a
    *block-level* language -- there is no "my thread's element" in this code.
    Nothing here names a thread, a warp, or a lane; the compiler chooses those.
    """
    prog_id = tl.program_id(0)                 # instance id: 0 .. grid-1

    # ELEMENT slots inside my chunk -- an index vector, NOT hardware lanes.
    # Hardware lanes number num_warps*32, which is a different count entirely
    # (the printout at the bottom shows the ratio).
    elem_in_prog = tl.arange(0, ELEMS_PER_PROG)

    # The global positions in a/b/c that this instance owns:
    #   prog 0 -> [0..255], prog 1 -> [256..511], ...
    elem_idx = prog_id * ELEMS_PER_PROG + elem_in_prog

    # n_elements is rarely a multiple of ELEMS_PER_PROG, so the LAST instance
    # runs off the end. Masked-off positions are neither loaded nor stored, so
    # we never touch memory we do not own.
    in_bounds = elem_idx < n_elements

    # Whole-vector ops, not scalar ones: `a_ptr + elem_idx` is ELEMS_PER_PROG
    # addresses at once, so one tl.load fetches that many values in one go.
    a_vals = tl.load(a_ptr + elem_idx, mask=in_bounds)
    b_vals = tl.load(b_ptr + elem_idx, mask=in_bounds)
    tl.store(c_ptr + elem_idx, a_vals + b_vals, mask=in_bounds)


n_elements     = 1_000_000
ELEMS_PER_PROG = 256               # ELEMENTS per instance -- *not* a thread count

a = torch.randn(n_elements, device="cuda")
b = torch.randn(n_elements, device="cuda")
c = torch.empty_like(a)

# THE GRID: how many instances to cover n_elements, ELEMS_PER_PROG at a time?
# cdiv rounds UP; `in_bounds` cleans up the overhang in the final instance.
grid = (triton.cdiv(n_elements, ELEMS_PER_PROG),)
compiled = vector_add[grid](a, b, c, n_elements, ELEMS_PER_PROG=ELEMS_PER_PROG)
assert torch.allclose(c, a + b)

# ---- Level 1: what YOU specified (the programming model) --------------------
n_progs   = grid[0]
slots     = n_progs * ELEMS_PER_PROG
print("PROGRAMMING MODEL  (every number below came from your source)")
print(f"  program instances : {n_progs:,}")
print(f"  elements/instance : {ELEMS_PER_PROG}")
print(f"  slots covered     : {slots:,} for {n_elements:,} elements"
    f"   ({slots - n_elements} masked off in the last instance)")

# ---- Level 2: what the COMPILER built (the hardware mapping) ----------------
# None of these appear anywhere in the kernel above.
warps_per_prog = compiled.metadata.num_warps
lanes_per_prog = warps_per_prog * WARP
print("\nHARDWARE MAPPING   (none of this appears in your source)")
print(f"  warps/instance    : {warps_per_prog}   <- Triton chose this, you did not")
print(f"  lanes/instance    : {warps_per_prog} x {WARP} = {lanes_per_prog}")
print(f"  elements/LANE     : {ELEMS_PER_PROG} / {lanes_per_prog}"
    f" = {ELEMS_PER_PROG // lanes_per_prog}"
    f"   <- >1 means `elem_in_prog` indexes ELEMENTS, not lanes")
```

On the course laptop that prints a grid of 3,907 instances covering 1,000,192
slots — 192 of them masked off — and then the part that matters:

```text
num_warps      : 4   <- Triton chose this, you did not
hardware lanes : 4 x 32 = 128 per instance
elements/lane  : 256 / 128 = 2
```

**`BLOCK` is 256 but there are only 128 hardware lanes.** Each lane handles two
elements. A tempting shortcut — "256 elements ÷ 32 per warp = 8 warps" — is simply
wrong here, and the compiler will happily tell you so. `BLOCK` is a *logical* tile
size you chose; `num_warps` is the *hardware* mapping the compiler chose. Keep them
apart.

So the levels are:

| Level | In the code | What it is | Who chose it |
| ----- | ----------- | ---------- | ------------ |
| **grid** | `grid = (cdiv(n, BLOCK),)` | all the instances needed to cover the problem | you |
| **program** ≈ block / workgroup | `tl.program_id(0)` | one chunk of work, scheduled onto one CU | you (via `BLOCK`) |
| **element / lane position** | one entry of `tl.arange(0, BLOCK)` | one logical position in that chunk | you (via `BLOCK`) |
| **warp** / wavefront | *nowhere* — `num_warps` | what the hardware **actually** schedules | the compiler |
| **thread** | *nowhere* | one hardware lane, may handle several elements | the compiler |

Note the asymmetry. You wrote the grid, the block and the per-element work. **You
never wrote the warp, and you never wrote a thread.** Nothing in that kernel
mentions 32. Those two rows are filled in underneath you — which is why they are
the levels that surprise people, and why the rest of this lecture keeps returning
to them.

!!! question "💬 If `tl.arange(0, 256)` does not create 256 threads, what does the number 256 actually control?"

    ??? hint "Answer"
        **How much data one program instance owns**, and nothing more directly. It sets
        the tile size, which in turn sets the grid (`n / BLOCK` instances) and how much
        work each hardware lane ends up with (`BLOCK / (num_warps × 32)` elements).
        Triton then picks `num_warps` — 4 here — to map that tile onto the machine. This
        is why the warp-width experiment in Depth 4 ([Lecture 7c](Lecture7c.md)) pins `num_warps=1`: with
        one warp, `BLOCK` and the lane count finally coincide, and only then does
        sweeping `BLOCK` measure something about warps.

Two of those rows deserve a sentence more.

A **block / workgroup** is assigned to exactly one compute unit at a time, and the
threads inside it can cooperate through shared memory. That is what makes it a
meaningful unit rather than an arbitrary grouping.

A **warp / wavefront** is a fixed-size group of threads that execute instructions
together. NVIDIA uses 32-thread warps; AMD hardware uses 32- or 64-thread
wavefronts depending on the architecture. The size is an architectural choice, not
a universal constant — this laptop contains two AMD GPUs with different widths, as
[Appendix B](Lecture7c.md#appendix-b-this-laptops-gpus) shows.

> **The hardware schedules groups of threads, not independent CPU-like threads.**

---

## Next

You launched a kernel, and the compiler quietly chose warps and blocks you never
wrote. **[Lecture 7b](Lecture7b.md)** meets the machine those warps land on — and
measures the latency hiding it exists to perform.

References for all three parts are collected at the end of
[Lecture 7c](Lecture7c.md#references-and-further-reading).
