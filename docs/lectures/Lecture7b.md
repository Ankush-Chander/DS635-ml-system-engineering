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

# Matmul to silicon 7b: the throughput machine

> **Module thesis:** a GPU is not built to make one thread fast — it is built to
> never have an idle ALU. This part is where that thesis gets measured.

In [Lecture 7a](Lecture7a.md) we watched `C = A @ B` become a kernel launch: the
CPU filled out a work order, rang a doorbell and moved on, and the compiler — not
you — chose the warps and blocks. This part asks what that work is launched
*onto*, and why the machine is shaped so strangely: the hardware hierarchy, a
register file larger than the cache beside it, and an experiment that measures
latency hiding directly *(Depth 3 of the descent; [7c](Lecture7c.md) continues
with Depths 4–5)*.

Every measurement on this page was produced by a cell you can re-run on your own
GPU. If you do not have one, open the same page as a notebook on a free Colab T4:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ankush-Chander/DS635-ml-system-engineering/blob/main/docs/lectures/Lecture7b.ipynb)

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

```python
# Re-establish the handles every cell below uses. Lecture 7a builds these up
# step by step; here they are just plumbing.
import torch, triton, triton.language as tl

assert torch.cuda.is_available(), "these cells need an NVIDIA or AMD GPU (or a free Colab T4)"
dev  = torch.cuda.get_device_properties(0)
WARP = getattr(dev, "warp_size", 32)
print(f"{dev.name} | {dev.multi_processor_count} units | warp {WARP} | torch {torch.__version__}")
```

---

## Depth 3 — The throughput machine

We know what gets launched. Now: what is it launched *onto*, and why is that
machine shaped so strangely?

### CPU versus GPU

![CPU die budget versus GPU die budget: control logic and cache dominate the CPU, arithmetic units dominate the GPU](../images/ai_acceleration/cpu_vs_gpu.png)

A CPU spends a large fraction of its transistor budget making **one or a few instruction streams progress quickly**:

* out-of-order execution
* branch prediction
* speculation
* large caches
* sophisticated control logic

A GPU makes a different trade:

> **Spend silicon on arithmetic lanes and keep many independent pieces of work in flight.**

The two machines are solving the same problem—execute instructions—but they use very different strategies for dealing with latency. The CPU makes *one* thread fast. The GPU keeps *many* threads in flight and hides latency with parallelism.

That figure is drawn in the abstract: *SM 0 … SM M-1*, *32 lanes*, *multiple
warps per SM*. The rest of this section puts a number from one real card into
every one of those boxes, so keep the figure in view while you read it.

### The GPU in this laptop

Zooming in on the discrete GPU — the same boxes the figure draws, with this
laptop's numbers in them:

```text
GPU
│
├── VRAM  10.7 GB                     <- "HBM / GDDR Memory (GPU Global Memory)"
│
└── 36 × Compute Unit                 <- "SM 0 ... SM M-1", so M = 36
       │
       ├── 2 × SIMD                   <- "Execution Units (ALU Lanes)"
       │     ├── 32 ALU lanes         <- "Lane 0 ... Lane 31", one warp wide
       │     ├── register file        <- "Register File (per SM)"
       │     └── scheduler            <- "Warp Scheduler (Instruction Issue)"
       │
       ├── L1                         ┐  "L1 Cache / Shared Memory (per SM)"
       └── shared memory / LDS        ┘  -- one box in the figure, two jobs
```

None of those numbers are invented. Every one of them comes out of a vendor
tool:

```shell
rocminfo                 # AMD
nvidia-smi -q            # NVIDIA
```

On this machine:

```text
Marketing Name:          AMD Radeon RX 6700M
Compute Unit:             36
SIMDs per CU:             2
Wavefront Size:           32
Max Waves Per CU:         32
Workgroup Max Size:       1024
L1:                       16 KB
L2:                       3 MB
L3:                       96 MB
LDS (shared memory):      64 KB
Cacheline Size:           128 B
VRAM:                     10.7 GB
```

Now read that dump and the figure side by side. Almost every line is a box, and
almost every box is a line:

| Box in the figure | What it is | This laptop's value |
| ----------------- | ---------- | ------------------- |
| **Lane 0 … Lane 31**, inside *Execution Units* | one **ALU lane**: an arithmetic unit doing multiply, add, fused multiply-add — the thing we are trying to never leave idle | `Wavefront Size: 32`, so 32 lanes step together, exactly as drawn |
| **Execution Units (ALU Lanes) SIMD** | a group of lanes executing the same instruction together, so one instruction fetch is amortized over all of them | `SIMDs per CU: 2` — the figure draws one per SM, this card has two |
| **SM 0 … SM M-1** (AMD: **Compute Unit**) | the repeating execution block — lanes, registers, scheduler, cache, shared memory. The unit a block is scheduled onto | `Compute Unit: 36`, so M = 36 |
| **Warps (multiple per SM)** | the work sitting resident on the SM, for the scheduler to choose between | `Max Waves Per CU: 32` — the figure draws 4 to stay legible; the real ceiling is 32 |
| **Warp Scheduler (Instruction Issue)** | picks a ready warp each cycle and issues one instruction for it — this is where latency hiding physically happens | no line in the dump; we measure its effect two sections below |
| **Register File (per SM)** | fast per-thread storage holding the live state of every resident thread; unusually large, so many threads can stay resident at once | not reported by `rocminfo` either — occupancy is what exposes its size |
| **L1 Cache / Shared Memory (per SM)** (AMD: **LDS**) | one box, two jobs: an automatic cache, plus a small explicitly-managed scratchpad that lets threads in a block cooperate without going to DRAM | `L1: 16 KB` and `LDS: 64 KB` |
| **Load/Store Units** | the path from the lanes to memory; it moves whole cache lines, never single floats | `Cacheline Size: 128 B` |
| **L2 Cache (GPU-wide)** | the last stop shared by every SM | `L2: 3 MB` |
| **HBM / GDDR Memory** | the GPU's own DRAM — high bandwidth, but still hundreds of cycles away | `VRAM: 10.7 GB` |
| *no box* | RDNA puts a large Infinity Cache between L2 and VRAM. Most GPUs have no equivalent, so the figure has none — but it is why some measurements later in this module bend where you would not expect | `L3: 96 MB` |

One line resists the exercise entirely. `Workgroup Max Size: 1024` constrains the
*programming model* — the largest block you are allowed to launch — rather than
any piece of silicon, so no box can hold it.

The important question is not yet what every box is called.

It is:

> **Why does the GPU need all of these boxes?**

The answer emerges as we follow one piece of work down to the hardware. For now
we need just enough vocabulary to describe execution — the cache hierarchy can
wait for [Lecture 8](Lecture8.md).

Most literature uses NVIDIA's vocabulary; `rocminfo` reports AMD's. This module uses **warp**, **SM**, and **shared memory** for the general programming model, and points out AMD terminology where it matters. The full correspondence is in [Appendix A](Lecture7c.md#appendix-a-nvidia-and-amd-terminology).

### Read your own GPU

Those tools are vendor-specific. PyTorch will tell you the same things on any of
them, which is what the cells below use:

```python
print(f"name          : {dev.name}")
print(f"units         : {dev.multi_processor_count}")
print(f"warp size     : {WARP}")
print(f"VRAM          : {dev.total_memory / 2**30:.1f} GB")
print(f"backend       : {'ROCm/HIP' if torch.version.hip else 'CUDA'}")
```

On the course laptop this prints **18 units** — and the card has 36 compute units.

That is not a bug and it is worth pausing on. On RDNA architectures two CUs are
fused into a **WGP (work-group processor)**, and that is what the driver reports.
Take the number at face value and every derived figure below is wrong by exactly
2×. Whenever a hardware count feeds a performance calculation, check what unit the
tool is counting in before you multiply.

### Think of the chip as nested containers

Read from the inside out:

```text
lane         = one FP32 ALU. Does one FMA per cycle. Nothing smaller matters.
                  ┌─┐
                  └─┘

SIMD         = 32 lanes bolted together. ONE instruction fetch drives all 32.
                 This is why a warp is 32 threads: the warp is the software
                 shadow of this piece of silicon.
                  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
                  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
             + a register file (128 KB) + a scheduler that picks which warp goes next

CU (=SM)     = 2 SIMDs + 16 KB L1 + 64 KB LDS. The unit a block is pinned to.
                  ┌───────────────────────────────┐
                  │ SIMD 0 [32 lanes] [128KB reg] │
                  │ SIMD 1 [32 lanes] [128KB reg] │
                  │ L1 16KB     LDS 64KB          │
                  └───────────────────────────────┘

GPU          = 36 CUs + 3 MB L2 + 96 MB L3 (shared by all 36)
```

Two counts live on every level — how many threads execute per cycle, and how
many can be parked waiting their turn:

```text
┌───────┬───────────────────────────┬────────────────────────────────┐
│ Level │    Executing per cycle    │        Resident (parked)       │
├───────┼───────────────────────────┼────────────────────────────────┤
│ SIMD  │ 32 lanes = 1 warp's instr │ 16 warps  (per CU: 32 ÷ 2)     │
├───────┼───────────────────────────┼────────────────────────────────┤
│ CU    │ 2 SIMD × 32 = 64 threads  │ 32 warps × 32 = 1,024 threads  │
├───────┼───────────────────────────┼────────────────────────────────┤
│ GPU   │ 36 × 64 = 2,304 threads   │ 36 × 32 = 1,152 warps          │
└───────┴───────────────────────────┴────────────────────────────────┘
```

Each CU has 1,024 threads resident and only 64 executing. The scheduler picks one
warp per cycle from the 16 sitting in each SIMD's register file. **That 16× ratio
is the whole architecture.**

The same two counts on the other machines this course runs on. First a free
Colab T4 — the per-SM structure is from NVIDIA's Turing whitepaper (compute
capability 7.5), and the counts are what `torch.cuda.get_device_properties`
prints if you run the cell above on Colab: 40 units, 1,024 threads per unit.
NVIDIA replaces the two 32-lane SIMDs with four 16-lane partitions per SM, each
issuing one warp instruction over two cycles:

```text
┌───────┬───────────────────────────┬────────────────────────────────┐
│ Level │    Executing per cycle    │        Resident (parked)       │
├───────┼───────────────────────────┼────────────────────────────────┤
│ Part. │ 16 lanes = ½ warp's instr │ 8 warps  (per SM: 32 ÷ 4)      │
├───────┼───────────────────────────┼────────────────────────────────┤
│ SM    │ 4 part × 16 = 64 threads  │ 32 warps × 32 = 1,024 threads  │
├───────┼───────────────────────────┼────────────────────────────────┤
│ GPU   │ 40 × 64 = 2,560 threads   │ 40 × 32 = 1,280 warps          │
└───────┴───────────────────────────┴────────────────────────────────┘
```

And the course server `watchtower` (RTX 6000 Ada; driver-reported: 142 SMs,
1,536 resident threads per SM — four 32-lane partitions per SM):

```text
┌───────┬───────────────────────────┬────────────────────────────────┐
│ Level │    Executing per cycle    │        Resident (parked)       │
├───────┼───────────────────────────┼────────────────────────────────┤
│ Part. │ 32 lanes = 1 warp's instr │ 12 warps  (per SM: 48 ÷ 4)     │
├───────┼───────────────────────────┼────────────────────────────────┤
│ SM    │ 4 part × 32 = 128 threads │ 48 warps × 32 = 1,536 threads  │
├───────┼───────────────────────────┼────────────────────────────────┤
│ GPU   │ 142 × 128 = 18,176 threads│ 142 × 48 = 6,816 warps         │
└───────┴───────────────────────────┴────────────────────────────────┘
```

Or stop reading tables and print your own. The cell below rebuilds the same
table for whatever card it runs on — every number is the driver's except the
lanes-per-partition lookup, which no driver exposes. (It even has an Apple
silicon branch, where Metal reports nothing at all and the entire shape rests
on community reverse engineering — the printout says so when that is the case.)

```python
# Print the executing-vs-resident table for the GPU you are on. Every number
# is driver-reported where the driver reports it; the facts no driver exposes
# (lanes per partition -- and on Apple, everything) come from the whitepapers
# and community measurements noted inline.

def gpu_counts():
    mps = getattr(torch.backends, "mps", None)
    if not torch.cuda.is_available() and mps and mps.is_available():
        # Apple silicon. Metal reports none of this, so the whole shape is
        # community reverse engineering (Asahi Linux, metal-benchmarks):
        # core = 4 schedulers x 32-lane SIMDs = 128 FP32 ALUs, up to
        # 96 SIMD-groups x 32 = 3,072 threads resident per core (M1/M2 era;
        # M3+ dynamic caching makes residency elastic).
        import re, subprocess
        sp = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                            capture_output=True, text=True).stdout
        name = re.search(r"Chipset Model:\s*(.+)", sp)
        cores = re.search(r"Total Number of Cores:\s*(\d+)", sp)
        assert cores, "could not read the GPU core count from system_profiler"
        return (name.group(1).strip() if name else "Apple GPU", int(cores.group(1)),
                3072, 32, "core", "sched", 4, 32,
                "Apple figures are reverse-engineered (M1/M2), not driver-reported")
    d = torch.cuda.get_device_properties(0)
    warp = getattr(d, "warp_size", 32)
    if torch.version.hip:
        # RDNA reports WGPs (2 fused CUs): 4 SIMDs x 32 lanes, wave32.
        # GCN/CDNA report CUs: 4 SIMDs x 16 lanes, wave64.
        unit, part = ("WGP", "SIMD") if warp == 32 else ("CU", "SIMD")
        parts, lanes = 4, (32 if warp == 32 else 16)
    else:
        # NVIDIA, by compute capability: (scheduler partitions per SM,
        # FP32 cores per SM). Pascal GP100 is the odd one out: 2 x 32.
        # Volta/Turing/A100: 4 x 16. Consumer Ampere onward: 4 x 32.
        shapes = {(6, 0): (2, 64), (6, 1): (4, 128),
                  (7, 0): (4, 64), (7, 5): (4, 64), (8, 0): (4, 64),
                  (8, 6): (4, 128), (8, 7): (4, 128), (8, 9): (4, 128),
                  (9, 0): (4, 128), (10, 0): (4, 128), (12, 0): (4, 128)}
        unit, part = "SM", "part."
        if (d.major, d.minor) not in shapes:
            print(f"note: cc {d.major}.{d.minor} not in the lookup -- "
                  f"assuming 4 x 32; check your card's whitepaper")
        parts, cores = shapes.get((d.major, d.minor), (4, 128))
        lanes = cores // parts
    return (d.name, d.multi_processor_count, d.max_threads_per_multi_processor,
            warp, unit, part, parts, lanes, None)

name, units, res_thr, warp, unit, part, parts, lanes, caveat = gpu_counts()
res_w = res_thr // warp                            # resident warps per unit

frac = {warp: "1", warp // 2: "½", warp // 4: "¼"}.get(lanes, f"{lanes}/{warp}")
rows = [
    ("Level", "Executing per cycle", "Resident (parked)"),
    (part, f"{lanes} lanes = {frac} warp's instr",
           f"{res_w // parts} warps  (per {unit}: {res_w} ÷ {parts})"),
    (unit, f"{parts} × {lanes} = {parts * lanes} threads",
           f"{res_w} warps × {warp} = {res_thr:,} threads"),
    ("GPU", f"{units} × {parts * lanes} = {units * parts * lanes:,} threads",
           f"{units} × {res_w} = {units * res_w:,} warps"),
]

w = [max(len(r[c]) for r in rows) for c in range(3)]
def rule(l, m, r):
    print(l + m.join("─" * (w[c] + 2) for c in range(3)) + r)
rule("┌", "┬", "┐")
for i, r in enumerate(rows):
    pad = str.center if i == 0 else str.ljust
    print("│ " + " │ ".join(pad(r[c], w[c]) for c in range(3)) + " │")
    rule("├", "┼", "┤") if i < len(rows) - 1 else rule("└", "┴", "┘")

print(f"\n{name}: {units * res_thr:,} resident / {units * parts * lanes:,} executing"
      f" = {units * res_thr / (units * parts * lanes):.0f}× over-subscribed")
if caveat:
    print(f"({caveat})")
```

On this laptop it prints the **WGP view** — `SIMD / WGP / GPU` with
`18 × 128 = 2,304` — because 18 WGPs is what the driver reports (the
[Read your own GPU](#read-your-own-gpu) correction again). Same GPU row, same
16×; only the middle level is drawn twice as wide.

Three machines spanning a laptop, a free cloud card and a datacentre part — an
8× spread in raw ALU count — and every one keeps **12–16× more threads resident
than it can execute**. The ratio is the design, not an accident of one chip;
what the bigger card buys is *width*, not more hiding per ALU.

### Arithmetic capacity

The left column of the table is the GPU's raw execution bandwidth. At 2.3 GHz,
with 2 FLOPs per fused multiply-add:

```text
2,304 ALUs × 2 FLOPs/FMA × 2.3 GHz ≈ 10.6 TFLOP/s
```

That is the theoretical FP32 compute roof. We will need it again in Lecture 8.

### Over-subscription is a feature

Now the right column — and the number that explains the whole architecture. Ask
the driver how much work it can hold resident, and compare with the left:

```python
resident_per_unit = dev.max_threads_per_multi_processor
units             = dev.multi_processor_count
resident          = resident_per_unit * units
alus              = 2304        # 36 CU x 2 SIMD x 32 lanes, from the box above

print(f"resident threads / unit : {resident_per_unit:,}")
print(f"units reported          : {units}")
print(f"resident threads total  : {resident:,}   ({resident // WARP:,} warps)")
print(f"FP32 ALUs               : {alus:,}")
print(f"over-subscription       : {resident / alus:.0f}x more work loaded than executable")

# What a big matmul actually asks for, at one thread per output element:
M = N = 4096
threads = M * N
print(f"\na {M}x{N} matmul, 1 thread per output : {threads:,} threads")
print(f"resident at once                      : {resident:,}")
print(f"                                      -> ~{threads/resident:.0f} waves through the GPU")
```

On the course laptop:

```text
resident threads / unit : 2,048
units reported          : 18
resident threads total  : 36,864   (1,152 warps)
FP32 ALUs               : 2,304
over-subscription       : 16x more work loaded than executable

a 4096x4096 matmul, 1 thread per output : 16,777,216 threads
resident at once                        : 36,864
                                      -> ~455 waves through the GPU
```

Two things fall out of that. First, the WGP correction holds together: 18 reported
units × 2,048 = 36,864, exactly the 36 CU × 1,024 the AMD tooling reports.

Second, and this is the design decision the whole lecture turns on:

> **1,024 threads can be resident on a CU, while 64 FP32 ALUs execute them.**

The GPU deliberately keeps **16× more work loaded than it can execute at any
instant**, and the 4096² matmul is 455× larger still — it flows through in waves,
dispatched by a hardware work distributor as blocks retire.

Why build the machine that way?

### The GPU does not wait — it runs somebody else

Consider a thread executing:

```text
load A[i]
compute
compute
compute
```

A floating-point instruction can be issued very quickly. A trip to DRAM can take
hundreds of cycles:

```text
FMA:
    █

DRAM:
    █████████████████████████████████████████████████
```

If the GPU had only one thread, its arithmetic units would spend most of their
time waiting. A CPU attacks this by making one instruction stream sophisticated
enough to keep making progress — out-of-order execution, speculation, prefetch.

A GPU uses a completely different strategy.

> **The GPU does not wait. It runs somebody else.**

Suppose warp 0 issues a memory load:

```text
warp 0
    │
    └── load from DRAM
             │
             │ waiting
             ▼
        scheduler
             │
             ├── warp 1 ──► execute
             │
             ├── warp 2 ──► execute
             │
             ├── warp 3 ──► execute
             │
             └── ...
```

When warp 0 is waiting for memory it simply becomes ineligible; when its data
arrives it becomes eligible again. The GPU converts **memory latency** into
**parallel work**. That single requirement forces most of the rest of the design:

```text
DRAM takes hundreds of cycles
          │
          ▼
don't wait
          │
          ▼
run another warp
          │
          ▼
many warps must be resident
          │
          ▼
their state must already be available
          │
          ▼
register file holds their live state
```

Read that chain in the other direction and it explains the hardware: the register
file is enormous *because* warp switching must be free, and warp switching must be
free *because* the ALUs must never idle.

### Why the switch is free

An operating-system context switch moves or reconstructs state — save registers,
change stack, possibly change address space. That is expensive.

A GPU warp switch does not need a comparable save/restore. The live register state
of every resident warp is **already sitting in the register file**, the entire
time, even while stalled. Switching warps is therefore just:

> **Choose a different set of already-resident registers for the next instruction.**

The GPU has converted a recurring context-switch cost into a one-time silicon
cost: build a register file large enough to hold the state of many resident
threads. Concretely, on this GPU:

```text
L1:
    16 KB per CU

Vector register file:
    128 KB per SIMD
    ≈ 256 KB per CU
```

The register file is **larger than the L1 cache beside it** — an inversion that
would look absurd on a CPU. It is not a cache at all. It is the seat of the
resident warps, and therefore the mechanism that makes cheap warp switching
possible.

### So how much alternative work is enough?

We now have the mechanism:

```text
warp waits for memory
        ↓
scheduler runs another warp
        ↓
enough other warps are available
        ↓
the GPU keeps working
```

But how much independent work does the GPU actually need before the waiting is effectively covered?

We can measure that directly.

#### The experiment

We need a kernel with one important property:

> **Every run must do exactly the same work. We will change only how much independent work is available to the GPU.**

A memory copy is ideal:

```text
src ──read──► GPU ──write──► dst
```

There is almost no arithmetic, so the experiment is dominated by memory traffic.

We will copy the same 64 MB in every run.

The only thing we will change is the **number of Triton programs** we launch.

#### The kernel

Here is the kernel:

```python
@triton.jit
def copy_kernel(src_ptr, dst_ptr, n_elements, num_programs, BLOCK_SIZE: tl.constexpr):
    # Which program instance am I?
    prog_id = tl.program_id(0)

    # Each program processes several chunks of the array.
    # Fewer programs simply means more chunks per program.
    for block_start in range(
        prog_id * BLOCK_SIZE,
        n_elements,
        num_programs * BLOCK_SIZE
    ):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        in_bounds = offsets < n_elements

        tl.store(
            dst_ptr + offsets,
            tl.load(src_ptr + offsets, mask=in_bounds, other=0.0),
            mask=in_bounds
        )
```

You only need to notice three things.

**1. Each program starts at a different chunk.**

```python
# prog_id = tl.program_id(0)
```

Program 0 starts at chunk 0, program 1 at chunk 1, and so on.

**2. Together, the programs still cover the entire array.**

The stride is:

```python
# num_programs * BLOCK_SIZE
```

So if we launch fewer programs, each program simply comes back around and
processes more chunks.

For example, with four programs:

```text
program 0 → chunk 0 → chunk 4 → chunk 8 → ...
program 1 → chunk 1 → chunk 5 → chunk 9 → ...
program 2 → chunk 2 → chunk 6 → chunk 10 → ...
program 3 → chunk 3 → chunk 7 → chunk 11 → ...
```

Thus:

> **Changing `num_programs` does not change the amount of data copied.**

It only changes how that work is divided among independent program instances.

**3. Each iteration is just a memory copy.**

```python
# tl.load(...)     # read from src
# tl.store(...)    # write to dst
```

There is essentially no arithmetic here.

So the experiment isolates the question we care about:

> **How does the amount of independent work affect the GPU's ability to keep the memory system busy?**

#### Make a prediction

If latency hiding is real, what should happen?

```text
few programs
     │
     ▼
little alternative work
     │
     ▼
more waiting
     │
     ▼
lower throughput


more programs
     │
     ▼
more alternative work
     │
     ▼
more opportunities to run while other work waits
     │
     ▼
higher throughput
```

But there should eventually be a point where adding more work stops helping.

Let's measure it.

#### Measuring it

The amount of work is fixed in every run:

```python
n_elements = 1 << 24                # 16,777,216 float32 elements ≈ 64 MB

src = torch.randn(n_elements, device="cuda")
dst = torch.empty_like(src)

# Every element is float32 = 4 bytes.
#
#     4 bytes READ
#   + 4 bytes WRITE
#   = 8 bytes per element
#
# The same 8 bytes/element are moved in every run.
gb_moved = 2 * n_elements * 4 / 1e9

print(
    f"copying {n_elements*4/2**20:.0f} MB, "
    f"{gb_moved:.3f} GB moved, "
    f"identical work every row"
)

print(f"{'blocks':>8} {'ms':>9} {'GB/s':>9}")

for num_programs in (
    1, 2, 4, 6, 8, 10, 12, 14, 18, 36, 144, 1152
):
    median_ms = triton.testing.do_bench(
        lambda: copy_kernel[
            (num_programs,)
        ](
            src,
            dst,
            n_elements,
            num_programs,
            BLOCK_SIZE=1024
        ),
        warmup=25,
        rep=150,
        return_mode="median"
    )

    effective_gbs = gb_moved / (median_ms * 1e-3)

    print(
        f"{num_programs:>8} "
        f"{median_ms:>9.3f} "
        f"{effective_gbs:>9.1f}"
    )

assert torch.equal(src, dst)
```

Notice what the benchmark changes:

```text
                         SAME
        ┌────────────────────────────┐
        │ data copied:     64 MB     │
        │ memory traffic:  same      │
        │ kernel:          same      │
        └────────────────────────────┘

                         ONLY THIS CHANGES

                    num_programs
                         ↓
                independent work
                         ↓
                  latency hiding
```

#### What the GPU actually does

Measured on the course laptop:

```text
  blocks        ms      GB/s
       1     5.512      24.3
       2     2.645      50.7
       4     1.399      95.9
       6     1.053     127.4
       8     0.886     151.5
      10     0.796     168.7
      12     0.768     174.7
      14     0.775     173.3
      18     0.783     171.5
      36     0.772     173.8
     144     0.794     169.0
    1152     0.802     167.4
```

Look at the two ends:

> **Same bytes, same kernel — but 7.2× the throughput.**

Nothing about the instructions became cheaper. Nothing about the amount of
data changed.

We simply gave the GPU more independent work to keep busy.

When one block is waiting on memory, there is now another block — and therefore
other warps — available to run.

> **That is latency hiding from the outside: more independent work turns waiting
> time into useful work.**

#### But more work eventually stops helping

There is a second, more interesting observation.

This GPU reports **18 units**. You might therefore expect performance to keep
increasing until 18 blocks — one block for every unit.

It doesn't.

The curve reaches its peak around **10–12 blocks**.

At 12 blocks, six of the eighteen units have not even been given a block, yet
the memory bandwidth is already saturated.

> **Saturating is not the same as occupying.**

Why?

Because this kernel is not limited by the number of compute units. It is limited
by **DRAM bandwidth**, a resource shared by the whole GPU.

Once enough blocks are generating memory requests to keep DRAM busy, giving
additional blocks to idle compute units cannot increase the amount of data DRAM
can deliver.

It only gives the scheduler more work that it does not need.

```text
             too little parallel work
                       │
                       ▼
              not enough requests
                       │
                       ▼
                 DRAM waits
                       │
                       ▼
              throughput is low
                       │
                       │ add more programs
                       ▼
              enough requests
                       │
                       ▼
                 DRAM is busy
                       │
                       ▼
              throughput saturates
                       │
                       │ add more programs
                       ▼
                 no further gain
```

This is why the table flattens around 12 blocks even though the GPU can hold
far more work.

In fact, Triton compiles this kernel to 128 threads per block, so 12 blocks
represent only 48 warps, compared with the **1,152 warp slots** this GPU can
hold resident.

That is only about **4% occupancy**, yet the memory system is already saturated.

#### What does this tell us about latency?

We should **not** reason like this:

```text
16× oversubscription
        ↓
400 cycles hidden
```

That does not follow.

The experiment shows something more useful:

```text
more independent work
        ↓
more memory requests available
        ↓
more opportunity to run while other work waits
        ↓
better coverage of memory latency
        ↓
until the bottleneck saturates
```

The exact amount of parallelism required depends on the kernel:

* how many warps can be resident,
* how much independent memory work each warp can generate,
* how much latency the memory system has,
* and, importantly, **what resource is actually the bottleneck**.

> **Oversubscription creates opportunities for latency hiding. It does not
> guarantee that all latency disappears.**

This experiment shows the left half of that story:

> **Too little independent work leaves the memory system underfed; enough
> independent work lets the GPU cover the waiting and reach the bottleneck's
> throughput ceiling.**

---

## Next

The machine never waits — but we have been saying "warp" for two parts without
looking at one. **[Lecture 7c](Lecture7c.md)** opens it up: SIMT, divergence, the
tile decomposition that finally decodes the kernel name from
[7a](Lecture7a.md), and occupancy — the budget residency lives on.

References for all three parts are collected at the end of
[Lecture 7c](Lecture7c.md#references-and-further-reading).
