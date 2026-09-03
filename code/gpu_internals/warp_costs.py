"""Two microbenchmarks for the GPU execution model, in Triton.

  1. divergence  -- what a branch costs when it splits *inside* a warp
  2. coalescing  -- what the same loads cost when a warp's addresses are
                    contiguous versus scattered

Both run on any Triton-capable GPU (NVIDIA via CUDA, AMD via ROCm) and on a
free Colab T4. Nothing here is vendor-specific.

    python warp_costs.py

Every result is printed as a fraction of the machine's theoretical peak as
well as a ratio. That is not decoration -- an earlier version of this file
reported a "divergent" kernel running at 175% of peak, which is how we found
out the compiler had closed-formed the loop out from under the benchmark. If a
number here exceeds 100% of peak, the benchmark is broken, not the hardware.
"""

import torch
import triton
import triton.language as tl

# Theoretical FP32 peak of the course laptop's GPU, from its own parts:
#   36 CU x 2 SIMDs x 32 lanes = 2304 FP32 ALUs, 2 FLOP per FMA, 2.465 GHz boost.
# Override for your own machine.
ALUS, BOOST_HZ = 2304, 2.465e9
PEAK = ALUS * 2 * BOOST_HZ


def bench(fn, warmup=25, rep=200):
    """Median milliseconds over `rep` runs, after `warmup` untimed runs."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def report(label, ms, flop):
    tflops = flop / (ms * 1e-3) / 1e12
    print(f"   {label:36s}: {ms:8.3f} ms  {tflops:6.2f} TFLOP/s "
          f"({flop / (ms * 1e-3) / PEAK * 100:3.0f}% of peak)")


# --------------------------------------------------------------------------
# 1. Divergence
#
# The loop body is deliberately NOT an affine recurrence. `y = y*c + d`
# repeated N times has a closed form, and the compiler will find it -- which
# silently deletes the work the benchmark is trying to time. The `maximum`
# keeps the chain honest.
# --------------------------------------------------------------------------

ITERS = 2048


@triton.jit
def _one_path(x_ptr, out_ptr, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    """No branch: every lane in the warp wants the same path, so one path runs."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    y = tl.load(x_ptr + offs, mask=mask, other=0.0)

    for _ in range(ITERS):
        y = tl.maximum(y * 1.0000001 + 1.0, 0.0) * 0.9999

    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _divergent(x_ptr, out_ptr, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    """The branch splits *inside* the warp: alternating lanes take opposite paths.

    The hardware cannot issue two different instructions to one warp, so it
    runs path A with the odd lanes masked off and path B with the even lanes
    masked off. Each element still receives ITERS iterations of useful work --
    but the warp has issued 2 x ITERS instructions to deliver it.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    a = x
    b = x
    for _ in range(ITERS):
        a = tl.maximum(a * 1.0000001 + 1.0, 0.0) * 0.9999
        b = tl.maximum(b * 0.9999999 - 1.0, 0.0) * 1.0001

    tl.store(out_ptr + offs, tl.where((offs % 2) == 0, a, b), mask=mask)


def divergence(n=1 << 20, BLOCK=1024):
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, BLOCK),)
    flop = n * ITERS * 2  # FMAs, x2 FLOP each, per element per iteration

    t_one = bench(lambda: _one_path[grid](x, out, n, BLOCK=BLOCK, ITERS=ITERS))
    t_div = bench(lambda: _divergent[grid](x, out, n, BLOCK=BLOCK, ITERS=ITERS))
    # Same instruction count as the divergent kernel, but as ONE dependent
    # chain instead of two independent ones. This is the control.
    t_2x = bench(lambda: _one_path[grid](x, out, n, BLOCK=BLOCK, ITERS=2 * ITERS))

    print("\n1. DIVERGENCE  (same useful work per element)")
    report("warp agrees, one path runs", t_one, flop)
    report("warp splits on (lane % 2)", t_div, 2 * flop)
    report("2x instructions, one dependent chain", t_2x, 2 * flop)
    print(f"\n   divergence costs 2x the instructions, but only "
          f"{t_div / t_one:.2f}x the time")
    print(f"   the same 2x as a dependent chain costs {t_2x / t_one:.2f}x")
    print("   -> the second path rides in issue slots the first was wasting")


# --------------------------------------------------------------------------
# 2. Coalescing
#
# Both kernels read exactly n floats from exactly the same buffer and write
# exactly n floats. Only the *order* of the addresses differs.
# --------------------------------------------------------------------------


@triton.jit
def _coalesced(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Neighbouring lanes read neighbouring addresses -> one transaction per warp."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask, other=0.0), mask=mask)


@triton.jit
def _strided(x_ptr, out_ptr, n, stride, BLOCK: tl.constexpr):
    """Neighbouring lanes read addresses `stride` floats apart.

    Same buffer, same elements, each read exactly once -- this is a permutation
    of the visiting order and nothing more. But each lane now lands on a
    different cache line, so one warp's load becomes many transactions.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rows = n // stride
    src = (offs % stride) * rows + (offs // stride)  # a permutation of 0..n-1
    tl.store(out_ptr + offs, tl.load(x_ptr + src, mask=mask, other=0.0), mask=mask)


def coalescing(BLOCK=1024, strides=(2, 8, 32, 128, 1024)):
    """Sweep the scatter, at a size that fits in last-level cache and one that does not.

    A single stride at a single size is misleading: this GPU has 96 MB of L3,
    and while the working set fits in it the cache quietly absorbs most of the
    penalty. The scatter only shows its real price once the reuse window
    outgrows the cache.
    """
    print("\n2. COALESCING  (same buffer, same elements, different order)")
    for n in (1 << 22, 1 << 26):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, BLOCK),)
        moved_gb = 2 * n * 4 / 1e9  # read + write, in GB
        t_co = bench(lambda: _coalesced[grid](x, out, n, BLOCK=BLOCK))

        print(f"\n   buffer {n * 4 / 2**20:.0f} MB in + {n * 4 / 2**20:.0f} MB out"
              f"   (L3 is 96 MB)")
        print(f"     contiguous          : {t_co:8.3f} ms"
              f"  {moved_gb / (t_co * 1e-3):7.1f} GB/s useful")
        for s in strides:
            t_st = bench(lambda: _strided[grid](x, out, n, s, BLOCK=BLOCK))
            print(f"     stride {s:<5d}        : {t_st:8.3f} ms"
                  f"  {moved_gb / (t_st * 1e-3):7.1f} GB/s useful"
                  f"   {t_st / t_co:6.2f}x")


def check():
    """The benchmark is worthless if the kernels compute the wrong thing."""
    n, BLOCK = 1 << 16, 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, BLOCK),)

    _divergent[grid](x, out, n, BLOCK=BLOCK, ITERS=64)
    a = x.clone()
    b = x.clone()
    for _ in range(64):
        a = torch.clamp(a * 1.0000001 + 1.0, min=0.0) * 0.9999
        b = torch.clamp(b * 0.9999999 - 1.0, min=0.0) * 1.0001
    offs = torch.arange(n, device="cuda")
    assert torch.allclose(out, torch.where(offs % 2 == 0, a, b), rtol=1e-3, atol=1e-2)

    _strided[grid](x, out, n, 32, BLOCK=BLOCK)
    assert torch.allclose(out, x.view(32, -1).t().reshape(-1))
    print("kernels : correct")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a Triton-capable GPU"
    print(f"device  : {torch.cuda.get_device_name(0)}")
    print(f"torch   : {torch.__version__}   triton: {triton.__version__}")
    print(f"peak    : {PEAK / 1e12:.1f} TFLOP/s FP32 "
          f"({ALUS} ALUs x 2 x {BOOST_HZ / 1e9:.3f} GHz)")
    check()
    divergence()
    coalescing()
