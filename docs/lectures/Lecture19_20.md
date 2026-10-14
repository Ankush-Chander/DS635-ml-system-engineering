# Number formats: fixed point and floating point

> **Module thesis:** [Lecture 9/10](Lecture9_10.md) showed that decode speed is set by bytes moved per token. Quantization cuts those bytes by storing each number in fewer bits. Every format spends its bits on two competing goods: **range** (how large a value fits) and **precision** (how finely values are spaced). Fixed point fixes that split once for the whole number line; floating point re-splits it at every power of two.

**Prerequisite:** [Inference optimization](Lecture9_10.md)

---

## 1. Fixed point notation

A fixed-point number is **an integer plus an agreed scale**. The hardware sees only the integer; the scale lives in the programmer's head.

```
real value = raw / 2^F
```

**Q-format** names the split. `Qm.n` means `m` integer bits and `n` fraction bits, plus one sign bit.

```
Q3.4, 8 bits:   s  i i i . f f f f
1.5         =   0  0 0 1 . 1 0 0 0   →  raw = 24  (1.5 × 16)
```

| Operation | On the raw integers | Why |
| --- | --- | --- |
| **Encode** | `raw = round(x × 2^F)` | move the binary point right by `F` |
| **Decode** | `x = raw / 2^F` | move it back |
| **Add** | `raw_a + raw_b` | both share one scale, so it is plain integer addition |
| **Multiply** | `(raw_a × raw_b) >> F` | the product carries scale `2^(2F)`; shifting by `F` restores `2^F` |
| **Overflow** | clamp to `[min, max]` | **saturate**; two's-complement wraparound turns `7 + 7` into a negative number |

### Code example

The class built in the classroom exercise behaves like this:

```python
>>> from fixed_point import FixedPoint as FP
>>> FP(1.5)
1.5 (Raw: 24)
>>> FP(1.5) + FP(2.25)
3.75 (Raw: 60)
>>> FP(1.5) * FP(2.0)
3.0 (Raw: 48)
>>> FP(7.0) + FP(7.0)          # saturates at the top of Q3.4
7.9375 (Raw: 127)
>>> FP(0.1)                    # 0.1 is not on the grid
0.125 (Raw: 2)
```

### Classroom exercise — build it

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Ankush-Chander/DS635-ml-system-engineering?devcontainer_path=.devcontainer/fixed-point/devcontainer.json)

Fill TODOs A–F in [`code/quantization/fixed_point/fixed_point.py`](https://github.com/Ankush-Chander/DS635-ml-system-engineering/tree/main/code/quantization/fixed_point): scale, range, saturation, addition, and a rounded multiply. `pytest -q` runs 16 self-checks, each named after the TODO it covers.

---

## 2. Range vs precision

With `B` bits, one sign bit, `I` integer bits and `F = B − 1 − I` fraction bits:

```
step  Δ = 2^-F
range   = [ -2^I , 2^I - Δ ]
```

**Every bit moved from fraction to integer doubles the range and doubles the step.** The total number of representable values is fixed at `2^B`; the format only chooses where they sit.

| 8-bit format | Step Δ | Range | Good for |
| --- | --- | --- | --- |
| Q7.0 | 1 | [−128, 127] | counts, indices |
| Q3.4 | 0.0625 | [−8, 7.9375] | moderate activations |
| Q0.7 | 0.0078125 | [−1, 0.9921875] | weights normalised to [−1, 1] |

**Two failure modes, one at each end:**

- **Clipping** — a value beyond the range saturates. Error is unbounded in principle: `100.0` in Q3.4 is stored as `7.9375`.
- **Rounding** — a value inside the range snaps to the nearest grid point. Error is at most `Δ/2`.

A larger range shrinks clipping error and grows rounding error. Choosing the split is choosing which error to pay.

!!! question "Pick a format"
    A layer's weights lie in [−0.9, 0.9] except for one value at 6.0. Which 8-bit Q-format do you pick, and what happens to the other weights? This is the outlier problem that per-channel and per-group scales exist to solve.

---

## 3. Truncating vs rounding off

After a multiply, the low `F` bits of the product must go. There are two ways to drop them.

| Mode | Code | Error per op | Mean error |
| --- | --- | --- | --- |
| **Truncate** | `p >> F` | in `(−Δ, 0]` | **≈ −Δ/2, always the same sign** |
| **Round half up** | `(p + (1 << (F-1))) >> F` | in `(−Δ/2, Δ/2]` | ≈ 0, except ties go up |
| **Round half to even** | ties go to the even neighbour | in `[−Δ/2, Δ/2]` | ≈ 0 |

In Python, `>>` on a negative integer floors toward −∞, so truncation by shift always rounds down, never toward zero.

### Errors that do not cancel accumulate

A dot product sums thousands of products. Zero-mean errors partly cancel; a biased error grows linearly with the length.

```python
import random
random.seed(0)
F = 4
xs = [random.randint(-64, 63) for _ in range(10_000)]
ys = [random.randint(-64, 63) for _ in range(10_000)]

exact = sum(x * y for x, y in zip(xs, ys)) / (1 << (2 * F))
trunc = sum((x * y) >> F for x, y in zip(xs, ys)) / (1 << F)
rnd   = sum((x * y + (1 << (F - 1))) >> F for x, y in zip(xs, ys)) / (1 << F)
```

| Mode | Sum | Error |
| --- | ---: | ---: |
| Exact | 687.09 | — |
| Truncate | 434.00 | **−253.09** |
| Round half up | 725.31 | +38.22 |
| Round half to even | 684.25 | −2.84 |

- **Truncation** loses between 0 and `Δ = 0.0625` on every product, 0.025 on average here, and 10,000 of them add up to −253.
- **Round half up** is close, but 1,266 of the 10,000 products land exactly on a tie, and every tie goes up.
- **Round half to even** sends half the ties each way, and the bias disappears. IEEE 754 uses this mode by default.

---

## 4. Floating point notation

Fixed point spaces its values evenly. Floating point spaces them **evenly in relative terms**: the same number of values between 1 and 2 as between 1024 and 2048.

```
value = (-1)^s × 1.m × 2^(e - bias)

FP32:   s | eeeeeeee | mmmmmmmmmmmmmmmmmmmmmmm
        1 |    8     |          23
```

- **Exponent bits buy range.** Each extra bit squares the largest representable magnitude.
- **Mantissa bits buy precision.** `M` mantissa bits give `2^M` steps in every power-of-two interval.
- **The step grows with the value.** Relative error stays near `2^-(M+1)`; absolute error does not.

### The step grows with magnitude

```python
>>> import numpy as np
>>> np.float32(16777217)                 # 2^24 + 1
16777216.0
>>> np.float16(2048) + np.float16(1)     # FP16 step at 2048 is 2
2048.0
>>> np.float16(70000.0)                  # beyond FP16 max
inf
>>> np.float16(1e-8)                     # below FP16's smallest subnormal
0.0
```

**Special values:** an all-ones exponent encodes ±∞ and NaN, and an all-zeros exponent encodes zero and the **subnormals**, which trade precision for a gentler underflow.

---

## 5. Different numerical formats

| Format | Bits (s / e / m) | Max | Step at 1.0 (eps) | Where it is used |
| --- | --- | ---: | ---: | --- |
| **FP32** | 1 / 8 / 23 | 3.4 × 10³⁸ | 1.2 × 10⁻⁷ | master weights, optimiser state |
| **FP16** | 1 / 5 / 10 | 65,504 | 9.8 × 10⁻⁴ | inference; training with loss scaling |
| **BF16** | 1 / 8 / 7 | 3.4 × 10³⁸ | 7.8 × 10⁻³ | training: FP32's range at half the bytes |
| **FP8 E4M3** | 1 / 4 / 3 | 448 | 0.125 | weights and activations |
| **FP8 E5M2** | 1 / 5 / 2 | 57,344 | 0.25 | gradients, which need range |
| **FP4 E2M1** | 1 / 2 / 1 | 6 | 0.5 | block-scaled weights (MXFP4, NVFP4) |
| **INT8** | 8-bit integer | 127 × scale | 1 × scale | weights and activations with a scale |
| **INT4** | 4-bit integer | 7 × scale | 1 × scale | weight-only LLM quantization |

- **BF16 vs FP16:** the same 16 bits split two ways. BF16 keeps FP32's exponent, so a value that fits in FP32 does not overflow; it gives up three mantissa bits to do it.
- **The two FP8s:** E4M3 has no infinities, which buys back range up to 448. E5M2 keeps FP16's exponent range.
- **FP4 E2M1** has only eight magnitudes: `0, 0.5, 1, 1.5, 2, 3, 4, 6`. It is usable only because a shared scale per block of 16 or 32 values moves that grid to where the data sits.
- **INT8 and INT4 are fixed point with an external scale.** Generalise Section 1's power-of-two scale `2^-F` to any real scale, chosen per tensor, per channel or per group, and you have symmetric integer quantization.

!!! question "Bytes to speed"
    A 27B-parameter model is stored in BF16, then in FP8, then in INT4. Using Lecture 9/10's decode ceiling (`bandwidth ÷ model size`), what speedup does each step predict, and what does each format give up to get it?

---

## References

1. IEEE, *IEEE Standard for Floating-Point Arithmetic* (IEEE 754-2019)
2. Micikevicius et al., [*FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433) (2022) — E4M3 and E5M2
3. Open Compute Project, [*Microscaling Formats (MX) Specification v1.0*](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) (2023) — block-scaled FP8, FP6, FP4
4. Goldberg, [*What Every Computer Scientist Should Know About Floating-Point Arithmetic*](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) (1991)
5. [Lecture 9/10](Lecture9_10.md) — why bytes per parameter set decode speed
