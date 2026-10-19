# Number formats: fixed point and floating point

> **Module thesis:** [Lecture 9/10](Lecture9_10.md) showed that decode speed is set by bytes moved per token. Quantization cuts those bytes by storing each number in fewer bits. Every format spends its bits on two competing goods: **range** (how large a value fits) and **precision** (how finely values are spaced). Fixed point fixes that split once for the whole number line; floating point re-splits it at every power of two.

**Prerequisite:** [Inference optimization](Lecture9_10.md)

A computer stores only integers. A fractional value is an integer plus a convention about where the **binary point** sits. Fixed point keeps that convention in the program, the same for every number. Floating point writes it into every number as the exponent.

---

## 1. Fractional binary numbers

Binary positional notation extends to the right of the point exactly as decimal does, with negative powers of two:

```
b = Σ  b_i · 2^i      for i = -n … m
```

| Binary digit | 1 | 0 | . | 1 | 1 | |
| --- | --- | --- | --- | --- | --- | --- |
| **Position i** | 1 | 0 | | −1 | −2 | |
| **Weight 2^i** | 2 | 1 | | 1/2 | 1/4 | |
| **Contribution** | 2 | 0 | | 1/2 | 1/4 | **= 2.75** |

### Decimal to binary: 12.375

**Integer part:** divide by 2 and read the remainders bottom-up.

```
12 ÷ 2 = 6  r 0
 6 ÷ 2 = 3  r 0
 3 ÷ 2 = 1  r 1
 1 ÷ 2 = 0  r 1      →  12 = 1100₂
```

**Fractional part:** multiply by 2 and read the integer parts top-down.

```
0.375 × 2 = 0.75   → 0
0.75  × 2 = 1.50   → 1
0.50  × 2 = 1.00   → 1      →  0.375 = 0.011₂
```

**12.375₁₀ = 1100.011₂.**

### Some decimals never terminate

The same procedure on 0.1 never reaches zero:

```
0.1₁₀ = 0.0001100110011…₂      (0011 repeats forever)
```

**A value is exact in binary only if it is a sum of powers of two.** 0.375 = 1/4 + 1/8 is exact; 0.1 is not, in any binary format, at any width. Every format below stores the nearest point on its grid.

---

## 2. Fixed point notation

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

## 3. Range vs precision

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

## 4. Truncating vs rounding off

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

## 5. Floating point notation

Fixed point spaces its values evenly. Floating point spaces them **evenly in relative terms**: the same number of values between 1 and 2 as between 1024 and 2048. The IEEE 754 standard writes every number as

```
V = (−1)^s × M × 2^E
```

- **Sign `s`** — 0 for positive, 1 for negative. Zero has both a +0 and a −0.
- **Significand `M`** — a fractional binary number, in `[1, 2)` for normal values and in `[0, 1)` for the subnormals near zero.
- **Exponent `E`** — weights the value by a power of two, positive or negative.

### What the exponent buys

Store `5 × 2^100`:

| | Fixed point | Floating point |
| --- | --- | --- |
| **Form** | `Σ b_i · 2^i` | `M × 2^E` |
| **Bits needed** | `101` followed by 100 zeros: **103 bits** | `M = 101`, `E = 100`: **3 + 7 bits** |

**Floating point stores the digits that carry information and the position separately.** The zeros that only place the point cost nothing.

### Standard formats

| Format | Sign | Exponent | Mantissa | Bias |
| --- | ---: | ---: | ---: | ---: |
| **Single precision** (FP32, `float`) | 1 | 8 | 23 | 127 |
| **Double precision** (FP64, `double`) | 1 | 11 | 52 | 1023 |

```
FP32:   s | eeeeeeee | mmmmmmmmmmmmmmmmmmmmmmm
        1 |    8     |          23
```

---

## 6. Inside an IEEE 754 float

### Normalisation: one number, one encoding

Without a rule, one value has many encodings:

```
101.11 = 10111    × 2^-2
       = 101110   × 2^-3
       = 1011100  × 2^-4
       = 10111000 × 2^-5
```

Multiple encodings waste bit patterns and make equality checks compare more than bits. **Normalisation** fixes the binary point relative to the most significant 1.

| | Explicit normalisation | Implicit normalisation (IEEE 754) |
| --- | --- | --- |
| **Rule** | point to the **left** of the leading 1 | point to the **right** of the leading 1 |
| **Form** | `(−1)^s × 0.M × 2^(e − bias)` | `(−1)^s × 1.M × 2^(e − bias)` |
| **Example** | `101.11 = 0.10111 × 2^3` | `101.11 = 1.0111 × 2^2` |
| **Leading 1** | stored | **not stored** |

**The hidden bit is free precision.** A normalised binary significand always starts with 1, so IEEE 754 leaves it out: FP32 stores 23 mantissa bits and gets 24 bits of precision.

### Biased exponent

The sign has its own bit, and the significand is always positive, so it is stored as unsigned binary. The exponent can be negative. Two's complement could hold it; IEEE 754 stores `e = E + bias` with `bias = 2^(w−1) − 1`.

| 3-bit pattern | Two's complement | Biased, `e − 3` |
| --- | ---: | ---: |
| 000 | 0 | −3 |
| 001 | 1 | −2 |
| 010 | 2 | −1 |
| 011 | 3 | 0 |
| 100 | **−4** | 1 |
| 101 | −3 | 2 |
| 110 | −2 | 3 |
| 111 | −1 | 4 |

**Biased exponents sort in the same order as their bit patterns; two's complement jumps from +3 to −4.** With the exponent above the mantissa, non-negative floats compare correctly as plain unsigned integers:

| Value | FP32 bits (hex) |
| ---: | --- |
| 1.0 | `3f800000` |
| 2.0 | `40000000` |
| 3.0 | `40400000` |

The all-zeros and all-ones patterns are reserved (see *Normalized, subnormal and special values* below), so the usable FP32 exponents run from −126 to +127.

### Worked example: 9.75 in FP32

| Step | Result |
| --- | --- |
| 1. Convert to binary | `9.75 = 1001.11₂` |
| 2. Normalise | `1.00111₂ × 2^3` |
| 3. Sign | `s = 0` |
| 4. Biased exponent | `3 + 127 = 130 = 10000010₂` |
| 5. Mantissa, hidden 1 dropped | `00111000000000000000000` |

| S | Exponent | Mantissa |
| --- | --- | --- |
| 0 | 10000010 | 00111000000000000000000 |

```
0100 0001 0001 1100 0000 0000 0000 0000  =  0x411C0000
```

**In memory the bytes are `00 00 1c 41` on x86 and most ARM systems,** which are little-endian: least significant byte at the lowest address.

```python
>>> import struct
>>> struct.pack('>f', 9.75).hex()      # big-endian: the bit pattern as written
'411c0000'
>>> struct.pack('<f', 9.75).hex()      # little-endian: the bytes as stored
'00001c41'
```

### Normalized, subnormal and special values

The exponent field decides how the rest of the bits are read. A **normalized** float has significand `1.f` with a hidden leading 1. A **subnormal** (denormalized) float has significand `0.f` and fills the gap around zero.

| Exponent bits `e` | Case | Significand M | Exponent E | Covers |
| --- | --- | --- | --- | --- |
| neither all 0s nor all 1s | **Normalized** | `1.f` (hidden 1) | `e − bias` | almost every number |
| all 0s | **Subnormal** | `0.f` (no hidden 1) | `1 − bias` | zero and tiny values near it |
| all 1s | **Special** | — | — | ±∞ when `f = 0`, NaN when `f ≠ 0` |

#### Why subnormals are needed

**With a hidden 1, every significand is at least 1, so normalized floats alone cannot represent zero and leave a hole around it.** The smallest normalized FP32 value is `1.0 × 2^-126`. Between it and 0 there would be nothing, while the step just above it is `2^-149`: a gap 2²³ times wider than its neighbourhood.

That hole breaks arithmetic that programs rely on:

- **`x − y == 0` would no longer mean `x == y`.** Take `x = 1.5 × 2^-126` and `y = 2^-126`. Both are normalized and they differ, but `x − y = 2^-127` is below the smallest normalized value. Without subnormals it flushes to 0, and code such as `if x != y: z = 1 / (x - y)` divides by zero.
- **Small values vanish abruptly.** Any result below `2^-126` drops straight to 0 and takes every later product with it.

Subnormals fill the hole. When the exponent bits are all zeros, **the hidden bit becomes 0**, so `M = 0.f`:

- **Zero falls out of the encoding:** `e = 0, f = 0` is ±0.
- **Values shrink toward 0 in even steps of `2^-149`,** so `x − y = 2^-127` is stored exactly. This is **gradual underflow**.

```python
>>> import numpy as np
>>> x, y = np.float32(1.5 * 2**-126), np.float32(2**-126)
>>> x - y                    # 2^-127: a subnormal, not zero
5.877472e-39
>>> (x - y) == 0
False
```

#### Why `E = 1 − bias`, not `0 − bias`

**It keeps the number line continuous.** The largest subnormal, `0.111…1 × 2^-126`, sits one step below the smallest normalized value, `1.000…0 × 2^-126`. The step is `2^-149` on both sides of the boundary, with no jump.

| FP32 | Value | Significant bits |
| --- | --- | ---: |
| Smallest normalized | `2^-126 ≈ 1.18 × 10⁻³⁸` | 24 |
| Largest subnormal | `(1 − 2^-23) × 2^-126` | 23 |
| Smallest subnormal | `2^-149 ≈ 1.4 × 10⁻⁴⁵` | 1 |

**The cost is precision.** A subnormal keeps the fixed step `2^-149` but loses one significant bit for every halving toward 0. Near zero, floating point turns back into fixed point.

!!! note "Why this matters in ML"
    FP16's smallest normalized value is `6.1 × 10⁻⁵`; its subnormals reach down to `6.0 × 10⁻⁸`. Small gradients routinely land in that band. They survive as subnormals with few significant bits, and anything smaller flushes to 0. Mixed-precision training scales the loss up before the backward pass to lift gradients back into the normalized range.

---

## 7. Implications

### The step grows with magnitude

**Exponent bits buy range; mantissa bits buy precision.** `M` mantissa bits give `2^M` steps in every power-of-two interval, so relative error stays near `2^-(M+1)` while absolute error grows with the value.

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

### Casting

| Cast | Kind | What happens |
| --- | --- | --- |
| `float` → `double` | widening | exact; the mantissa gains bits |
| `int` → `long` | widening | exact; sign extension copies the sign bit upward |
| `double` → `float` | narrowing | **rounded** to the nearest float; `0.1` becomes `0.10000000149011612` |
| `long` → `int` | narrowing | **high bits discarded**; `4294967301` becomes `5` |
| `int` → `float` | same width, different grid | exact only up to `2^24`; `16777217` becomes `16777216` |

### Mixing signed and unsigned

C and C++ **convert the signed operand to unsigned** when an expression mixes `int` and `unsigned int`:

```c
int a = -5;
unsigned int b = 3;
printf("%u\n", a + b);   // 4294967294: -5 is read as 2^32 - 5, then 3 is added
printf("%d\n", a < b);   // 0: the comparison is 4294967291 < 3
```

**Use unsigned types only for bit patterns and sizes**, where negative values cannot occur.

---

## 8. Different numerical formats

| Format | Bits (s / e / m) | Max | Step at 1.0 (eps) | Where it is used |
| --- | --- | ---: | ---: | --- |
| **FP64** | 1 / 11 / 52 | 1.8 × 10³⁰⁸ | 2.2 × 10⁻¹⁶ | scientific computing; rare in ML |
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
- **INT8 and INT4 are fixed point with an external scale.** Generalise Section 2's power-of-two scale `2^-F` to any real scale and an offset, and you have the affine quantization of [Lecture 21/22](Lecture21_22.md).

!!! question "Bytes to speed"
    A 27B-parameter model is stored in BF16, then in FP8, then in INT4. Using Lecture 9/10's decode ceiling (`bandwidth ÷ model size`), what speedup does each step predict, and what does each format give up to get it?

---

## References

1. Bryant and O'Hallaron, *Computer Systems: A Programmer's Perspective*, 3rd ed. — §2.2–2.3 integer representation and arithmetic, §2.4 floating point
2. IEEE, *IEEE Standard for Floating-Point Arithmetic* (IEEE 754-2019)
3. Goldberg, [*What Every Computer Scientist Should Know About Floating-Point Arithmetic*](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) (1991)
4. Micikevicius et al., [*FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433) (2022) — E4M3 and E5M2
5. Open Compute Project, [*Microscaling Formats (MX) Specification v1.0*](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) (2023) — block-scaled FP8, FP6, FP4
6. [Computer Organization & Architecture — number system videos](https://www.youtube.com/playlist?list=PLBlnK6fEyqRgLLlzdgiTUKULKJPYc0A4q) (YouTube playlist)
7. [Lecture 9/10](Lecture9_10.md) — why bytes per parameter set decode speed
