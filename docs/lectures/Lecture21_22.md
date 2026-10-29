# Affine quantization

> **Module thesis:** [Lecture 19/20](Lecture19_20.md) built fixed point: an integer, a power-of-two scale, and a grid centred on zero. A real tensor needs a grid fitted to its own data, with any real scale and an offset that places the grid over $[r_{min}, r_{max}]$. That map is affine quantization, and every int8 inference stack runs on it.

**Prerequisite:** [Number formats](Lecture19_20.md)

---

## 1. The fundamental equations

Four quantities describe every quantized tensor [1]:

| Symbol | Type | Meaning |
| --- | --- | --- |
| $r$ | float32 | the **real** weight, activation or bias |
| $q$ | int8 / uint8 | the **quantized** integer: what is stored and multiplied |
| $S$ | float32, $> 0$ | **scale**: the real value of one integer step |
| $Z$ | integer, same type as $q$ | **zero-point**: the integer that means real $0.0$ |

![](https://media.licdn.com/dms/image/v2/D5612AQEuNGuWfPPYfQ/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1721202548242?e=1794441600&v=beta&t=bWFplUDUzhSbIQqWLDnYUXn7UHCqkIPaWJAxBWRyw-o)

### Dequantize: integer → real

$$
r = S \cdot (q - Z)
\tag{2.2}
$$

**This is a straight line, $y = mx + c$, with slope $m = S$ and intercept $c = -S \cdot Z$.** Read it right to left: subtract $Z$ to "untare" the integer, then scale by the step size $S$.

### Quantize: real → integer

Invert Equation 2.2:

$$
q = \operatorname{round}\!\left(\frac{r}{S} + Z\right)
\tag{2.3}
$$

$q$ has a fixed bit width, so values outside the grid must be **clipped** to it:

$$
q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{r}{S} + Z\right),\; q_{min},\; q_{max}\right)
\tag{2.4}
$$

- $\operatorname{clip}(v, a, b) = \min(\max(v, a), b)$.
- $q_{min}, q_{max}$ are the integer type's limits: $[-128, 127]$ for int8, $[0, 255]$ for uint8.
- **The tie-breaking rule inside $\operatorname{round}$ is a source of drift between frameworks** — Lecture 19/20 §4 measured it.

**Fixed point is the special case $S = 2^{-F}$, $Z = 0$.** Q3.4 is the affine map with $S = 1/16$, $Z = 0$; `FP(1.5)` and an affine `quantize(1.5)` both store 24.

### Solving for S and Z

Map the data's $[r_{min}, r_{max}]$ onto the full grid $[q_{min}, q_{max}]$. Each end gives one equation, so there are two equations in two unknowns:

$$
\begin{aligned}
r_{min} &= S\,(q_{min} - Z) && \text{(constraint 1)} \\
r_{max} &= S\,(q_{max} - Z) && \text{(constraint 2)}
\end{aligned}
$$

**Subtract constraint 1 from constraint 2** and $Z$ cancels:

$$
S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}
\tag{2.5}
$$

$S$ is the real range per integer step. Weights in $[-0.123, 0.123]$ on int8 give $S = 0.246 / 255 \approx 0.00096$.

**Rearrange constraint 1** for $Z$:

$$
Z = q_{min} - \frac{r_{min}}{S}
\tag{2.6}
$$

$Z$ is a position on the integer grid, and Equation 2.6 almost never lands on one. **So $Z$ is rounded:**

$$
Z = \operatorname{round}\!\left(q_{min} - \frac{r_{min}}{S}\right)
\tag{2.7}
$$

**Rounding $Z$ nudges the represented range.** The grid covers a range shifted slightly off $[r_{min}, r_{max}]$. This is the first quantization error, and it is separate from the rounding of each individual value.

### Worked example: uint8, data in [−1.0, 3.0]

| Step | Value |
| --- | --- |
| $S = 4.0 / 255$ | $0.01569$ |
| $Z = 0 - (-1.0)/S$ | $63.75 \to$ **64** |
| Range covered, $S(0 - 64)$ to $S(255 - 64)$ | $[-1.004,\ 2.996]$: **the nudge** |

---

## 2. Why real zero must land on an integer

**Rounding $Z$ in Equation 2.7 is what makes $0.0$ exact.** Put $r = 0$ into Equation 2.3: $q = \operatorname{round}(Z) = Z$. Decode it with Equation 2.2: $S(Z - Z) = 0$, with no error.

With an unrounded $Z = 63.75$ from the worked example, no integer means zero. The nearest one, 64, decodes to $S(64 - 63.75) \approx 0.0039$. **Every zero in the tensor would come back as 0.0039.**

Two everyday operations depend on an exact zero:

- **Zero-padding.** A convolution pads its integer input with $Z$. If $Z$ decodes to $0.05$, the image gets a grey border and the edge-detecting filters fire on it.
- **ReLU sparsity.** ReLU outputs exact zeros, and hardware skips them. If $0.0$ decodes to $0.001$, nothing is skippable.

**Zero must also be inside the range.** Equation 2.4 clips anything outside $[r_{min}, r_{max}]$, so data in $[2.0, 5.0]$ would clip $0.0$ to $2.0$. Calibration stretches the range to include zero first: $[2.0, 5.0]$ is calibrated as $[0.0, 5.0]$.

!!! question "Wasted integers"
    Calibrate uint8 on $[2.0, 5.0]$ after stretching to $[0.0, 5.0]$. How many of the 256 integers now sit below $2.0$ and are never used?

---

## 3. Classroom exercise — build it

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Ankush-Chander/DS635-ml-system-engineering?devcontainer_path=.devcontainer/affine/devcontainer.json)

Fill TODOs A–F in [`code/quantization/affine/affine_quantizer.py`](https://github.com/Ankush-Chander/DS635-ml-system-engineering/tree/main/code/quantization/affine):

| TODO | Implements |
| --- | --- |
| **A** | integer grid $[q_{min}, q_{max}]$ |
| **B** | stretch the range to include $0.0$ (§2) |
| **C** | scale, Equation 2.5 |
| **D** | zero-point, Equation 2.7 |
| **E** | quantize, Equation 2.4 |
| **F** | dequantize, Equation 2.2 |
| **G** (stretch) | the four-term integer dot product (§4) |

`pytest -q` runs 18 self-checks, each named after the TODO it covers. One of them recovers Q3.4 as $S = 1/16$, $Z = 0$. Submitted and marked as [Lab 19–22a](../labs/Lab19_22a.md).

!!! question "Ties again"
    `np.round` sends ties to the even neighbour; FixedPoint sent them up. Calibrate a signed int8 quantizer on $[-8, 7.9375]$ and quantize $0.03125$ and $0.09375$. Which integers come back, and which row of Lecture 19/20's rounding table does that match?

---

## 4. Cost: affine vs fixed point

**Both store one integer per value. The difference is what a dot product costs.**

### Fixed point

Every value shares one power-of-two scale $2^{-F}$:

$$
y = 2^{-2F} \sum_i a_i\, b_i
$$

One integer multiply-accumulate loop, then **one shift** by $F$ to return to scale $2^{-F}$.

### Affine

Substitute $r = S(q - Z)$ for both weights $w$ and inputs $x$:

$$
\begin{aligned}
y &= S_w S_x \sum_i (q_{w,i} - Z_w)(q_{x,i} - Z_x) \\
  &= S_w S_x \Big( \underbrace{\textstyle\sum q_w q_x}_{\text{core}}
     \;-\; \underbrace{Z_x \textstyle\sum q_w}_{\text{weights only}}
     \;-\; \underbrace{Z_w \textstyle\sum q_x}_{\text{every input}}
     \;+\; \underbrace{N Z_w Z_x}_{\text{constant}} \Big)
\end{aligned}
$$

| Term | When it is computed |
| --- | --- |
| $\sum q_w q_x$ | every input: the main multiply-accumulate loop, as in fixed point |
| $-Z_x \sum q_w$ | once per weight row, offline |
| $-Z_w \sum q_x$ | **every input**: an extra pass over the activations |
| $N Z_w Z_x$ | once, folded into the bias |

**The rescale is a multiply, not a shift.** The int32 result must be written back on the output's grid, scale $S_y$, by multiplying by $M = S_w S_x / S_y$. $M$ is an arbitrary real. Integer-only hardware stores it as $M = 2^{-n} M_0$, with $M_0$ an int32 fixed-point multiplier, and pays an integer multiply plus a shift.

### Side by side

| Cost | Fixed point | Affine, $Z_w = 0$ | Affine, both $Z \neq 0$ |
| --- | --- | --- | --- |
| **Inner loop** | $\sum a b$ | $\sum q_w q_x$ | $\sum q_w q_x$ |
| **Extra pass per input** | none | none | $Z_w \sum q_x$ |
| **Offline terms** | none | $Z_x \sum q_w$ | $Z_x \sum q_w$, $N Z_w Z_x$ |
| **Output rescale** | shift by $F$ | multiply by $M_0$, shift | multiply by $M_0$, shift |
| **Grid fitted to data** | no | yes | yes |

**Symmetric weights ($Z_w = 0$) delete the third and fourth terms.** That is why weights are usually quantized symmetrically while activations keep a zero-point. Post-ReLU activations are all $\geq 0$, and a symmetric grid would waste half its integers on negatives that never occur.

---

## 5. Granularity: who shares one (S, Z)

![](https://apple.github.io/coremltools/docs-guides/_images/quantization-granularities.png)

| Granularity | One $(S, Z)$ per | Cost | Outlier damage |
| --- | --- | --- | --- |
| **Per-tensor** | whole tensor | 2 numbers | one outlier sets $S$ for everything |
| **Per-channel** | output channel | 2 per channel | confined to its channel |
| **Per-group** | block of 32 / 64 / 128 | 2 per block | confined to its block; standard for 4-bit LLM weights |

**The outlier, in affine form.** Values in $[0, 1]$ plus one value at $100.0$, per-tensor uint8, give $S = 100/255 \approx 0.39$. **All of $[0, 1]$ lands in four integers, 0 to 3.**

### What smaller groups cost

Metadata is paid per group, so it is spread over the group size $g$:

$$
\text{bits per weight} = b + \frac{\text{metadata bits}}{g}
$$

| 4-bit scheme, $g = 32$ | Metadata per group | Bits per weight |
| --- | --- | ---: |
| scale only, FP16 | 16 bits | $4 + 16/32 =$ **4.5** |
| scale + offset, FP16 each | 32 bits | $4 + 32/32 =$ **5.0** |

**Halving $g$ doubles the metadata overhead.** These two rows are exactly llama.cpp's `Q4_0` and `Q4_1`.

<!--
---

## 6. Classroom exercise — dissect a GGUF block

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Ankush-Chander/DS635-ml-system-engineering?devcontainer_path=.devcontainer/gguf-block/devcontainer.json)

**The affine quantizer as shipped in llama.cpp.** `Q4_0` is symmetric; `Q4_1` is asymmetric. Both use blocks of 32 weights.

The Codespace checks out the three llama.cpp source files at tag [`b10758`](https://github.com/ggml-org/llama.cpp/tree/b10758) next to [`code/quantization/gguf_block/gguf_block.py`](https://github.com/Ankush-Chander/DS635-ml-system-engineering/tree/main/code/quantization/gguf_block). `pytest -q` runs 61 self-checks; TODOs D–F are compared against `gguf`, llama.cpp's own Python package, byte for byte.

### Read

1. **The block structs** in [`ggml/src/ggml-common.h`](https://github.com/ggml-org/llama.cpp/blob/b10758/ggml/src/ggml-common.h#L194-L269): `block_q4_0`, `block_q4_1`, `block_q8_1`. Count the bytes in each and check them against §5's table.
2. **The decoders** in [`ggml/src/ggml-quants.c`](https://github.com/ggml-org/llama.cpp/blob/b10758/ggml/src/ggml-quants.c#L459-L498): `dequantize_row_q4_0`, `dequantize_row_q4_1`. Write each one in the form $r = S(q - Z)$.

| Format | Stored per block | Decoder | As $r = S(q - Z)$ |
| --- | --- | --- | --- |
| `Q4_0` | `d` (FP16), 16 bytes of nibbles | $r = d\,(q - 8)$ | $S = d$, $Z = 8$ |
| `Q4_1` | `d`, `m` (FP16), 16 bytes of nibbles | $r = d\,q + m$ | $S = d$, $Z = -m/d$ |

### Build

| TODO | Implements |
| --- | --- |
| **A** | bytes per block and bits per weight for `Q4_0`, `Q4_1`, `Q8_1` |
| **B** | unpack 16 bytes into 32 nibbles |
| **C** | pack them back |
| **D** | decode `Q4_0` |
| **E** | decode `Q4_1` |
| **F** | encode `Q4_1`, byte-identical to llama.cpp |
| **G** (stretch) | split the dot product into core and offset terms |

- **Nibble layout.** Byte `qs[j]` holds weight `j` in its **low** nibble and weight `j + 16` in its **high** nibble.
- **Encoding** follows [`quantize_row_q4_1_ref`](https://github.com/ggml-org/llama.cpp/blob/b10758/ggml/src/ggml-quants.c#L150-L185): $d = (\max - \min)/15$, $m = \min$, $q = \min\!\big(15,\ \lfloor (r - m)/d + 0.5 \rfloor\big)$.
- **Precision.** $q$ is computed with the float32 $d$; $d$ and $m$ are stored as float16. Decoding with the float32 values does not reproduce llama.cpp's output.

### The dot product

TODO G takes a 32-element activation vector $a$ and computes $\sum_i \hat{r}_i a_i$ two ways:

$$
\underbrace{\sum_i (d\,q_i + m)\, a_i}_{\text{dequantize, then dot}}
\;=\;
\underbrace{d \sum_i q_i\, a_i}_{\text{core}} \;+\; \underbrace{m \sum_i a_i}_{\text{offset term}}
$$

The second term is §4's zero-point term. llama.cpp quantizes the activations to `block_q8_1`, which stores $s = d_a \sum q_{a,i}$ for exactly this term. [`ggml_vec_dot_q4_1_q8_1`](https://github.com/ggml-org/llama.cpp/blob/b10758/ggml/src/ggml-cpu/quants.c#L262-L296) then pays one multiply per block for it.

!!! question "Discuss"
    - `Q4_1`'s implied $Z = -m/d$ is not an integer, so real $0.0$ is not exact. Why is that acceptable for weights, given §2's two reasons?
    - `quantize_row_q4_0_ref` sets $d = \text{max} / {-8}$, where `max` is the signed value of largest magnitude. When is $d$ negative, and what does that buy?
-->

---

## References

1. Vivek Kalyanarangan, *Quantization and Fast Inference*, Manning Publications — Chapter 2: the fundamental equations (2.2–2.7), the two-constraint solve for $S$ and $Z$, and the zero-point nudge
2. Jacob et al., [*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*](https://arxiv.org/abs/1712.05877) (CVPR 2018) — the affine scheme $r = S(q - Z)$, its integer dot product, and the $M = 2^{-n} M_0$ rescale
3. [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/b10758), tag `b10758` — `ggml/src/ggml-common.h` block formats; `ggml/src/ggml-quants.c` quantize and dequantize; `ggml/src/ggml-cpu/quants.c` the Q4_1 × Q8_1 dot product
4. [Lecture 19/20](Lecture19_20.md) — fixed point, rounding modes, and the formats being quantized into
5. [Lecture 9/10](Lecture9_10.md) — why bytes per parameter set decode speed
