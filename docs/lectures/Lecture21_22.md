# Affine quantization

> **Module thesis:** [Lecture 19/20](Lecture19_20.md) built fixed point: an integer, a power-of-two scale, and a grid centred on zero. A real tensor needs a grid fitted to its own data, with any real scale and an offset that places the grid over `[r_min, r_max]`. That map is affine quantization, and every int8 inference stack runs on it.

**Prerequisite:** [Number formats](Lecture19_20.md)

---

## 1. From fixed point to affine

[Lecture 19/20](Lecture19_20.md)'s fixed point carries two restrictions. **The scale is a power of two**, so the grid can only zoom by factors of 2. **Real 0.0 is pinned to raw 0**, so the grid is always centred on zero. Affine quantization lifts both.

```
dequantize:   r = S · (q − Z)
quantize:     q = clip( round(r / S + Z), q_min, q_max )
```

| Symbol | Type | Meaning |
| --- | --- | --- |
| `r` | float32 | the real weight or activation |
| `q` | int8 / uint8 | what is stored and multiplied |
| `S` | float32, > 0 | **scale**: the real value of one integer step |
| `Z` | integer, same type as `q` | **zero-point**: the integer that means real 0.0 |

**Fixed point is the special case `S = 2^-F`, `Z = 0`.** Q3.4 is the affine map with `S = 1/16` and `Z = 0`; `FP(1.5)` and an affine `quantize(1.5)` both store 24.

---

## 2. Fitting S and Z to the data

Map the data's `[r_min, r_max]` onto the full grid `[q_min, q_max]`. That gives two equations in two unknowns:

```
r_min = S · (q_min − Z)
r_max = S · (q_max − Z)

S = (r_max − r_min) / (q_max − q_min)      subtract the two
Z = round( q_min − r_min / S )             rearrange the first
```

- **`S` is the real range per integer step.** Weights in `[−0.123, 0.123]` on int8 give `S = 0.246 / 255 ≈ 0.00096`.
- **`Z` must be rounded**, because it is a grid position. Rounding nudges the represented range slightly away from `[r_min, r_max]`.

Worked example, uint8, data in `[−1.0, 3.0]`:

| Step | Value |
| --- | --- |
| `S = 4.0 / 255` | 0.01569 |
| `Z = 0 − (−1.0) / S` | 63.75 → **64** |
| Range actually covered, `S·(0 − 64)` to `S·(255 − 64)` | `[−1.004, 2.996]`: the nudge |

---

## 3. Real zero must be exact

**Real 0.0 must map to the integer `Z` with no error.** Two everyday operations depend on it:

- **Zero-padding.** A convolution pads its integer input with `Z`. If `Z` decodes to 0.05, the image gets a grey border and the edge-detecting filters fire on it.
- **ReLU sparsity.** ReLU outputs exact zeros, and hardware skips them. If 0.0 decodes to 0.001, nothing is skippable.

So calibration **stretches the range to include 0.0** before solving for `S` and `Z`. Data in `[2.0, 5.0]` is calibrated as `[0.0, 5.0]`.

---

## 4. Granularity: who shares one (S, Z)

| Granularity | One (S, Z) per | Cost | Outlier damage |
| --- | --- | --- | --- |
| **Per-tensor** | whole tensor | 2 numbers | one outlier sets `S` for everything |
| **Per-channel** | output channel | 2 per channel | confined to its channel |
| **Per-group** | block of 32 / 64 / 128 | 2 per block | confined to its block; standard for 4-bit LLM weights |

Lecture 19/20's outlier question, in affine form: values in `[0, 1]` plus one value at 100.0, per-tensor uint8, give `S = 100 / 255 ≈ 0.39`. **All of `[0, 1]` lands in four integers, 0 to 3.**

---

## 5. What Z costs in a dot product

Substitute `r = S · (q − Z)` for both weights `w` and inputs `x`:

```
y = S_w · S_x · Σ (q_w − Z_w)(q_x − Z_x)
  = S_w · S_x · ( Σ q_w q_x  −  Z_x Σ q_w  −  Z_w Σ q_x  +  N · Z_w Z_x )
```

| Term | When it is computed |
| --- | --- |
| `Σ q_w q_x` | every input: the main multiply-accumulate loop, as in fixed point |
| `−Z_x Σ q_w` | once per weight row, offline |
| `−Z_w Σ q_x` | **every input**: an extra pass over the activations |
| `N · Z_w Z_x` | once, folded into the bias |

**Symmetric weights (`Z_w = 0`) delete the third and fourth terms.** That is why weights are usually quantized symmetrically while activations keep a zero-point: post-ReLU activations are all ≥ 0, and a symmetric grid would waste half its integers on negatives that never occur.

---

## Classroom exercise — build it

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Ankush-Chander/DS635-ml-system-engineering?devcontainer_path=.devcontainer/affine/devcontainer.json)

Fill TODOs A–F in [`code/quantization/affine/affine_quantizer.py`](https://github.com/Ankush-Chander/DS635-ml-system-engineering/tree/main/code/quantization/affine): integer grid, stretch to zero, scale, zero-point, quantize, dequantize. TODO G (stretch) is the four-term integer dot product. `pytest -q` runs 18 self-checks, each named after the TODO it covers; one of them recovers Q3.4 as `S = 1/16`, `Z = 0`.

!!! question "Ties again"
    `np.round` sends ties to the even neighbour; FixedPoint sent them up. Calibrate a signed int8 quantizer on `[−8, 7.9375]` and quantize `0.03125` and `0.09375`. Which integers come back, and which row of Lecture 19/20's rounding table does that match?

---

## References

1. Jacob et al., [*Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*](https://arxiv.org/abs/1712.05877) (CVPR 2018) — the affine scheme `r = S(q − Z)` and its integer dot product
2. [Lecture 19/20](Lecture19_20.md) — fixed point, rounding modes, and the formats being quantized into
3. [Lecture 9/10](Lecture9_10.md) — why bytes per parameter set decode speed
