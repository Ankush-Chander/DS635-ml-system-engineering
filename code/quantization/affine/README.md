# Affine quantization by hand

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Ankush-Chander/DS635-ml-system-engineering?devcontainer_path=.devcontainer/affine/devcontainer.json)

**FixedPoint with both restrictions lifted.** The scale can be any positive
real, and real 0.0 can sit on any integer.

## Two maps

| | Fixed point (Q3.4) | Affine |
| --- | --- | --- |
| Decode | `real = raw / 2**F` | `real = S * (q - Z)` |
| Encode | `raw = round(real * 2**F)` | `q = clip(round(real / S + Z), q_min, q_max)` |
| Scale | power of two, fixed by the format | `S = (r_max - r_min) / (q_max - q_min)`, fitted to the data |
| Zero | real 0.0 is raw 0 | real 0.0 is the integer `Z = round(q_min - r_min / S)` |

Q3.4 is the affine map with `S = 1/16`, `Z = 0`. One test checks exactly that.

## Run

```bash
pytest -q                   # self-checks, grouped by TODO
python affine_quantizer.py  # small demo
```

Fill in `affine_quantizer.py` in order: **A** grid, **B** stretch to zero,
**C** scale, **D** zero-point, **E** quantize, **F** dequantize.
**G** (stretch) is the four-term integer dot product. Each test name starts
with the TODO it checks.

## Think about

- FixedPoint rounded ties **up**; `np.round` rounds them **to even**. Which
  one did the Section 3 dot product prefer, and why do two frameworks that
  disagree here drift apart on the same model?
- Calibrate on `[2.0, 5.0]`. How many of the 256 integers now sit below 2.0
  and are never used? What would break if TODO B were skipped?
- A ReLU output is exactly 0.0. Pad a convolution input with integer `Z`.
  Why must `dequantize(Z)` be exactly 0.0, not 0.004?
- In TODO G, set `z_w = 0`. Which two terms vanish, and why is that the
  reason weights are usually quantized symmetrically?
