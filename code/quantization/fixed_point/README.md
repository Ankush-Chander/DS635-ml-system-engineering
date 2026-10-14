# Fixed point by hand

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Ankush-Chander/DS635-ml-system-engineering?devcontainer_path=.devcontainer/fixed-point/devcontainer.json)

A real number stored as an integer plus an agreed scale. Every int8
quantization scheme is this idea with a per-tensor or per-group scale.

## Format

| Format | Bits | Step | Range |
| --- | --- | --- | --- |
| Q3.4 (default) | 1 sign + 3 int + 4 frac = 8 | 1/16 = 0.0625 | [-8.0, 7.9375] |

`real = raw / scale`, with `scale = 2**frac_bits`.

## Run

```bash
pytest -q              # self-checks, grouped by TODO
python fixed_point.py  # small demo
```

Fill in `fixed_point.py` in order: **A** scale, **B** range, encode,
**C** saturate, add, **D-F** multiply. Each test name starts with the
TODO it checks.

## Think about

- Why does `7.0 + 7.0` print `7.9375`?
- `0.0625 * 0.5` gives raw 1, but `-0.0625 * 0.5` gives raw 0. Why are
  ties asymmetric, and what does that do across a million-term dot product?
- Q1.6 and Q3.4 are both 8 bits. Which one would you pick for weights in
  [-1, 1], and what do you give up?
