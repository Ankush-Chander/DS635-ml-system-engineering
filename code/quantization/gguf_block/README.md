# GGUF block dissection

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Ankush-Chander/DS635-ml-system-engineering?devcontainer_path=.devcontainer/gguf-block/devcontainer.json)

**The affine quantizer as llama.cpp ships it.** `Q4_0` is symmetric, `Q4_1` is
asymmetric, and both store 32 weights per block with FP16 metadata.

## Two formats

| | `Q4_0` | `Q4_1` |
| --- | --- | --- |
| Stored per block | `d`, 16 bytes of nibbles | `d`, `m`, 16 bytes of nibbles |
| Decode | `r = d * (q - 8)` | `r = d * q + m` |
| As `r = S * (q - Z)` | `S = d`, `Z = 8` | `S = d`, `Z = -m / d` |

## Read first

The Codespace checks out the three llama.cpp files at tag `b10758` into `llama.cpp/`:

| File | Line | What |
| --- | ---: | --- |
| `ggml/src/ggml-common.h` | 194 | `block_q4_0` |
| | 201 | `block_q4_1` |
| | 258 | `block_q8_1`, with `s = d * sum(qs)` |
| `ggml/src/ggml-quants.c` | 113 | `quantize_row_q4_0_ref` |
| | 150 | `quantize_row_q4_1_ref` |
| | 459 | `dequantize_row_q4_0` |
| | 479 | `dequantize_row_q4_1` |
| `ggml/src/ggml-cpu/quants.c` | 262 | `ggml_vec_dot_q4_1_q8_1_generic` |

## Run

```bash
pytest -q              # self-checks, grouped by TODO
python gguf_block.py   # small demo
```

Fill in `gguf_block.py` in order: **A** block sizes and bits per weight,
**B** unpack nibbles, **C** pack nibbles, **D** decode `Q4_0`, **E** decode
`Q4_1`, **F** encode `Q4_1` byte-for-byte. **G** (stretch) splits the dot
product into its core and offset terms. Each test name starts with the TODO
it checks; tests D–F compare against `gguf`, llama.cpp's own Python package.

## Think about

- Byte `j` holds weight `j` and weight `j + 16`, not weights `2j` and `2j + 1`.
  What does that layout let a SIMD kernel do with one 16-byte load?
- TODO F computes `q` with the float32 `d` but stores `d` as float16. How large
  is the decode error that this mismatch adds, next to the rounding error?
- `Q4_1`'s `Z = -m / d` is not an integer, so real 0.0 has no exact code. Why
  is that acceptable for weights, when activations need an exact zero?
- `quantize_row_q4_0_ref` sets `d = max / -8`, where `max` is the signed value
  of largest magnitude. When is `d` negative, and which extra value does it buy?
