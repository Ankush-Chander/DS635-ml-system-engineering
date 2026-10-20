"""GGUF block dissection: fill in the TODOs, then run `pytest -q`.

llama.cpp stores 4-bit weights in blocks of 32, each block with its own FP16 metadata:
    Q4_0   r = d * (q - 8)     symmetric:  S = d, Z = 8
    Q4_1   r = d * q + m       asymmetric: S = d, Z = -m / d
The C sources are checked out under llama.cpp/ggml/src/ (see README.md).
"""

import numpy as np

QK = 32  # weights per block


# TODO A: bytes per block, read off the structs in ggml-common.h.
# ggml_half is 2 bytes. Q4 packs two weights per byte; Q8_1 stores one int8 per weight.
BLOCK_BYTES = {
    "q4_0": ...,
    "q4_1": ...,
    "q8_1": ...,
}


def bits_per_weight(fmt):
    # TODO A: the bits of one block spread over its QK weights.
    ...


def unpack_nibbles(qs):
    """16 packed bytes -> 32 integers in [0, 15], in weight order."""
    # TODO B: byte j holds weight j in its LOW nibble and weight j + 16 in its HIGH nibble.
    # Return the 16 low nibbles followed by the 16 high nibbles, as uint8.
    ...


def pack_nibbles(q):
    """32 integers in [0, 15] -> 16 packed bytes. The inverse of unpack_nibbles."""
    # TODO C
    ...


def dequantize_q4_0(block):
    """One 18-byte Q4_0 block (a uint8 array) -> 32 float32 weights."""
    # TODO D: bytes 0-1 are d as float16 (`block[:2].view(np.float16)`); the rest is qs.
    # Mirror dequantize_row_q4_0: r = d * (q - 8), computed in float32.
    ...


def dequantize_q4_1(block):
    """One 20-byte Q4_1 block -> 32 float32 weights."""
    # TODO E: d, then m, each a float16; then qs. Mirror dequantize_row_q4_1: r = d * q + m.
    ...


def quantize_q4_1(x):
    """32 float32 weights -> one 20-byte Q4_1 block, byte-identical to quantize_row_q4_1_ref."""
    x = np.asarray(x, dtype=np.float32)
    # TODO F: d = (max - min) / 15 and m = min, both float32.
    # v = (x - min) * (1 / d); then q = trunc(v + 0.5), the C cast (int8_t)(v + 0.5f).
    # Cap q at 15. When d == 0, every q is 0.
    # q is computed with the float32 d; d and m become float16 only when stored.
    # Return d, m and qs as one uint8 array: np.concatenate of their `.view(np.uint8)`.
    ...


def dot_q4_1(block, a):
    """Dot product of one Q4_1 block with 32 float activations, split into two terms.

    TODO G (stretch): return (core, offset) with
        core   = d * sum(q * a)
        offset = m * sum(a)
    so that core + offset equals dequantize_q4_1(block) @ a.
    When a is itself quantized to block_q8_1, which term does its field `s` precompute?
    """
    ...


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.02, QK).astype(np.float32)
    block = quantize_q4_1(x)
    d, m = block[:4].view(np.float16)
    print("bytes      =", block.nbytes, " bits/weight =", bits_per_weight("q4_1"))
    print("d, m       =", float(d), float(m))
    print("q          =", unpack_nibbles(block[4:]))
    print("max |err|  =", np.abs(dequantize_q4_1(block) - x).max())
    a = rng.normal(size=QK).astype(np.float32)
    core, offset = dot_q4_1(block, a)
    print("core + off =", core + offset, " dequantize @ a =", dequantize_q4_1(block) @ a)
