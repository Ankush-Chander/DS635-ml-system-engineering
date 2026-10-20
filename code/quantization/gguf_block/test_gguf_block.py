"""Self-checks, grouped by TODO. Run `pytest -q`; work top to bottom.

The reference is `gguf`, llama.cpp's own Python package, which reproduces the C
quantize and dequantize routines byte for byte.
"""

import numpy as np
import pytest
from gguf import GGMLQuantizationType as QT
from gguf.quants import dequantize, quantize

from gguf_block import (
    BLOCK_BYTES,
    QK,
    bits_per_weight,
    dequantize_q4_0,
    dequantize_q4_1,
    dot_q4_1,
    pack_nibbles,
    quantize_q4_1,
    unpack_nibbles,
)

SEEDS = range(8)

# Byte j packs weight j (low) and weight j + 16 (high): weights read 0..15 then 15..0.
PATTERN_QS = (np.arange(16) | ((15 - np.arange(16)) << 4)).astype(np.uint8)
PATTERN_Q = np.concatenate([np.arange(16), 15 - np.arange(16)]).astype(np.uint8)


def weights(seed, outlier=False):
    x = np.random.default_rng(seed).normal(0.0, 0.02, QK).astype(np.float32)
    if outlier:
        x[seed % QK] = 0.5
    return x


def fp16_bytes(*values):
    return np.array(values, dtype=np.float16).view(np.uint8)


# TODO A
def test_a_block_bytes():
    assert BLOCK_BYTES == {"q4_0": 18, "q4_1": 20, "q8_1": 36}


@pytest.mark.parametrize("fmt, bpw", [("q4_0", 4.5), ("q4_1", 5.0), ("q8_1", 9.0)])
def test_a_bits_per_weight(fmt, bpw):
    assert bits_per_weight(fmt) == pytest.approx(bpw)


# TODO B
def test_b_unpack_layout():
    q = unpack_nibbles(PATTERN_QS)
    assert q.dtype == np.uint8
    np.testing.assert_array_equal(q, PATTERN_Q)


def test_b_low_nibble_comes_first():
    qs = np.zeros(16, dtype=np.uint8)
    qs[0] = 0x2A  # weight 0 = 0xA, weight 16 = 0x2
    q = unpack_nibbles(qs)
    assert (q[0], q[16], q[1]) == (10, 2, 0)


# TODO C
def test_c_pack_layout():
    np.testing.assert_array_equal(pack_nibbles(PATTERN_Q), PATTERN_QS)


@pytest.mark.parametrize("seed", SEEDS)
def test_c_round_trip(seed):
    q = np.random.default_rng(seed).integers(0, 16, QK).astype(np.uint8)
    np.testing.assert_array_equal(unpack_nibbles(pack_nibbles(q)), q)


# TODO D
def test_d_q4_0_by_hand():
    block = np.concatenate([fp16_bytes(0.5), PATTERN_QS])
    expected = 0.5 * (PATTERN_Q.astype(np.float32) - 8)
    np.testing.assert_allclose(dequantize_q4_0(block), expected)


def test_d_returns_float32():
    block = np.concatenate([fp16_bytes(0.5), PATTERN_QS])
    assert dequantize_q4_0(block).dtype == np.float32


@pytest.mark.parametrize("seed", SEEDS)
def test_d_q4_0_matches_llama_cpp(seed):
    block = quantize(weights(seed, outlier=True), QT.Q4_0)
    np.testing.assert_allclose(dequantize_q4_0(block), dequantize(block, QT.Q4_0), rtol=1e-6)


# TODO E
def test_e_q4_1_by_hand():
    block = np.concatenate([fp16_bytes(0.25, -1.0), PATTERN_QS])
    expected = 0.25 * PATTERN_Q.astype(np.float32) - 1.0
    np.testing.assert_allclose(dequantize_q4_1(block), expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_e_q4_1_matches_llama_cpp(seed):
    block = quantize(weights(seed, outlier=True), QT.Q4_1)
    np.testing.assert_allclose(dequantize_q4_1(block), dequantize(block, QT.Q4_1), rtol=1e-6)


# TODO F
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("outlier", [False, True])
def test_f_bytes_match_llama_cpp(seed, outlier):
    x = weights(seed, outlier)
    block = quantize_q4_1(x)
    assert block.dtype == np.uint8 and block.shape == (20,)
    np.testing.assert_array_equal(block, quantize(x, QT.Q4_1))


def test_f_constant_block():
    x = np.full(QK, 0.125, dtype=np.float32)
    np.testing.assert_array_equal(quantize_q4_1(x), quantize(x, QT.Q4_1))


def test_f_error_at_most_half_a_step():
    x = weights(3)
    d = (x.max() - x.min()) / 15
    err = np.abs(dequantize_q4_1(quantize_q4_1(x)) - x)
    assert err.max() <= d / 2 * 1.01  # 1% slack for the float16 d and m


# TODO G
@pytest.mark.parametrize("seed", SEEDS)
def test_g_terms_sum_to_dot(seed):
    block = quantize_q4_1(weights(seed, outlier=True))
    a = np.random.default_rng(100 + seed).normal(size=QK).astype(np.float32)
    core, offset = dot_q4_1(block, a)
    assert core + offset == pytest.approx(float(dequantize_q4_1(block) @ a), rel=1e-5, abs=1e-6)


def test_g_offset_is_min_times_activation_sum():
    block = np.concatenate([fp16_bytes(0.25, -1.0), PATTERN_QS])
    a = np.arange(QK, dtype=np.float32)
    _, offset = dot_q4_1(block, a)
    assert offset == pytest.approx(-1.0 * a.sum())
