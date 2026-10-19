"""Self-checks, grouped by TODO. Run `pytest -q`; work top to bottom."""

import numpy as np

from affine_quantizer import AffineQuantizer as AQ
from affine_quantizer import affine_dot


def calibrated(lo, hi, **kw):
    aq = AQ(**kw)
    aq.calibrate(np.array([lo, hi]))
    return aq


# TODO A
def test_a_range_uint8():
    aq = AQ()
    assert (aq.q_min, aq.q_max) == (0, 255)


def test_a_range_int8():
    aq = AQ(signed=True)
    assert (aq.q_min, aq.q_max) == (-128, 127)


def test_a_range_uint4():
    aq = AQ(bit_width=4)
    assert (aq.q_min, aq.q_max) == (0, 15)


# TODO B
def test_b_range_stretched_to_zero():
    aq = calibrated(2.0, 5.0)  # behaves as [0.0, 5.0]
    assert np.isclose(aq.scale, 5.0 / 255)
    assert aq.zero_point == 0


def test_b_all_negative_range():
    aq = calibrated(-4.0, -1.0)  # behaves as [-4.0, 0.0]
    assert np.isclose(aq.scale, 4.0 / 255)
    assert aq.zero_point == 255


# TODO C
def test_c_scale_is_range_ratio():
    aq = calibrated(-1.0, 3.0)
    assert np.isclose(aq.scale, 4.0 / 255)


# TODO D
def test_d_zero_point_rounded():
    aq = calibrated(-1.0, 3.0)  # 0 - (-1) / (4/255) = 63.75
    assert aq.zero_point == 64


def test_d_zero_point_is_python_int():
    assert type(calibrated(-1.0, 3.0).zero_point) is int


def test_d_fixed_point_is_a_special_case():
    aq = calibrated(-8.0, 7.9375, signed=True)  # the Q3.4 range
    assert (aq.scale, aq.zero_point) == (0.0625, 0)


# TODO E
def test_e_quantize_ends_and_zero():
    aq = calibrated(-1.0, 3.0)
    q = aq.quantize(np.array([-1.0, 0.0, 3.0]))
    assert q.tolist() == [0, 64, 255]


def test_e_quantize_matches_fixed_point():
    aq = calibrated(-8.0, 7.9375, signed=True)
    assert aq.quantize(np.array([1.5, 0.1])).tolist() == [24, 2]  # FP(1.5), FP(0.1)


def test_e_quantize_clips():
    aq = calibrated(-1.0, 3.0)
    assert aq.quantize(np.array([-100.0, 100.0])).tolist() == [0, 255]


def test_e_ties_round_to_even():
    aq = calibrated(-8.0, 7.9375, signed=True)
    q = aq.quantize(np.array([0.03125, 0.09375]))  # 0.5 and 1.5 steps
    assert q.tolist() == [0, 2]


# TODO F
def test_f_zero_is_exact():
    aq = calibrated(-0.37, 1.91)
    assert aq.dequantize(aq.quantize(np.array([0.0]))).tolist() == [0.0]


def test_f_error_within_half_step():
    x = np.random.default_rng(0).uniform(-0.7, 2.3, 10_000)
    aq = AQ()
    aq.calibrate(x)
    err = np.abs(aq.dequantize(aq.quantize(x)) - x)
    assert err.max() <= aq.scale / 2 + 1e-6


def test_f_outlier_collapses_bins():
    x = np.append(np.linspace(0.0, 1.0, 1000), 100.0)
    aq = AQ()
    aq.calibrate(x)
    assert len(np.unique(aq.quantize(x[:1000]))) <= 4


# TODO G (stretch)
def test_g_dot_matches_direct():
    rng = np.random.default_rng(1)
    q_w = rng.integers(0, 256, 512)
    q_x = rng.integers(0, 256, 512)
    direct = int(np.sum((q_w - 7) * (q_x - 131)))
    assert affine_dot(q_w, 7, q_x, 131) == direct


def test_g_symmetric_weights():
    rng = np.random.default_rng(2)
    q_w = rng.integers(-127, 128, 512)
    q_x = rng.integers(0, 256, 512)
    two_terms = int(np.sum(q_w * q_x)) - 131 * int(np.sum(q_w))
    assert affine_dot(q_w, 0, q_x, 131) == two_terms
