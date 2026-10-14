"""Self-checks, grouped by TODO. Run `pytest -q`; work top to bottom."""

from fixed_point import FixedPoint as FP


# TODO A
def test_a_scale():
    assert FP(0.0).scale == 16


# TODO B
def test_b_range_q3_4():
    f = FP(0.0)
    assert (f.min_int, f.max_int) == (-128, 127)


def test_b_range_q1_6():
    f = FP(0.0, int_bits=1, frac_bits=6)
    assert (f.min_int, f.max_int) == (-128, 127)


# Encoding
def test_encode_exact():
    assert FP(1.5).raw == 24


def test_encode_rounds_to_nearest():
    assert FP(0.1).raw == 2  # 0.1 * 16 = 1.6


def test_int_is_raw():
    assert FP(5).raw == 5


def test_repr():
    assert repr(FP(1.5)) == "1.5 (Raw: 24)"


# TODO C
def test_c_saturate_high():
    assert FP(100.0).raw == 127


def test_c_saturate_low():
    assert FP(-100.0).raw == -128


# Addition
def test_add():
    assert (FP(1.5) + FP(2.25)).raw == 60


def test_add_saturates():
    assert (FP(7.0) + FP(7.0)).raw == 127


def test_add_keeps_format():
    s = FP(1.5, int_bits=5, frac_bits=2) + FP(10.0, int_bits=5, frac_bits=2)
    assert (s.int_bits, s.n, s.raw) == (5, 2, 46)


# TODO D-F
def test_mul():
    assert (FP(1.5) * FP(2.0)).raw == 48


def test_mul_negative():
    assert (FP(-1.5) * FP(2.0)).raw == -48


def test_mul_rounds_half_up():
    assert (FP(0.0625) * FP(0.5)).raw == 1  # exactly half a step


def test_mul_saturates():
    assert (FP(4.0) * FP(4.0)).raw == 127
