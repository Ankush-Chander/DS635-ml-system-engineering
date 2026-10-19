"""Affine quantization: fill in the TODOs, then run `pytest -q`.

FixedPoint stored real = raw / 2**F: a power-of-two scale, real 0.0 pinned to raw 0.
AffineQuantizer stores real = S * (q - Z): any positive scale S, any integer zero-point Z.
"""

import numpy as np


class AffineQuantizer:
    """Maps a real tensor onto the integer grid [q_min, q_max] with one (scale, zero_point) pair."""

    def __init__(self, bit_width=8, signed=False):
        # TODO A: the integer grid. Write every end with shifts, as in FixedPoint TODO B.
        # signed 8-bit: [-128, 127]    unsigned 8-bit: [0, 255]
        if signed:
            self.q_min = ...
            self.q_max = ...
        else:
            self.q_min = ...
            self.q_max = ...

        self.scale = None       # S, a positive float: the real value of one integer step
        self.zero_point = None  # Z, an int on the grid: the integer that means real 0.0

    def calibrate(self, tensor):
        r_min, r_max = tensor.min(), tensor.max()

        # TODO B: real 0.0 must sit inside [r_min, r_max] so it gets an exact integer.
        # Stretch the range to include it: [2.0, 5.0] becomes [0.0, 5.0].
        r_min = ...
        r_max = ...

        real_range = r_max - r_min
        if real_range == 0:  # an all-zero tensor: any scale works
            self.scale, self.zero_point = 1.0, 0
            return

        # TODO C: the real range spread evenly over the integer range.
        self.scale = ...

        # TODO D: solve r_min = S * (q_min - Z) for Z. Z lives on the grid, so round it
        # to a Python int, then clamp it into [q_min, q_max].
        self.zero_point = ...

    def quantize(self, r):
        # TODO E: real -> integer. Divide by the scale, shift by the zero-point,
        # round with np.round, clip into the grid. Return an integer array.
        ...

    def dequantize(self, q):
        # TODO F: integer -> real. Subtract the zero-point first, then scale.
        # Convert q to float32 before subtracting.
        ...


def affine_dot(q_w, z_w, q_x, z_x):
    """Integer dot product sum((q_w - z_w) * (q_x - z_x)), computed as four terms.

    TODO G (stretch): expand the product and return
        core + weight_sum + input_sum + offset
    where core = sum(q_w * q_x). Which terms depend only on the weights,
    and which one must be recomputed for every new input?
    Cast to np.int64 first so 8-bit products cannot overflow.
    """
    ...


if __name__ == "__main__":
    x = np.array([-1.0, 0.0, 0.5, 3.0])
    aq = AffineQuantizer()
    aq.calibrate(x)
    q = aq.quantize(x)
    print("scale      =", aq.scale)
    print("zero_point =", aq.zero_point)
    print("q          =", q)
    print("x_hat      =", aq.dequantize(q))

    q34 = AffineQuantizer(signed=True)
    q34.calibrate(np.array([-8.0, 7.9375]))
    print("Q3.4 as affine: S =", q34.scale, " Z =", q34.zero_point)
