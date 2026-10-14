"""Signed fixed-point numbers: fill in the TODOs, then run `pytest -q`."""


class FixedPoint:
    """Signed fixed-point number in Q<int_bits>.<frac_bits> format, plus 1 sign bit.

    Default Q3.4: 8 bits total, step = 1/16 = 0.0625, range [-8.0, 7.9375].
    The value is stored as an integer `raw`; the real value is raw / scale.
    """

    def __init__(self, value, int_bits=3, frac_bits=4):
        self.int_bits = int_bits
        self.n = frac_bits

        # TODO A: scale = 2**frac_bits. Compute it with a bit shift.
        self.scale = ...

        total_bits = int_bits + frac_bits + 1

        # TODO B: two's-complement range of a signed `total_bits`-bit integer.
        # For 8 bits that is [-128, 127]. Write both ends with shifts.
        self.min_int = ...
        self.max_int = ...

        if isinstance(value, float):
            # TODO: encode the real number: scale it, round to nearest, make it an int.
            self.raw = ...
        else:
            # An int is taken to be an already-encoded raw value.
            self.raw = value

        # TODO C: saturate raw into [min_int, max_int]. One line, no if-statements.
        self.raw = ...

    def __repr__(self):
        return f"{self.raw / self.scale} (Raw: {self.raw})"

    def __add__(self, other):
        # TODO: both operands share one scale. How do the raws combine?
        # Return a new FixedPoint from the raw sum, keeping int_bits and frac_bits.
        # Think: where does overflow get handled?
        ...

    def __mul__(self, other):
        # TODO D: multiply the raws. What scale does the product carry?
        wide_product = ...

        # TODO E: rounding bias = half of one output step, written as a shift.
        rounding_bias = ...

        # TODO F: bring the product back to `scale`: add the bias, shift right by n.
        final_raw = ...

        return FixedPoint(final_raw, self.int_bits, self.n)


if __name__ == "__main__":
    a, b = FixedPoint(1.5), FixedPoint(2.25)
    print("a     =", a)
    print("b     =", b)
    print("a + b =", a + b)
    print("a * b =", a * b)
    print("7 + 7 =", FixedPoint(7.0) + FixedPoint(7.0))
