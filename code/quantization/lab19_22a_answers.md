# Lab 19–22a — Answers

Copy this file into your Codespace, rename it `answers.md`, and fill in your roll number and name.
Write **3–6 sentences** under each question, in place of *(your answer here)*. Leave the `##` headings
and the `>` question lines as they are: the marking script reads your answers by heading.

Each answer is marked on two things: naming the **mechanism** (saturation, rounding bias, range against
step, the zero-point), and pointing to **your code** (a TODO, a line) or to **a number you printed**.
Questions are worth 40 marks in total; the heaviest are **Q2** and **Q6**.

**Roll**: your roll number
**Name**: your name

---

## Q1 — Saturation (5 marks)

> In `fixed_point.py`, `FixedPoint(7.0) + FixedPoint(7.0)` prints `7.9375`. Which TODO produces that
> number, and what raw values go in and come out? What would a plain int8 addition return instead,
> and why is saturating the safer failure for a model's activations?

*(your answer here)*

## Q2 — Rounding bias (8 marks)

> `FixedPoint(0.0625) * FixedPoint(0.5)` gives raw 1, but `FixedPoint(-0.0625) * FixedPoint(0.5)` gives
> raw 0. Trace both through TODOs D–F: the wide product, the bias, the shift. Why are the ties
> asymmetric, and what does that do to the error across a million-term dot product
> (Lecture 19/20 §4)?

*(your answer here)*

## Q3 — Range against step (5 marks)

> Q1.6 and Q3.4 are both 8 bits. Give the step and the range of each, computed with your TODO A and
> TODO B. Which would you pick for weights in [-1, 1], and what do you give up?

*(your answer here)*

## Q4 — Ties up against ties to even (5 marks)

> `FixedPoint` rounds ties **up**; `np.round` in your TODO E rounds them **to even**. Quantize
> `0.03125` and `0.09375` with a signed int8 `AffineQuantizer` calibrated on `[-8.0, 7.9375]` and
> report the integers. Which rule had the smaller error in Lecture 19/20 §4, and why do two
> frameworks that differ only here drift apart on the same model?

*(your answer here)*

## Q5 — Stretching the range to include zero (5 marks)

> Calibrate an unsigned 8-bit quantizer on `[2.0, 5.0]`. Report the scale and zero-point it chooses.
> How many of the 256 integers sit below real 2.0 and are never used? What would break if TODO B
> were skipped?

*(your answer here)*

## Q6 — Why zero must be exact (8 marks)

> A ReLU output is exactly 0.0, and a convolution pads its input with the integer `Z`. Why must
> `dequantize(Z)` be exactly 0.0, not 0.004? Point to the lines in TODO B and TODO D that guarantee
> it, and say what error a padded border would pick up if they did not (Lecture 21/22 §2).

*(your answer here)*

## Q7 — The four-term dot product (4 marks)

> Expand `sum((q_w - z_w) * (q_x - z_x))` into the four terms of TODO G. Which can be computed once
> from the weights, and which must be recomputed for every input? Set `z_w = 0`: which two terms
> vanish, and why is that the reason weights are usually quantized symmetrically (Lecture 21/22 §4)?
> Answer from the algebra even if you did not attempt TODO G.

*(your answer here)*
