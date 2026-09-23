"""
Stage 0 — Speculative decoding without a neural network.

Before we bring in a "draft model" and a "target model", let's isolate
the one idea that makes speculative decoding work.

We have two distributions over the same vocabulary:

    p = target model's distribution
    q = draft model's distribution

We want to SAMPLE FROM p, but we'd like to use q because it is cheaper.

The trick:

    1. Ask q for a token.
    2. Accept that token with probability min(1, p[x] / q[x]).
    3. If we reject it, sample from the RESIDUAL distribution:

           r(x) ∝ max(0, p(x) - q(x))

The surprising result:

    Even though we sometimes started with q,
    THE FINAL OUTPUT IS DISTRIBUTED EXACTLY AS p.

That's the core correctness argument behind speculative decoding.

We use only FIVE symbols here so that we can see the probability mass
directly instead of getting lost in a 50k-token vocabulary.

Run:
    python3 stage0_rejection.py
    python3 stage0_rejection.py --n 1000000
    python3 stage0_rejection.py --broken
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "common"))
from stats import chi_square_test  # noqa: E402


# ---------------------------------------------------------------------------
# The setup
# ---------------------------------------------------------------------------

# Think of P as the "truth": what the expensive target model says.
#
# Think of Q as a cheap draft model. It is reasonably good, but imperfect:
#
#   symbol 0:     gets it exactly right
#   symbol 1, 2:  underestimates them
#   symbol 3:     completely misses it
#   symbol 4:     is much too confident
#
# The question is:
#
#       Can we use Q to propose tokens,
#       while STILL producing samples exactly from P?
#
P = np.array([0.40, 0.25, 0.20, 0.10, 0.05])
Q = np.array([0.40, 0.15, 0.10, 0.00, 0.35])


# ---------------------------------------------------------------------------
# Step 1: What probability mass did Q "miss"?
# ---------------------------------------------------------------------------

def residual(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Return the probability distribution corresponding to:

        max(0, p - q)

    This is the probability mass that Q failed to account for.

    Example:

        p = [0.40, 0.25, 0.20, 0.10, 0.05]
        q = [0.40, 0.15, 0.10, 0.00, 0.35]

        max(p-q, 0)
          = [0.00, 0.10, 0.10, 0.10, 0.00]

    Notice what happened:

      - symbol 0: Q already has enough probability mass
      - symbol 1: Q is short by 0.10
      - symbol 2: Q is short by 0.10
      - symbol 3: Q is short by 0.10
      - symbol 4: Q has TOO MUCH, so there is no residual mass

    The residual distribution is therefore:

        [0, 1/3, 1/3, 1/3, 0]

    We only use this distribution when Q's proposed token is rejected.
    """

    # "What is left in P after taking away Q?"
    leftover = np.maximum(p - q, 0.0)

    # Turn the leftover MASS into a probability DISTRIBUTION.
    total = leftover.sum()

    if total == 0.0:
        # This case means q >= p for every symbol.
        # In that situation there is no probability mass left over.
        #
        # More importantly, rejection cannot actually happen, so this
        # fallback is unreachable during a normal speculative step.
        return p.copy()

    return leftover / total


# ---------------------------------------------------------------------------
# Step 2: One speculative decoding step
# ---------------------------------------------------------------------------

def draw(
    p: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
    broken: bool = False,
) -> tuple[int, bool]:
    """
    Perform ONE speculative sampling step.

    Conceptually:

                 cheap
                   │
                   ▼
              sample x ~ q
                   │
                   ▼
          accept with probability
              min(1, p[x]/q[x])
                /       \
             accept     reject
               │           │
               │           ▼
               │       sample from
               │       residual(p, q)
               │           │
               └─────┬─────┘
                     ▼
                  output

    The important point is that q is NOT the final distribution.

    q is merely used to propose a candidate cheaply.
    """

    # ---------------------------------------------------------------
    # 1. Let the cheap distribution propose a token.
    # ---------------------------------------------------------------

    x = rng.choice(len(q), p=q)

    # ---------------------------------------------------------------
    # 2. Decide whether to keep the proposal.
    #
    # The acceptance probability is:
    #
    #                  p(x)
    #       min(1,  -------- )
    #                  q(x)
    #
    # If p(x) >= q(x), accept with probability 1.
    #
    # If p(x) < q(x), accept only some of the time.
    #
    # This is the crucial correction that turns proposals from q
    # into samples from p.
    # ---------------------------------------------------------------

    if q[x] == 0:
        # q can never propose x if q[x] == 0, so this branch is
        # technically unreachable. Keeping the calculation explicit
        # makes the p/q rule easier to understand.
        acceptance_probability = 1.0
    else:
        acceptance_probability = min(1.0, p[x] / q[x])

    if rng.random() < acceptance_probability:
        return int(x), True

    # ---------------------------------------------------------------
    # 3. The proposal was rejected.
    #
    # This is where the magic happens.
    #
    # We DON'T simply sample again from p.
    #
    # Instead, we sample from:
    #
    #               max(0, p - q)
    #
    # because the rejection has already accounted for the part of
    # probability mass that q supplied.
    #
    # The residual supplies exactly what q was missing.
    # ---------------------------------------------------------------

    if broken:
        # Deliberately WRONG:
        #
        # "If the proposal failed, just sample from p."
        #
        # This looks reasonable, but it double-counts probability mass.
        # The statistical test below should catch this mistake.
        fallback = p
    else:
        fallback = residual(p, q)

    token = rng.choice(len(p), p=fallback)
    return int(token), False


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def main(n: int, seed: int, broken: bool) -> int:
    rng = np.random.default_rng(seed)

    counts = np.zeros(len(P), dtype=np.int64)
    accepted = 0

    for _ in range(n):
        token, was_accepted = draw(P, Q, rng, broken)

        counts[token] += 1
        accepted += was_accepted

    observed = counts / n

    # If speculative sampling is correct, the millions of outputs should
    # look as though we had simply sampled directly from P.
    statistic, dof, pvalue, _ = chi_square_test(counts, P)

    # -----------------------------------------------------------------------
    # Acceptance rate
    # -----------------------------------------------------------------------
    #
    # How often should q's proposal survive?
    #
    # The answer is:
    #
    #       sum_x min(p(x), q(x))
    #
    # which is also:
    #
    #       1 - TV(p, q)
    #
    # So the closer q is to p, the more often we can accept the cheap
    # draft token.
    #
    predicted_acceptance = float(np.minimum(P, Q).sum())
    observed_acceptance = accepted / n

    # -----------------------------------------------------------------------
    # Show what actually happened
    # -----------------------------------------------------------------------

    print("P = target distribution")
    print("Q = draft distribution")
    print()

    print(f"{'symbol':>7} {'P(target)':>12} {'observed':>12} {'error':>12}")

    for i, (target, got) in enumerate(zip(P, observed)):
        print(
            f"{i:>7} "
            f"{target:>12.4f} "
            f"{got:>12.4f} "
            f"{got - target:>+12.4f}"
        )

    print()

    print(
        f"chi-square      "
        f"stat={statistic:.3f}  "
        f"dof={dof}  "
        f"p={pvalue:.4f}"
    )

    print(
        f"acceptance      "
        f"observed={observed_acceptance:.4f}  "
        f"predicted={predicted_acceptance:.4f}"
    )

    # -----------------------------------------------------------------------
    # Check 1: Are our outputs actually distributed as P?
    # -----------------------------------------------------------------------

    distribution_ok = pvalue > 0.01

    # -----------------------------------------------------------------------
    # Check 2: Is the acceptance rate what probability theory predicts?
    #
    # The standard deviation of a Bernoulli proportion is:
    #
    #       sqrt(p(1-p)/n)
    #
    # We allow four standard deviations here, which is deliberately generous.
    # -----------------------------------------------------------------------

    standard_error = math.sqrt(
        predicted_acceptance
        * (1.0 - predicted_acceptance)
        / n
    )

    acceptance_ok = abs(
        observed_acceptance - predicted_acceptance
    ) < 4.0 * standard_error

    # -----------------------------------------------------------------------
    # The --broken mode
    # -----------------------------------------------------------------------

    if broken:
        print()
        print("--broken: rejection incorrectly resamples from P")
        print(
            f"detected        "
            f"{'PASS' if not distribution_ok else 'FAIL — test has no power'}"
        )

        # In broken mode, we WANT the distribution test to fail.
        return 0 if not distribution_ok else 1

    print()
    print(f"distribution    {'PASS' if distribution_ok else 'FAIL'}")
    print(f"acceptance rate {'PASS' if acceptance_ok else 'FAIL'}")

    return 0 if distribution_ok and acceptance_ok else 1


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=1_000_000,
        help="number of samples to draw",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )

    parser.add_argument(
        "--broken",
        action="store_true",
        help="use the deliberately incorrect rejection rule",
    )

    args = parser.parse_args()

    raise SystemExit(
        main(
            n=args.n,
            seed=args.seed,
            broken=args.broken,
        )
    )
