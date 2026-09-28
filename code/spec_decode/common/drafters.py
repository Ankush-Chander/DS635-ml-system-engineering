"""Drafters: propose k tokens, cheaply. One interface, two rungs of the ladder.

    draft(context, k) -> (tokens, q)

`q` is the draft distribution at each proposed position, `[k, vocab]` — what the
accept/reject test divides by. A drafter that proposes deterministically returns
a one-hot `q`, which is a real distribution and makes the sampled loop below
work unchanged: min(1, p/1) = p(x), and the residual norm(max(0, p - e_x))
is `p` with `x` removed.
"""
from __future__ import annotations

import torch

from blackbox import KVCache, LM, probs_from


class PromptLookupDrafter:
    """n-gram drafter. No weights, no second model, no GPU.

    Find the most recent earlier occurrence of the last `n` tokens and copy
    whatever followed it. Strong on code and on any text that repeats itself,
    useless on genuinely novel prose — which is the point: it isolates the
    speculative loop from the cost of a neural drafter.
    """

    name = "prompt-lookup"

    def __init__(self, vocab_size: int, device: str = "cpu", max_ngram: int = 3, min_ngram: int = 1) -> None:
        self.vocab_size = vocab_size
        self.device = device
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self.passes = 0  # a drafter with no model does no forward passes

    def reset(self) -> None:
        pass

    def draft(self, context: list[int], k: int) -> tuple[list[int], torch.Tensor | None]:
        for n in range(self.max_ngram, self.min_ngram - 1, -1):
            if len(context) < n + 1:
                continue
            pattern = context[-n:]
            for start in range(len(context) - n - 1, -1, -1):
                if context[start : start + n] == pattern:
                    guess = context[start + n : start + n + k]
                    if guess:
                        return guess, self._one_hot(guess)
        return [], None

    def _one_hot(self, tokens: list[int]) -> torch.Tensor:
        q = torch.zeros(len(tokens), self.vocab_size, device=self.device)
        q[torch.arange(len(tokens)), torch.tensor(tokens)] = 1.0
        return q


class ModelDrafter:
    """A smaller model from the same family. k forward passes per block.

    Must share a tokenizer with the target: the accept/reject test compares
    p(x) and q(x) for the same integer `x`, so the two models must mean the same
    thing by it.
    """

    def __init__(self, lm: LM, temperature: float = 1.0, greedy: bool = False) -> None:
        self.lm = lm
        self.temperature = temperature
        self.greedy = greedy
        self.cache = KVCache()
        self.name = f"model({lm.name})"
        self._valid = 0  # cached positions whose keys match the committed sequence

    @property
    def passes(self) -> int:
        return self.lm.passes

    def reset(self) -> None:
        self.cache.reset()
        self._valid = 0
        self.lm.reset_counters()

    def draft(self, context: list[int], k: int) -> tuple[list[int], torch.Tensor | None]:
        # Draft tokens from the previous block sit past `_valid` and may have been
        # rejected, so they are dropped before the committed tail is fed.
        self.cache.truncate(self._valid)
        logits = self.lm.forward(context[self._valid :], self.cache)
        self._valid = len(context)

        tokens: list[int] = []
        rows: list[torch.Tensor] = []
        last = logits[-1]
        for i in range(k):
            q = probs_from(last, self.temperature)
            token = int(q.argmax()) if self.greedy else int(torch.multinomial(q, 1))
            tokens.append(token)
            rows.append(q)
            if i < k - 1:
                last = self.lm.forward([token], self.cache)[-1]
        return tokens, torch.stack(rows)
