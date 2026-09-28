"""The model as a black box: `(tokens, cache) -> logits`. Forward passes only.

Nothing below reaches inside the transformer. Speculative decoding needs three
things from a language model — logits at every fed position, a cache that can be
appended to, and a cache that can be rolled back — and this file is the whole
interface. Swapping in a transformer you wrote yourself means reimplementing
`LM.forward`; the sampling loops do not change.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


class KVCache:
    """Cached attention keys/values for a token prefix, with rollback.

    Rollback is what separates speculative decoding from ordinary generation:
    rejected draft tokens leave keys in the cache that no longer correspond to
    the committed sequence, so the sampler must be able to drop them.
    """

    def __init__(self) -> None:
        self.inner = DynamicCache()

    def __len__(self) -> int:
        return self.inner.get_seq_length()

    def truncate(self, n: int) -> None:
        """Keep the first `n` positions, drop the rest."""
        held = len(self)
        if n > held:
            raise ValueError(f"cannot truncate to {n}: cache holds {held}")
        if n == held:
            return
        self.inner.crop(-(held - n))  # negative removes that many; positive is the legacy "keep n"
        if len(self) != n:
            raise RuntimeError(f"cache truncate({n}) left {len(self)} positions")

    def reset(self) -> None:
        self.inner = DynamicCache()


class LM:
    """A causal LM reduced to logits, with forward passes counted."""

    def __init__(self, name: str, device: str = "cuda", dtype: str = "float32") -> None:
        self.name = name
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=getattr(torch, dtype)
        ).to(self.device)
        self.model.eval()
        self.passes = 0
        self.positions = 0

    @torch.inference_mode()
    def forward(self, tokens: list[int], cache: KVCache | None = None) -> torch.Tensor:
        """Logits at **every** fed position: `[len(tokens), vocab]`.

        With a cache, `tokens` are the positions not yet cached; the cache
        supplies the prefix and grows by `len(tokens)` on return.
        """
        if not tokens:
            raise ValueError("forward needs at least one token")
        self.passes += 1
        self.positions += len(tokens)
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        if cache is None:
            out = self.model(input_ids=ids, use_cache=False)
        else:
            out = self.model(input_ids=ids, past_key_values=cache.inner, use_cache=True)
            cache.inner = out.past_key_values
        return out.logits[0].float()

    def reset_counters(self) -> None:
        self.passes = 0
        self.positions = 0


def probs_from(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Target and draft distributions must be shaped the same way, or the
    accept/reject ratio compares two different models."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use the greedy loop instead")
    return torch.softmax(logits / temperature, dim=-1)
