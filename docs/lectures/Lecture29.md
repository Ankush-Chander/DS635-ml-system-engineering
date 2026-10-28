# Serving: continuous batching and PagedAttention

> **Module thesis:** [Lecture 27/28](Lecture27_28.md) sized the KV cache per user. A server holds many users at once, so its real capacity is the **KV pool**, and two mechanisms decide how much of that pool does useful work: the **scheduler** that chooses who runs each step, and the **allocator** that places each sequence's cache in memory.

**Prerequisites:** [Prefill vs decode and the memory wall](Lecture9_10.md) · [Attention and the KV cache](Lecture27_28.md)

---

!!! question "Before we start"
    An A6000 (48 GB) serves Llama-3-8B in BF16. About 25 GiB is left for the KV cache: **≈ 200K tokens, or ≈ 25 users at the full 8K window**. Why does a naive server fit far fewer?

## 1. Static batching wastes the GPU

- A static batch runs until its **longest** sequence finishes.
- Sequences that finish early leave idle slots; new requests wait for the whole batch.
- [Lecture 9/10](Lecture9_10.md) §1: online serving cannot wait for a full batch, so idle slots cost directly in occupancy.

## 2. Continuous batching

- Schedule **per decode step**, not per request: sequences join and leave the batch every iteration (Orca).
- One step can mix a new request's **prefill** with other requests' **decode**.
- **Chunked prefill** splits a long prompt across steps, so one large prompt does not stall every user's next token.

## 3. The memory problem it creates

- A request's output length is unknown when it is admitted.
- A server that reserves `max_len` of **contiguous** KV per request leaves most of that reservation empty, and fragments what is left.
- Measured across earlier serving systems, only **20–38% of KV memory held live tokens**.

## 4. PagedAttention

| Operating system | KV cache |
| --- | --- |
| page | **block** of 16 tokens' keys and values |
| page table | **block table** per sequence |
| allocate on touch | allocate a block when the previous one fills |

- Blocks need not be contiguous, so fragmentation disappears and waste falls **below 4%**.
- The attention kernel follows the block table to gather a sequence's keys and values.

## 5. Sharing and preemption

- **Prefix sharing:** requests with the same system prompt, or parallel samples of one prompt, point at the same blocks. A write copies the block first (copy-on-write).
- **Preemption:** when no free block remains, a sequence is evicted, then **recomputed** from its tokens or **swapped** to host memory.
- Preemption is the visible symptom of a pool sized below its load.

## 6. Read it off a real server

- vLLM reports its pool at startup: `GPU KV cache size` in tokens and `Maximum concurrency` at the configured length.
- Both numbers follow from [Lecture 27/28](Lecture27_28.md) §5.2: pool bytes $\div$ KV bytes per token.

!!! question "Check the server"
    From a startup log, recover the KV bytes per token and the maximum concurrency. Then predict both with FP8 KV.

---

## References

1. Yu et al., [*Orca: A Distributed Serving System for Transformer-Based Generative Models*](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022) — iteration-level scheduling
2. Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (SOSP 2023) — block tables, KV utilisation of earlier systems, copy-on-write sharing, preemption
3. Agrawal et al., [*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*](https://arxiv.org/abs/2403.02310) (OSDI 2024) — chunked prefill
4. Vivek Kalyanarangan, *Quantization and Fast Inference*, Manning Publications — §12.1 serving a four-bit model through vLLM
5. [Lecture 9/10](Lecture9_10.md) — batch vs online workloads
6. [Lecture 27/28](Lecture27_28.md) — sizing the KV cache
