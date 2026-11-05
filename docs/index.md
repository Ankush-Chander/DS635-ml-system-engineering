# DS635 - Machine Learning System Engineering | DAU

Welcome to the course on Machine Learning System Engineering.

---

## Lectures

### Module 1 — Traditional Software Systems vs ML Systems

- [0: Course Overview](lectures/Lecture0.md) — ML systems vs traditional software, evolution of AI, the triad, The Bitter Lesson
- [1: Machine Learning Lifecycle](lectures/Lecture1.md) — lifecycle, engineering practices, Waymo case study, core challenges

### Module 2 — GPU Fundamentals & Hardware-Software Stack

- [2: Why Accelerators Exist](lectures/Lecture2.md) — the CPU baseline, the memory wall, what the matmul workload demands, design the hardware yourself
- [3_4: Memory Hierarchy & Roofline](lectures/Lecture3_4.md) — the matmul ladder from naive Python to GPU, SIMD, tiling, threads, arithmetic intensity, the roofline model, two GPU timing traps
- [5_6: GPU Job Submission](lectures/Lecture5_6.md) — PCIe, MMIO and BARs, DMA, ring/doorbell/fence, why an unsynchronized kernel launch measures nothing  
- [7: Matmul to Silicon — Execution & Latency Hiding](lectures/Lecture7.md) (in three parts: [7a](lectures/Lecture7a.md), [7b](lectures/Lecture7b.md), [7c](lectures/Lecture7c.md)) — SMs, warps and SIMT, a CUDA core is not a core, latency hiding by oversubscription, why the register file is bigger than L1, divergence and occupancy
- [8: Inside the GPU — Memory & the Roofline](lectures/Lecture8.md) — the memory hierarchy, shared memory is not a cache, coalescing and the working-set cliff, building both roofline axes from `rocminfo`, why LLM decode is memory-bound and training compute-bound

### Module 3 — Inference & Model Optimization

- [9_10: Inference Optimization](lectures/Lecture9_10.md) — server + model + hardware + workload, online vs batch vs embedding, TTFT/TPOT/MFU/MBU, prefill compute-bound vs decode memory-bound, predicting token rate on paper
- [11_12: Model Artifacts](lectures/Lecture11_12.md) — a live model object flattened to disk and back: pickle, safetensors, GGUF, ONNX; strides and contiguity; what survives the trip and what the reader must supply
- [13_14: Decoding Strategies](lectures/Lecture13_14.md) — autoregressive loop, greedy search, beam search, temperature, top-k vs nucleus, choosing a strategy per task
- [15_16: Speculative Decoding](lectures/Lecture15_16.md) — draft model, verification as a prefill, what it does and does not promise, the mathematical guarantee, speculation across decoding strategies
- [17_18: Speculative Decoding Variants](lectures/Lecture17_18.md) — three axes: where the draft comes from, what shape the candidates take, how verification is done; choosing a variant
- [19_20: Number Formats](lectures/Lecture19_20.md) — range vs precision, fixed point, IEEE 754 floating point, FP64 down to FP4, INT8 and INT4
- [21_22: Affine Quantization](lectures/Lecture21_22.md) — scale and zero-point, why real zero must land on an integer, per-tensor vs per-channel vs per-group, dissecting a GGUF block
- [23_24: Quantization Schemes & Targets](lectures/Lecture23_24.md) — symmetric vs asymmetric, static vs dynamic, weight-only vs weights and activations, matching the target to the bottleneck
- [27_28: Attention & KV Cache](lectures/Lecture27_28.md) — attention as content-based routing, the n² cost, sizing the cache from `config.json`, capacity vs traffic, GQA/MQA, MLA, sliding-window and hybrid layers, KV-cache quantization
- [29: Serving](lectures/Lecture29.md) — FlashAttention: tiling and online softmax; static vs dynamic vs continuous batching; chunked prefill

## Labs

- [Lab 5_6: GPU Job Submission](labs/Lab5_6.md) — measure the ring/doorbell/fence protocol yourself, on Colab or your own GPU ([notebook](labs/Lab5_6_gpu_job_submission.ipynb))
- [Lab 2: The Concurrency Budget](labs/Lab2.md) — derive the bytes your GPU must keep in flight, then find where it stops scaling; needs CUDA or ROCm ([notebook](labs/Lab2_concurrency_budget.ipynb))
- [Lab 11_12: Three Formats, One Model](labs/Lab11_12.md) — save one model as `.pt`, `.safetensors` and `.gguf` and measure what each flattening costs; CPU-only ([notebook](labs/Lab11_12_model_artifacts.ipynb))
- [Lab 13_14: Decoding Strategies, Measured](labs/Lab13_14.md) — measure the next-token distribution and what greedy, temperature, top-k/top-p and beam each do to it; CPU-only ([notebook](labs/Lab13_14_decoding_strategies.ipynb))
- [Lab 15_16: Speculative Decoding, Measured](labs/Lab15_16.md) — implement the accept/reject rule on dummy data, then a real gpt2 ← distilgpt2 stitch; CPU-only ([notebook](labs/Lab15_16_speculative_decoding.ipynb))
- [Lab 19–22a: Fixed Point & Affine, by Hand](labs/Lab19_22a.md) — build `FixedPoint` and `AffineQuantizer`, marked by a hidden test suite; runs in Codespaces
- [Lab 19–22b: Quantization Error, Measured](labs/Lab19_22b.md) — MSE vs `S² / 12`, SNR per bit, the cost of one outlier, a digit classifier's accuracy vs bits; CPU-only ([notebook](labs/Lab19_22b_quantization_error.ipynb))

## Notebooks

- [GGUF, the long way around](notebooks/gguf_the_long_way_around.ipynb) — Lecture 11_12: from a two-parameter PyTorch model to the byte layout of a real GGUF file
- [Decoding strategies, live](notebooks/decoding_strategies.ipynb) — Lecture 13_14: every strategy on the same model and prompt
- [KV cache, measured](notebooks/kv_cache_shapes.ipynb) — Lecture 27_28: generation with and without the cache, shapes and bytes against the formula

---
