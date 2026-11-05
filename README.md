# DS635 — Machine Learning System Engineering | DAU

Course content for **DS635: Machine Learning System Engineering** at Dhirubhai Ambani University.

📖 Course site: [ds635.ankushchander.com](https://DS635.ankushchander.com)

## Lectures

### Module 1 — Traditional software systems vs ML systems

> In traditional software, behavior is specified; in ML systems, behavior is learned — and everything hard about ML systems follows from that.

- [Lecture 0 — Course Overview](docs/lectures/Lecture0.md): How ML systems differ from traditional software · evolution of AI paradigms · the data/algorithms/infrastructure triad · The Bitter Lesson
- [Lecture 1 — Machine Learning Lifecycle](docs/lectures/Lecture1.md): The ML lifecycle vs the traditional build–test–release loop · engineering practices (versioning, tracking, monitoring) · Waymo case study · core challenges

### Module 2 — GPU Fundamentals & Hardware-Software Stack
- [Lecture 2 — Why Accelerators Exist](docs/lectures/Lecture2.md): The CPU baseline · the memory wall · what the matmul workload demands · design the hardware yourself
- [Lecture 3_4 — The Memory Hierarchy and the Roofline](docs/lectures/Lecture3_4.md): The matmul ladder from naive Python to GPU · SIMD, tiling, threads · arithmetic intensity · the roofline model · two GPU timing traps
- [Lecture 5_6 — GPU Job Submission](docs/lectures/Lecture5_6.md): Who moves the bytes during a disk read · PCIe, posted vs non-posted · MMIO and BARs · DMA · ring, doorbell and fence · why an unsynchronized kernel launch measures nothing
- [Lecture 7 — Matmul to Silicon: Execution & Latency Hiding](docs/lectures/Lecture7.md) (in three parts: [7a](docs/lectures/Lecture7a.md), [7b](docs/lectures/Lecture7b.md), [7c](docs/lectures/Lecture7c.md)): How a GPU keeps thousands of ALUs busy · SMs, warps and SIMT · a CUDA core is not a core · latency hiding by oversubscription · why the register file is bigger than L1 · divergence and occupancy
- [Lecture 8 — Inside the GPU: Memory & the Roofline](docs/lectures/Lecture8.md): What limits GPU performance · the memory hierarchy · shared memory is not a cache · coalescing and the working-set cliff · building both roofline axes from `rocminfo` · why LLM decode is memory-bound and training compute-bound on the same GPU

### Module 3 — Inference & Model Optimization

- [Lecture 9_10 — Inference Optimization](docs/lectures/Lecture9_10.md): Server + model + hardware + workload · online vs batch vs embedding workloads · TTFT, TPOT, MFU and MBU mapped onto them · why prefill is compute-bound and decode memory-bound · GPU-only vs CPU offloading vs CPU-only · predicting token rate on paper · a purchase decision worked end to end
- [Lecture 11_12 — Model Artifacts](docs/lectures/Lecture11_12.md): What is inside a `.pt` · pickle as a program, not a document · safetensors and safety by removing the mechanism · strides, contiguity and who pays for packing · GGUF for runtimes with no Python · ONNX and the model as a graph · what a format cannot do for you
- [Lecture 13_14 — Decoding Strategies](docs/lectures/Lecture13_14.md): Transformer revision ending at the LM head · the autoregressive loop and why `select()` is outside the model · greedy search and the repetition trap · beam search and why likelihood is the wrong objective for open-ended text · temperature as exponential ratio reshaping · the long-tail failure of pure sampling · top-k vs nucleus (top-p) · choosing a strategy per task
- [Lecture 15_16 — Speculative Decoding](docs/lectures/Lecture15_16.md): Decoding is serial and memory-bound, verification parallel and compute-bound · draft decodes, target verifies, and verification is a prefill · what speculation promises and what it does not · acceptance as a chain and choosing `k` · the accept/reject rule and why the output distribution is unchanged · speculation under greedy, temperature, top-k/top-p and constrained decoding · a deployment checklist and failure modes
- [Lecture 17_18 — Variations of Speculative Decoding](docs/lectures/Lecture17_18.md): The design space as three independent axes · where the draft comes from: separate model, self-speculation, prompt and n-gram lookup, heads on the target, EAGLE, block drafters, cascades · what shape the candidates take: linear, tree, Jacobi · exact vs approximate verification · how much to speculate · choosing a variant by acquisition cost against expected acceptance
- [Lecture 19_20 — Number Formats](docs/lectures/Lecture19_20.md): Every format spends its bits on range and precision · fractional binary and fixed point · truncating vs rounding, and errors that accumulate · floating point and the IEEE 754 layout · normalized, subnormal and special values · casting and signed/unsigned traps · FP64 down to FP4, INT8 and INT4 compared · classroom exercise: [build `FixedPoint`](code/quantization/fixed_point/)
- [Lecture 21_22 — Affine Quantization](docs/lectures/Lecture21_22.md): The map $r = S(q - Z)$ and solving for scale and zero-point · why real zero must land on an integer · the cost of affine vs fixed point · per-tensor, per-channel and per-group granularity, and what one outlier does to each · classroom exercises: [build `AffineQuantizer`](code/quantization/affine/) and [dissect a llama.cpp `Q4_0`/`Q4_1` block](code/quantization/gguf_block/)
- [Lecture 23_24 — Quantization Schemes & Targets](docs/lectures/Lecture23_24.md): Symmetric vs asymmetric · static vs dynamic scales · weight-only (W8A16) vs weights and activations (W8A8) · weight-only attacks the bytes, W8A8 attacks the FLOPs, so match the target to the bottleneck
- [Lecture 27_28 — Attention and the KV Cache](docs/lectures/Lecture27_28.md): Attention as content-based routing · scaled dot-product scores and the causal mask · multi-head attention as a projection, not a slice · the n² score matrix · why caching keys and values is exact and queries are never cached · sizing the cache from `config.json` · capacity vs traffic · when the cache outweighs the weights, by context and by batch · attention variants that shrink it: GQA/MQA, MLA, sliding-window and hybrid linear-attention layers (DeepSeek Sparse Attention as an extra) · KV-cache quantization: dynamic scales, keys per-channel and values per-token, the streaming catch, FP8 in practice, TurboQuant
- [Lecture 29 — Serving](docs/lectures/Lecture29.md): Attention is limited by memory traffic, not FLOPs · FlashAttention: exact attention in tiles, with no score matrix in HBM · online softmax, worked by hand · where it pays: prefill and training, not decode · static vs dynamic vs continuous batching · chunked prefill
<!-- Restore when Lecture 29 §4-7 are un-commented: why reserving KV per request wastes memory · PagedAttention block tables · prefix sharing and preemption · reading the KV pool size and maximum concurrency off a vLLM startup log -->

## Labs

- [Lab 5/6 — GPU Job Submission](docs/labs/Lab5_6.md) ([notebook](docs/labs/Lab5_6_gpu_job_submission.ipynb)): Measure the submission protocol on your own GPU or a free Colab T4 · submission vs execution (and the throughput number that beats the hardware) · per-launch and per-fence cost · pinned memory · stream overlap · CUDA Graphs. Marked 40 automatic + 60 rubric; see [`code/gpu_submission/grade_submissions.py`](code/gpu_submission/grade_submissions.py)
- [Lab 2 — The Concurrency Budget](docs/labs/Lab2.md) ([notebook](docs/labs/Lab2_concurrency_budget.ipynb)): Pairs with Lecture 7 and shares no experiment with it · memory latency by pointer chase · achieved bandwidth from a grid-stride copy · Little's Law to predict the bytes a GPU must keep in flight · the block-scheduling quantum read off a wave-quantisation staircase · the tail effect. Needs CUDA or ROCm (Triton has no Metal backend). Marked 35 automatic + 65 rubric; see [`code/gpu_internals/grade_lab2.py`](code/gpu_internals/grade_lab2.py)
- [Lab 11/12 — Three Formats, One Model](docs/labs/Lab11_12.md) ([notebook](docs/labs/Lab11_12_model_artifacts.ipynb)): Pairs with Lecture 11/12 · save one model as `.pt`, `.safetensors` and `.gguf` and measure what each flattening costs · file size, cold vs warm load, RSS and `rchar` · the `.contiguous()` cost curve · the pickle trust boundary · read a real GGUF. CPU-only, no GPU needed. Marked 40 automatic + 60 rubric; see [`code/artifacts/grade_lab11_12.py`](code/artifacts/grade_lab11_12.py)
- [Lab 13/14 — Decoding Strategies, Measured](docs/labs/Lab13_14.md) ([notebook](docs/labs/Lab13_14_decoding_strategies.ipynb)): Pairs with Lecture 13/14 · measure the next-token distribution and what each strategy does to it · greedy determinism and argmax · temperature's `P(top)` and sample diversity · top-k vs top-p nucleus size · beam vs greedy log-prob (and why beam can lose). CPU-only (`gpt2`), no GPU needed. Marked 40 automatic + 60 rubric; see [`code/decoding/grade_lab13_14.py`](code/decoding/grade_lab13_14.py)
- [Lab 15/16 — Speculative Decoding, Measured](docs/labs/Lab15_16.md) ([notebook](docs/labs/Lab15_16_speculative_decoding.ipynb)): Pairs with Lecture 15/16 · implement the accept/reject rule on dummy distributions, then a real **gpt2 ← distilgpt2 two-model stitch** · the residual and its silent bug (identical acceptance, wrong distribution) · vectorised verification · greedy-stitch identity + target-pass count. CPU-only (same-tokenizer pair), no GPU needed. Marked 40 automatic + 60 rubric; see [`code/spec_decode/grade_lab15_16.py`](code/spec_decode/grade_lab15_16.py). The staged `make` version (Qwen2.5, KV-cache, `k`-sweep) lives in [`code/spec_decode/`](code/spec_decode/)
- [Lab 19–22a — Fixed Point and Affine Quantization, by Hand](docs/labs/Lab19_22a.md) (Codespaces): Pairs with Lectures 19/20 and 21/22 · fill in `FixedPoint` (scale, two's-complement range, saturation, addition, rounded multiply) and `AffineQuantizer` (grid, stretch to zero, scale, zero-point, quantize, dequantize, and the four-term dot product as a stretch) · `pytest` self-checks in the Codespace, marked by a hidden suite. Marked 60 automatic + 40 rubric; see [`code/quantization/grade_lab19_22a.py`](code/quantization/grade_lab19_22a.py)
- [Lab 19–22b — Quantization Error, Measured](docs/labs/Lab19_22b.md) ([notebook](docs/labs/Lab19_22b_quantization_error.ipynb)): Pairs with Lectures 19/20 and 21/22 · quantize a sine and Gaussian noise at 1–8 bits with the affine map · MSE against `S² / 12` and about 6 dB of SNR per bit · one outlier's cost, and blocks of 64 · quantize a small digit classifier's weights per-tensor and per-channel and find where accuracy collapses. CPU-only, numpy and scikit-learn, under a minute. Marked 40 automatic + 60 rubric; see [`code/quantization/grade_lab19_22b.py`](code/quantization/grade_lab19_22b.py)

## Notebooks

- [GGUF, the long way around](docs/notebooks/gguf_the_long_way_around.ipynb) — *Lecture 11/12.* From a two-parameter PyTorch model to the byte layout of a real model file: `state_dict` → pickle (and its exploit) → safetensors → checkpoints → GGML → GGUF. Implements safetensors and GGUF readers/writers from their specs, then parses a real quantised model and derives its bits-per-weight. After Vicki Boykis, [*GGUF, the long way around*](https://vickiboykis.com/2024/02/28/gguf-the-long-way-around/).
- [Decoding strategies, live](docs/notebooks/decoding_strategies.ipynb) — *Lecture 13/14.* Every strategy on the same model (`gpt2`) and prompt: greedy's repetition loop · beam search · temperature sweeps · top-k vs top-p, and how big the surviving set is at each step.
- [KV cache, measured](docs/notebooks/kv_cache_shapes.ipynb) — *Lecture 27/28.* Qwen3-0.6B on a free CPU runtime: generation with and without the cache (identical tokens; per-token time grows with the context only without it) · cache tensor shapes and bytes matched against the formula · when the cache outweighs the weights, by context length and by batch size.

## Demos

- [The acceptance rule](docs/viz/acceptance-rule.html) — *Lecture 15/16.* One proposal at a time: `q` proposes, the ratio test rules, the residual corrects.
- [Convergence bench](docs/viz/rejection-sampling.html) — *Lecture 15/16.* Run the rule a million times and watch the output settle onto `p`.

## Running the site locally

```bash
pip install -r requirements.txt
mkdocs serve
```

## References

- Vijay Janapa Reddi, [*Machine Learning Systems*](https://mlsysbook.ai)
- Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022)
