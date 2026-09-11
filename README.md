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
- [Lecture 7 — Inside the GPU: Execution & Latency Hiding](docs/lectures/Lecture7.md): How a GPU keeps thousands of ALUs busy · SMs, warps and SIMT · a CUDA core is not a core · latency hiding by oversubscription · why the register file is bigger than L1 · divergence and occupancy
- [Lecture 8 — Inside the GPU: Memory & the Roofline](docs/lectures/Lecture8.md): What limits GPU performance · the memory hierarchy · shared memory is not a cache · coalescing and the working-set cliff · building both roofline axes from `rocminfo` · why LLM decode is memory-bound and training compute-bound on the same GPU
- [Lecture 9_10 — Inference Optimization](docs/lectures/Lecture9_10.md): Server + model + hardware + workload · online vs batch vs embedding workloads · TTFT, TPOT, MFU and MBU mapped onto them · why prefill is compute-bound and decode memory-bound · GPU-only vs CPU offloading vs CPU-only · predicting token rate on paper · a purchase decision worked end to end
- [Lecture 11_12 — Model Artifacts](docs/lectures/Lecture11_12.md): What is inside a `.pt` · pickle as a program, not a document · safetensors and safety by removing the mechanism · strides, contiguity and who pays for packing · GGUF for runtimes with no Python · ONNX and the model as a graph · what a format cannot do for you

### Labs

- [Lab 5/6 — GPU Job Submission](docs/labs/Lab5_6.md) ([notebook](docs/labs/Lab5_6_gpu_job_submission.ipynb)): Measure the submission protocol on your own GPU or a free Colab T4 · submission vs execution (and the throughput number that beats the hardware) · per-launch and per-fence cost · pinned memory · stream overlap · CUDA Graphs. Marked 40 automatic + 60 rubric; see [`code/gpu_submission/grade_submissions.py`](code/gpu_submission/grade_submissions.py)
- [Lab 2 — The Concurrency Budget](docs/labs/Lab2.md) ([notebook](docs/labs/Lab2_concurrency_budget.ipynb)): Pairs with Lecture 7 and shares no experiment with it · memory latency by pointer chase · achieved bandwidth from a grid-stride copy · Little's Law to predict the bytes a GPU must keep in flight · the block-scheduling quantum read off a wave-quantisation staircase · the tail effect. Needs CUDA or ROCm (Triton has no Metal backend). Marked 35 automatic + 65 rubric; see [`code/gpu_internals/grade_lab2.py`](code/gpu_internals/grade_lab2.py)



## Notebooks

- [GGUF, the long way around](docs/notebooks/gguf_the_long_way_around.ipynb) — *Module 4.* From a two-parameter PyTorch model to the byte layout of a real model file: `state_dict` → pickle (and its exploit) → safetensors → checkpoints → GGML → GGUF. Implements safetensors and GGUF readers/writers from their specs, then parses a real quantised model and derives its bits-per-weight. After Vicki Boykis, [*GGUF, the long way around*](https://vickiboykis.com/2024/02/28/gguf-the-long-way-around/).

## Running the site locally

```bash
pip install -r requirements.txt
mkdocs serve
```

## References

- Vijay Janapa Reddi, [*Machine Learning Systems*](https://mlsysbook.ai)
- Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022)
