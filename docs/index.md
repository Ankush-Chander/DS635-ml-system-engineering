# DS635 - Machine Learning System Engineering | DAU

Welcome to the course on Machine Learning System Engineering.

---

## Lectures

### Week 1 — From Code to Learned Behavior

- [0: Course Overview](lectures/Lecture0.md) — ML systems vs traditional software, evolution of AI, the triad, The Bitter Lesson
- [1: Machine Learning Lifecycle](lectures/Lecture1.md) — lifecycle, engineering practices, Waymo case study, core challenges

### Module 2 — GPU Fundamentals & Hardware-Software Stack

- [2: Why Accelerators Exist](lectures/Lecture2.md) — the CPU baseline, the memory wall, what the matmul workload demands, design the hardware yourself
- [3_4: Memory Hierarchy & Roofline](lectures/Lecture3_4.md) — the matmul ladder from naive Python to GPU, SIMD, tiling, threads, arithmetic intensity, the roofline model, two GPU timing traps
- [5_6: GPU Job Submission](lectures/Lecture5_6.md) — PCIe, MMIO and BARs, DMA, ring/doorbell/fence, why an unsynchronized kernel launch measures nothing  
- [7: Inside the GPU — Execution & Latency Hiding](lectures/Lecture7.md) (in three parts: [7a](lectures/Lecture7a.md), [7b](lectures/Lecture7b.md), [7c](lectures/Lecture7c.md)) — SMs, warps and SIMT, a CUDA core is not a core, latency hiding by oversubscription, why the register file is bigger than L1, divergence and occupancy
- [8: Inside the GPU — Memory & the Roofline](lectures/Lecture8.md) — the memory hierarchy, shared memory is not a cache, coalescing and the working-set cliff, building both roofline axes from `rocminfo`, why LLM decode is memory-bound and training compute-bound

### Module 3 — Inference & Model Optimization

- [9_10: Inference Optimization](lectures/Lecture9_10.md) — server + model + hardware + workload, online vs batch vs embedding, TTFT/TPOT/MFU/MBU, prefill compute-bound vs decode memory-bound, predicting token rate on paper
- [11_12: Model Artifacts](lectures/Lecture11_12.md) — a live model object flattened to disk and back: pickle, safetensors, GGUF, ONNX; strides and contiguity; what survives the trip and what the reader must supply

## Labs

- [Lab 5_6: GPU Job Submission](labs/Lab5_6.md) — measure the ring/doorbell/fence protocol yourself, on Colab or your own GPU ([notebook](labs/Lab5_6_gpu_job_submission.ipynb))
- [Lab 2: The Concurrency Budget](labs/Lab2.md) — derive the bytes your GPU must keep in flight, then find where it stops scaling; needs CUDA or ROCm ([notebook](labs/Lab2_concurrency_budget.ipynb))
- [Lab 11_12: Three Formats, One Model](labs/Lab11_12.md) — save one model as `.pt`, `.safetensors` and `.gguf` and measure what each flattening costs; CPU-only ([notebook](labs/Lab11_12_model_artifacts.ipynb))

---
