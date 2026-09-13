# Model artifacts

> **Module thesis:** a model in memory is a live object — arrays with a shape, a layout and a dtype, plus the code that gives them meaning. Persisting or shipping it means flattening all of that into a byte stream and rebuilding it later. Every format below is a different answer to **what survives the trip to disk, and what the reader must already have to put the model back together.**

Lectures 9/10 covered how fast a model runs. These notes are about what a model **is** when it is not running, and why that shape was forced on it.

**The arc:** `.pt` → safetensors → GGUF → ONNX. Each format exists because the previous one failed a systems constraint — not because someone preferred it.

---

## 1. What is inside a `.pt`?

Before it is a file, a model is a **live object in memory**: a `state_dict` — an ordered map from names to tensors — where each tensor is a raw block of numbers plus the **shape, stride and dtype** that make those numbers mean something. Saving is the act of turning that object into a byte stream; loading is rebuilding it.

Open a `.pt` — PyTorch's default — and you see the first attempt at that flattening. It is **a ZIP archive**, holding two different kinds of thing:

| Entry | What it is |
| --- | --- |
| `data.pkl` | the **recipe** — a pickle stream describing how to rebuild the object |
| `data/0`, `data/1`, … | the **raw tensor storages**, flat bytes |

**That split is the first design decision that matters.** Bulk bytes are kept *out* of the pickle stream so they can be read without interpreting it — every later format takes the idea further.

---

## 2. Pickle — a recipe, not data

**A pickle is a program.** It is a stack machine whose opcodes build objects, and two of those opcodes are the entire story:

- **`GLOBAL` / `STACK_GLOBAL`** — push a callable *named* by import path
- **`REDUCE`** — call it

### Consequence 1: classes travel by name, never by value

The file records `__main__.Recipe`. It does not record what `Recipe` is — so unpickling resolves that name **in the loading process**. Paste this into a Python terminal:

```python
import pickle
class Recipe: pass

blob = pickle.dumps(Recipe())   # the name "__main__.Recipe" goes into the bytes, not the class
# del Recipe                      # stand in for loading where the class is undefined
pickle.loads(blob)              # AttributeError: Can't get attribute 'Recipe' on <module '__main__'>
```

The bytes are intact; the **name** had nowhere to resolve. A checkpoint is therefore **not self-contained** — it depends on the class layout of the environment that wrote it. Rename a class between versions, or refactor which module it lives in, and old checkpoints stop loading.

### Consequence 2: loading *is* execution

`__reduce__` lets an object say *"to rebuild me, call this with these arguments."* `pickle.loads` obeys unconditionally. No sandbox, no confirmation.

!!! question "💬 Where is the vulnerability — in a function you call, or in the load itself?"

    ??? hint "In the load"
        Nothing in your code has to invoke anything. Deserialising the stream *is* the invocation. This is why downloading a checkpoint from an untrusted source is equivalent to running a script from one.

### The mitigation, and the shape of it

`torch.load(weights_only=True)` — the **default since PyTorch 2.6** — restricts which globals may be resolved.

**It is an allow-list, not a sandbox.** It works by naming what is permitted, so its completeness is load-bearing. That is a materially weaker guarantee than *having no mechanism at all*, which is where §3 goes.

!!! tip "Notebook — GGUF, the long way around · §§1–4"
    Open it and dissect a `.pt` yourself: read its opcodes, then watch a payload fire on load and get blocked by `weights_only=True`.

    [**Open in Colab**](https://colab.research.google.com/github/Ankush-Chander/DS635-ml-system-engineering/blob/main/docs/notebooks/gguf_the_long_way_around.ipynb) · [view on GitHub](https://github.com/Ankush-Chander/DS635-ml-system-engineering/blob/main/docs/notebooks/gguf_the_long_way_around.ipynb)

---

## 3. safetensors — data, not a program

**Remove the mechanism and the vulnerability cannot exist.** safetensors stores a flat `string → tensor` map in three regions:

```
[0:8]      u64 little-endian  = header length
[8:8+n]    UTF-8 JSON header, padded to an 8-byte boundary
[8+n:]     contiguous raw tensor bytes
```

A per-tensor header entry contains exactly three things — **`dtype`, `shape`, `data_offsets`**. No opcodes, no type registry, no callable names anywhere in the file.

### What the layout buys

Because every tensor's byte range is known from the header alone, a loader can `mmap` the file and hand out views **without a deserialisation pass**:

- **Lazy loading** — mapping a 10 GB file reserves *virtual* address space; pages fault in on demand
- **Zero-copy** — bytes stay in the OS page cache; no user-space duplicate is built
- **Cost scales with what you read**, not with file size

!!! question "💬 A 20 GB model, a machine with 8 GB of RAM. Pickle versus mmap?"

    ??? hint
        Pickle materialises the whole object graph in RAM and is killed by the OOM reaper. `mmap` maps the file and faults in only the pages actually touched — the working set, not the file.

        **This is what made safetensors viable as the default for multi-hundred-GB repos.**

### What it refuses, and why that is the design

| Attempted | Result |
| --- | --- |
| `{"a": tensor, "epoch": 5}` | rejected — not a tensor |
| nested dict (optimizer state) | rejected |
| two keys sharing one storage | rejected — a byte range cannot express identity |
| a non-contiguous view (`.T`) | rejected — **see §4** |

It holds **weights, not training state**. Optimizer state and counters need a second file.

---

## 4. Memory layout — what the formats quietly mandate

**Hardware memory is one line of addresses.** It has no concept of a grid. Every N-dimensional tensor is a **shape plus a stride tuple** laid over a flat buffer.

| Order | A 2×3 matrix becomes | Used by |
| --- | --- | --- |
| **Row-major** (C) | `[1,2,3,4,5,6]` | PyTorch, `llama.cpp`, NumPy |
| Column-major (Fortran) | `[1,4,2,5,3,6]` | BLAS lineage, Julia, R |

**Strides** say how far to jump per dimension. Row-major 2×3 → `(3, 1)`: three columns, so changing row skips 3.

### The zero-copy transpose

Swap the strides from `(3,1)` to `(1,3)` and you have a transposed tensor. **Zero bytes moved.** The result is a *non-contiguous view* — same buffer, different reading rules.

!!! question "💬 Neither safetensors nor GGUF store strides. Is that an omission?"

    ??? hint "No — it is a guarantee"
        Both formats **mandate** that a tensor's bytes are tightly packed row-major. Because that promise is unconditional, strides are **derivable** from shape and dtype and never need serialising. For `shape = [2,3,4]` the implied strides are `[12,4,1]`.

        Arbitrary strides are a *live-memory* optimisation. Serialization trades that flexibility for one canonical layout.

        This is the module thesis at its sharpest: the stride is part of the live object and **does not survive the trip**. The file keeps the shape and discards the layout freedom, and the reader rebuilds a fresh, canonical layout on load.

### Who pays

Saving a non-contiguous view forces **`.contiguous()`** — allocate a fresh block, physically reshuffle, then write. That is why safetensors refuses the view rather than silently copying.

> **Write once, read millions of times.** A model is saved once on a machine with terabytes of RAM. It is loaded millions of times on laptops and phones. The cost belongs on the writer.

This is also why layout is a *performance* property and not bookkeeping: a GPU fetches contiguous blocks in one transaction (Lecture 8), so reading along stride 1 is one fetch and reading across a large stride is many.

!!! tip "Notebook — Memory layout and loading"
    Measure it yourself: a transpose that moves zero bytes, the real cost of `.contiguous()`, the header as plain JSON, and a 134 MB file loading for 0.5 MB of resident memory.

    [**Open in Colab**](https://colab.research.google.com/github/Ankush-Chander/DS635-ml-system-engineering/blob/main/code/artifacts/memory_layout_and_loading.ipynb) · [view on GitHub](https://github.com/Ankush-Chander/DS635-ml-system-engineering/blob/main/code/artifacts/memory_layout_and_loading.ipynb)

---

## 5. GGUF — one file, no Python

safetensors stores **only** `string → tensor`. It assumes an external Python library already knows the layer names, the dimensions, and how to tokenize.

> The Python script builds the skeleton; safetensors supplies the meat.

**Remove Python and that assumption collapses.**

!!! question "💬 A compiled C++ binary, no Python, one file. What must be in that file besides the weights?"

    ??? hint "Two things"
        1. **Architecture hyperparameters** — layer count, attention heads, hidden dimension, context length. Without them there is no computation graph to assemble.
        2. **Tokenizer vocabulary and rules** — without them text cannot become numbers. In the Python ecosystem this lives in a separate `tokenizer.json`.

GGUF packs both into a key-value metadata block ahead of the tensor data.

| | safetensors | GGUF |
| --- | --- | --- |
| Weights | ✅ | ✅ |
| Architecture | ✗ runtime supplies | ✅ in-file |
| Tokenizer | ✗ separate file | ✅ in-file |
| Quantized types | bolted on | **first-class** |
| `mmap` / zero-copy | ✅ | ✅ |
| Stores strides | ✗ | ✗ |

**GGUF is safetensors plus the assumptions safetensors left to Python.** Same data-not-program bargain, same contiguity guarantee — it just refuses to depend on a runtime that already knows the model.

!!! tip "Notebook — GGUF, the long way around · §§6, 9–11"
    Build them from the spec yourself: safetensors, then a GGUF writer and reader, then parse a real model file and read its `Q8_0` block.

    [**Open in Colab**](https://colab.research.google.com/github/Ankush-Chander/DS635-ml-system-engineering/blob/main/docs/notebooks/gguf_the_long_way_around.ipynb) · [view on GitHub](https://github.com/Ankush-Chander/DS635-ml-system-engineering/blob/main/docs/notebooks/gguf_the_long_way_around.ipynb)

---

## 6. ONNX — the model as a program

Every format so far stores **numbers**. ONNX stores the **computation**.

An ONNX file is a **directed acyclic graph** serialized as Protocol Buffers:

| Element | Holds |
| --- | --- |
| **Nodes** | standardized operators — `Gemm`, `Conv`, `Softmax` |
| **Edges** | tensors flowing between them, with dtype and shape |
| **Initializers** | the constant weights |

**Versioned opsets** are the portability contract: an operator's meaning is pinned to a version, so the graph means the same thing to every conforming runtime.

### What the graph buys

!!! question "💬 A compiler can see the whole graph ahead of time. What can it do that a weights-only runtime cannot?"

    ??? hint "Fuse"
        Matmul → bias add → activation normally makes **three** round trips through VRAM: write the intermediate, read it back, write again, read again.

        A compiler that sees all three collapses them into **one kernel**, keeping intermediates in registers.

        **Why it pays so much:** those steps are memory-bound, not compute-bound (Lecture 9/10, §4). The saving is eliminated traffic, not arithmetic.

### The cost

Protocol Buffers enforce a **hard 2 GB per-message limit**. Modern weights blow straight through it, so they move to **external data files** — which fixes the size problem and **destroys the single-file property** that makes GGUF attractive for distribution.

---

## 7. The split that organises all of it

| | Model as **data** | Model as **program** |
| --- | --- | --- |
| Formats | safetensors, GGUF | ONNX *(and pickle, accidentally)* |
| Runtime must supply | the architecture | nothing |
| Optimiser can see | nothing | the whole graph |
| Code execution on load | impossible by construction | schema-checked |
| Single file at scale | ✅ | ✗ external data |

**Pickle is on the wrong side of this table by accident.** It stores a program because it was built to serialize *any* Python object, not because anyone wanted a model to be executable. ONNX is on that side deliberately, and gets an optimising compiler in return.

---

## What a format cannot do for you

Three limits, because "we use safetensors" is often over-read:

1. **The weights may be adversarial.** Backdoored parameters are still just numbers; no format has an opinion on what they compute.
2. **The surrounding repo still runs code.** `trust_remote_code=True` executes custom modelling code regardless of weight format.
3. **Format is not provenance.** Knowing a file *cannot* execute code is orthogonal to knowing it is the file the publisher intended. That is a hashing and signing question.

---

## References

1. [safetensors — Simple, safe way to store and distribute tensors](https://github.com/safetensors/safetensors#yet-another-format)
2. [GGUF, the long way around](https://vickiboykis.com/2024/02/28/gguf-the-long-way-around) — Vicki Boykis
3. [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
4. [ONNX — Open Neural Network Exchange](https://onnx.ai/onnx/intro/)
5. [`pickle` — Python object serialization](https://docs.python.org/3/library/pickle.html) · [`pickletools`](https://docs.python.org/3/library/pickletools.html)
6. [PyTorch 2.6 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.6.0) — `weights_only=True` becomes the default
