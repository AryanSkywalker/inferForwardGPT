# forwardGPT2

GPT-2 (124M) inference in C++. No ML frameworks. Learning project.

The idea is to build the full forward pass from scratch — weight loading, a tensor class, the math ops, attention, tokenizer, generation. Each phase is written by hand before moving on.

## progress

- **phase 1** — weight dump script pulls GPT-2 small from HuggingFace and writes each tensor as a raw float32 binary. C++ reads a manifest and loads all 148 tensors into an unordered_map. 124,439,808 params.
- **phase 2** — Tensor class. owns or views a float buffer. rule of five, view(), row-major at() accessors.
- **phase 3** — core ops: matmul, softmax, layer norm, gelu. in progress.
- **phase 4** — attention + MLP blocks. todo.
- **phase 5** — BPE tokenizer. todo.
- **phase 6** — full forward pass + generation loop. todo.
- **phase 7** — performance: OpenMP, AVX2 SIMD, compare against OpenBLAS. todo.

## build

needs the `infer` conda env for the weight dump.

```bash
# dump weights once
conda activate infer
python scripts/dump_weights.py

# build
mkdir -p build && cd build
cmake ..
make

# run from repo root
./build/gpt2
```

## structure

```
include/   headers
src/       implementation
scripts/   weight dump script
weights/   gitignored, populated by dump script
```
