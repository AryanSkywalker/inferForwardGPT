# forwardGPT2

GPT-2 (124M) inference in C++. No ML frameworks. Learning project.

The idea is to build the full forward pass from scratch — weight loading, a tensor class, the math ops, attention, tokenizer, generation. Each phase is written by hand before moving on.

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
