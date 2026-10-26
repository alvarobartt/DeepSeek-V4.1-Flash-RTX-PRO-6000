# 4 x NVIDIA RTX PRO 6000

The patch is cumulative against `vllm/vllm-openai:v0.32.0`; every working milestone contains its complete recipe.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 0. Four-GPU baseline

- Reuse the eight-GPU track's cumulative sm120 patch and expert parallelism to run the checkpoint on four GPUs at its full context length.

Requirements, on a fresh vLLM image:

```bash
python3 -m pip install --no-deps --upgrade flashinfer-python==0.7.1
python3 -m pip install --no-deps --upgrade flashinfer-cubin==0.7.1 --index-url https://flashinfer.ai/whl
python3 -m pip install --no-deps --upgrade flashinfer-jit-cache==0.7.1 flashinfer-jit-cache-sm120f==0.7.1 --index-url https://flashinfer.ai/whl/cu130
vllm_package_dir=$(python3 -c 'import importlib.util; print(importlib.util.find_spec("vllm").submodule_search_locations[0])')
patch --batch --forward -p1 -d "$vllm_package_dir/.." < 4-nvidia-rtx-pro-6000/diff.patch
mkdir -p /tmp/nccl-tuner/tuner \
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC --cudart shared \
export NCCL_TUNER_PLUGIN=/usr/local/lib/libvllm-sm120-tuner.so
```

Then, run server as:

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --tokenizer-mode deepseek_v41 \
    --tensor-parallel-size 4 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --mm-encoder-tp-mode data \
    --block-size 64 \
    --enable-expert-parallel
```

| Concurrency | Prompts | Output t/s | p95 TTFT (ms) | p95 TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 68.9 | 134.4 | 14.24 |
| 4 | 128 | 216.8 | 103.5 | 18.08 |
| 8 | 128 | 317.3 | 227.0 | 23.89 |
| 16 | 128 | 425.8 | 308.0 | 33.99 |
| 32 | 128 | 553.8 | 4002.8 | 45.14 |
| 64 | 128 | 735.3 | 1037.5 | 82.04 |
| 128 | 256 | 1147.5 | 1674.5 | 97.09 |
| 256 | 512 | 1683.6 | 2808.1 | 184.56 |

## Accepted progress

| Stage | Change | c1 t/s | Revision |
|---:|---|---:|---|
| 0 | Four-GPU baseline | 68.9 | This revision |
