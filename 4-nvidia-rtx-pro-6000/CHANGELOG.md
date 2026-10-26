# 4 x NVIDIA RTX PRO 6000

The patch is cumulative against `vllm/vllm-openai:v0.32.0`; every working milestone contains its complete recipe.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 1. DSpark decoding

- Reuse three-token adaptive DSpark with local argmax reduction to verify multiple tokens per target pass without gathering the full vocabulary.
- Raise GPU memory utilization to 97% because DSpark's larger CUDA graph allocation leaves no KV-cache blocks at the default 92%.
-    --enable-expert-parallel

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
    --enable-expert-parallel \
    --speculative-config '{"method":"dspark","num_speculative_tokens":3,"revision":"1bae0e1088993b9b82de08f59de5be5a34e42170","enable_adaptive_verification":true,"use_local_argmax_reduction":true}' \
    --gpu-memory-utilization 0.97
```

| Concurrency | Prompts | Output t/s | p95 TTFT (ms) | p95 TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 131.3 | 134.1 | 9.31 |
| 4 | 128 | 305.4 | 139.3 | 14.95 |
| 8 | 128 | 450.0 | 232.0 | 20.57 |
| 16 | 128 | 646.7 | 306.9 | 29.31 |
| 32 | 128 | 929.3 | 433.6 | 36.76 |
| 64 | 128 | 1216.4 | 788.1 | 50.59 |
| 128 | 256 | 1779.2 | 1669.4 | 76.92 |
| 256 | 512 | 2090.1 | 5044.1 | 513.58 |

## Accepted progress

| Stage | Change | c1 t/s | Revision |
|---:|---|---:|---|
| 0 | Four-GPU baseline | 68.9 | [`98e3fb6`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/98e3fb632bb37dea066d2f7d9cb6f1b4a436cb4d) |
| 1 | DSpark decoding | 131.3 | This revision |
