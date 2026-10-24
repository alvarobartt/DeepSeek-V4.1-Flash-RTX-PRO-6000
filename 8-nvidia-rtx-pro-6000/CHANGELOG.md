# 8 x NVIDIA RTX PRO 6000

The patch is cumulative against `vllm/vllm-openai:v0.32.0`; every working milestone contains its complete recipe.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 1. sm120 support

- Upgrade the bundled FlashInfer Python, cubin, and sm120 cache packages from 0.7.0.post1 to 0.7.1 because the older version lacks runtime page-size support.
- Both diff.patch and `--block-size 64` are required because the patch enables 64-token pages in the sm120 backends and the flag selects them instead of the 128-token pages DeepGEMM rejects.
-    --mm-encoder-tp-mode data

Requirements, on a fresh vLLM image:

```bash
python3 -m pip install --no-deps --upgrade flashinfer-python==0.7.1
python3 -m pip install --no-deps --upgrade flashinfer-cubin==0.7.1 --index-url https://flashinfer.ai/whl
python3 -m pip install --no-deps --upgrade flashinfer-jit-cache==0.7.1 flashinfer-jit-cache-sm120f==0.7.1 --index-url https://flashinfer.ai/whl/cu130
vllm_package_dir=$(python3 -c 'import importlib.util; print(importlib.util.find_spec("vllm").submodule_search_locations[0])')
patch --batch --forward -p1 -d "$vllm_package_dir/.." < 8-nvidia-rtx-pro-6000/diff.patch
```

Then, run server as:

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --tokenizer-mode deepseek_v41 \
    --tensor-parallel-size 8 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --mm-encoder-tp-mode data \
    --block-size 64
```

| Concurrency | Prompts | Output t/s | p95 TTFT (ms) | p95 TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 85.0 | 128.5 | 11.49 |
| 4 | 128 | 253.2 | 93.9 | 15.64 |
| 8 | 128 | 375.4 | 215.1 | 20.46 |
| 16 | 128 | 553.1 | 287.9 | 25.91 |
| 32 | 128 | 657.4 | 4198.5 | 40.74 |
| 64 | 128 | 866.2 | 831.4 | 71.84 |
| 128 | 256 | 1242.4 | 1582.8 | 97.01 |
| 256 | 512 | 1814.1 | 2901.4 | 181.96 |

## Accepted progress

| Stage | Change | c1 t/s | Revision |
|---:|---|---:|---|
| 1 | sm120 support | 85.0 | This revision |
