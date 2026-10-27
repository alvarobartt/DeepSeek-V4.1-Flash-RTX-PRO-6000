# 4 x NVIDIA RTX PRO 6000

The patch is cumulative against `vllm/vllm-openai:v0.32.0`; every working milestone contains its complete recipe.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 4. Grouped BF16 projections

- Use a split-K tensor-core kernel for two-group BF16 attention projections to keep more GPU blocks active at small batch sizes.

Requirements, on a fresh vLLM image:

```bash
python3 -m pip install --no-deps --upgrade flashinfer-python==0.7.1
python3 -m pip install --no-deps --upgrade flashinfer-cubin==0.7.1 --index-url https://flashinfer.ai/whl
python3 -m pip install --no-deps --upgrade flashinfer-jit-cache==0.7.1 flashinfer-jit-cache-sm120f==0.7.1 --index-url https://flashinfer.ai/whl/cu130
vllm_package_dir=$(python3 -c 'import importlib.util; print(importlib.util.find_spec("vllm").submodule_search_locations[0])')
patch --batch --forward -p1 -d "$vllm_package_dir/.." < 4-nvidia-rtx-pro-6000/diff.patch
mkdir -p /tmp/nccl-tuner/tuner \
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC --cudart shared \
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
| 1 | 128 | 178.5 | 137.7 | 6.60 |
| 4 | 128 | 418.9 | 133.6 | 11.76 |
| 8 | 128 | 582.1 | 231.6 | 16.42 |
| 16 | 128 | 780.9 | 297.9 | 23.41 |
| 32 | 128 | 1117.8 | 428.0 | 32.00 |
| 64 | 128 | 1354.1 | 902.9 | 57.05 |
| 128 | 256 | 1822.6 | 1659.9 | 85.47 |
| 256 | 512 | 2483.4 | 3109.7 | 150.76 |

## Accepted progress

| Stage | Change | c1 t/s | Revision |
|---:|---|---:|---|
| 0 | Four-GPU baseline | 68.9 | [`98e3fb6`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/98e3fb632bb37dea066d2f7d9cb6f1b4a436cb4d) |
| 1 | DSpark decoding | 131.3 | [`dee02d3`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/dee02d3567a50684bf4abb462d28390b3c3c9460) |
| 2 | Ninety-six local experts | 161.5 | [`26ae9ab`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/26ae9abd68d901a24ec36c582ae2c354a34eb81c) |
| 3 | Four-rank host collectives | 173.8 | [`630b1eb`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/630b1eb24926b2c70072dbe1d0c3a8af4adb4226) |
| 4 | Grouped BF16 projections | 178.5 | This revision |
