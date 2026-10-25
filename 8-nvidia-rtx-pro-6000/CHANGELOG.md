# 8 x NVIDIA RTX PRO 6000

The patch is cumulative against `vllm/vllm-openai:v0.32.0`; every working milestone contains its complete recipe.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 9. Bounded candidate masking

- Mask only valid logits during sm120 decode in diff.patch because top-k already ignores columns beyond each request's context.

Requirements, on a fresh vLLM image:

```bash
python3 -m pip install --no-deps --upgrade flashinfer-python==0.7.1
python3 -m pip install --no-deps --upgrade flashinfer-cubin==0.7.1 --index-url https://flashinfer.ai/whl
python3 -m pip install --no-deps --upgrade flashinfer-jit-cache==0.7.1 flashinfer-jit-cache-sm120f==0.7.1 --index-url https://flashinfer.ai/whl/cu130
vllm_package_dir=$(python3 -c 'import importlib.util; print(importlib.util.find_spec("vllm").submodule_search_locations[0])')
patch --batch --forward -p1 -d "$vllm_package_dir/.." < 8-nvidia-rtx-pro-6000/diff.patch
mkdir -p /tmp/nccl-tuner/tuner \
export NCCL_TUNER_PLUGIN=/usr/local/lib/libvllm-sm120-tuner.so
export NCCL_TUNER_PLUGIN=/usr/local/lib/libvllm-sm120-tuner.so
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
    --block-size 64 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":3,"revision":"1bae0e1088993b9b82de08f59de5be5a34e42170","enable_adaptive_verification":true}' \
    --enable-expert-parallel
```

| Concurrency | Prompts | Output t/s | p95 TTFT (ms) | p95 TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 152.0 | 129.8 | 7.88 |
| 4 | 128 | 378.7 | 113.4 | 12.74 |
| 8 | 128 | 582.4 | 214.0 | 15.74 |
| 16 | 128 | 831.4 | 275.5 | 22.44 |
| 32 | 128 | 1139.8 | 388.6 | 30.54 |
| 64 | 128 | 1470.3 | 801.5 | 43.60 |
| 128 | 256 | 2097.0 | 1354.1 | 72.29 |
| 256 | 512 | 2820.1 | 2783.5 | 139.30 |

## Accepted progress

| Stage | Change | c1 t/s | Revision |
|---:|---|---:|---|
| 1 | sm120 support | 85.0 | [`0cd46b2`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/0cd46b25086a61436eb0979abf115f66442e889e) |
| 2 | DSpark decoding | 159.2 | [`e5ac344`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/e5ac344074d1e9141c50e3178cde4e2a3680da8b) |
| 3 | Three-token drafts | 155.8 | [`992d4ea`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/992d4eac7bd5f3aab0f2e62a012719471ffffb18) |
| 4 | Adaptive verification | 150.8 | [`aa424d6`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/aa424d635c4004b144357dbd9518c91f112f8602) |
| 5 | Expert parallelism | 138.0 | [`c5aada3`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/c5aada30e6d6be4d72714d1bad1446d6e35abd00) |
| 6 | Indexer scratch memory | 138.4 | [`891d19f`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/891d19f0991391ec7d65bf11ca467aef21ffa5c4) |
| 7 | Small-message all-reduce | 151.5 | [`36eef82`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/36eef8206664ec7fe510d0b24f52296a87a950b9) |
| 8 | Native FP8 cache | 151.8 | [`a9e6ae8`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/a9e6ae8dd71ddca1d6a113493d382719633b3a5c) |
| 9 | Bounded candidate masking | 152.0 | This revision |
