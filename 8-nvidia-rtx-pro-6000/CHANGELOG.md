# 8 x NVIDIA RTX PRO 6000

The patch is cumulative against `vllm/vllm-openai:v0.32.0`; every working milestone contains its complete recipe.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 14. Grouped expert GEMMs

- Use the grouped sm120 GEMMs already provided by FlashInfer 0.7.1 in diff.patch for two-to-four-row expert batches because they avoid scalar dot-product loops.
- Cache transposed scales so each decode step can reuse FlashInfer’s scale layout.
- Keep the existing one-row kernel because grouped dispatch adds overhead for that batch size.

Requirements, on a fresh vLLM image:

```bash
python3 -m pip install --no-deps --upgrade flashinfer-python==0.7.1
python3 -m pip install --no-deps --upgrade flashinfer-cubin==0.7.1 --index-url https://flashinfer.ai/whl
python3 -m pip install --no-deps --upgrade flashinfer-jit-cache==0.7.1 flashinfer-jit-cache-sm120f==0.7.1 --index-url https://flashinfer.ai/whl/cu130
vllm_package_dir=$(python3 -c 'import importlib.util; print(importlib.util.find_spec("vllm").submodule_search_locations[0])')
patch --batch --forward -p1 -d "$vllm_package_dir/.." < 8-nvidia-rtx-pro-6000/diff.patch
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
| 1 | 128 | 209.0 | 129.4 | 5.88 |
| 4 | 128 | 422.1 | 113.0 | 11.60 |
| 8 | 128 | 598.6 | 213.6 | 16.86 |
| 16 | 128 | 849.2 | 276.5 | 22.39 |
| 32 | 128 | 1152.1 | 519.7 | 31.86 |
| 64 | 128 | 1476.6 | 827.4 | 44.20 |
| 128 | 256 | 2099.3 | 1414.5 | 72.53 |
| 256 | 512 | 2827.0 | 2794.2 | 140.71 |

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
| 9 | Bounded candidate masking | 152.0 | [`51061a6`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/51061a663497d98b907a35da5ddb76048eaacddb) |
| 10 | Packed FP4 experts | 167.8 | [`9253782`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/92537829d7ce003664a820e67d61e12b39680329) |
| 11 | Host all-reduce | 182.1 | [`f00f364`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/f00f36484aed613e5b1a86566f60fdf9b9e78284) |
| 12 | Host reduce-scatter | 194.2 | [`d6c2e43`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/d6c2e436e35434ee70c24ac354db8cbd22f86df8) |
| 13 | Wider host reductions | 194.5 | [`2a5cf80`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/2a5cf807aa60bccd4f3ced6a49b39ee1c5f25970) |
| 14 | Grouped expert GEMMs | 209.0 | This revision |
