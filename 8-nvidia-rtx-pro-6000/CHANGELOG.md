# 8 x NVIDIA RTX PRO 6000

The patch is cumulative against `vllm/vllm-openai:v0.32.0`; every working milestone contains its complete recipe.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 20. Compact two-way experts

- Split each expert across two GPUs and four expert groups without padding to reduce small-batch expert work.
- Pass route counts and scale strides at runtime to avoid compiling kernels for each request length.
- Reduce the prefill budget to 4,096 tokens to limit prefill stalls while retaining the full 1,048,576-token context.
- Raise GPU memory utilization to 96.5% to preserve KV-cache capacity with the new expert layout.
-    --enable-expert-parallel

Requirements, on a fresh vLLM image:

```bash
python3 -m pip install --no-deps --upgrade \
    flashinfer-python==0.7.1
python3 -m pip install --no-deps --upgrade \
    flashinfer-cubin==0.7.1 \
    --index-url https://flashinfer.ai/whl
python3 -m pip install --no-deps --upgrade \
    flashinfer-jit-cache==0.7.1 \
    flashinfer-jit-cache-sm120f==0.7.1 \
    --index-url https://flashinfer.ai/whl/cu130
vllm_package_dir=$(python3 -c 'import importlib.util; print(importlib.util.find_spec("vllm").submodule_search_locations[0])')
patch --batch --forward -p1 -d "$vllm_package_dir/.." < 8-nvidia-rtx-pro-6000/diff.patch
nccl_headers=https://raw.githubusercontent.com/NVIDIA/nccl/73cf112295c33aee2b895f329f592f2a9b4b0f97/src/include
mkdir -p /tmp/nccl-tuner/tuner
curl -fL "$nccl_headers/plugin/nccl_tuner.h" -o /tmp/nccl-tuner/nccl_tuner.h
curl -fL "$nccl_headers/nccl_common.h" -o /tmp/nccl-tuner/nccl_common.h
for nccl_api in 2 3 4 5 6; do
    curl -fL "$nccl_headers/plugin/tuner/tuner_v${nccl_api}.h" -o "/tmp/nccl-tuner/tuner/tuner_v${nccl_api}.h"
done
g++ -std=c++17 -O3 -shared -fPIC -Wall -Wextra -Werror \
    -I/tmp/nccl-tuner \
    -I/usr/local/cuda/include \
    -I"$vllm_package_dir/../nvidia/nccl/include" \
    "$vllm_package_dir/distributed/nccl_tuner.cc" \
    -o /usr/local/lib/libvllm-sm120-tuner.so
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC --cudart shared \
    -gencode arch=compute_120a,code=sm_120a \
    "$vllm_package_dir/distributed/device_communicators/sm120_host_all_reduce.cu" \
    -o /usr/local/lib/libvllm-sm120-host-ll.so
export NCCL_TUNER_PLUGIN=/usr/local/lib/libvllm-sm120-tuner.so
export VLLM_SM120_EXPERT_TP=2
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
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 3, "revision": "1bae0e1088993b9b82de08f59de5be5a34e42170", "enable_adaptive_verification": true, "use_local_argmax_reduction": true}' \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.965 \
    --max-num-batched-tokens 4096
```

| Concurrency | Prompts | Output t/s | p95 TTFT (ms) | p95 TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 228.9 | 131.3 | 5.22 |
| 4 | 128 | 544.2 | 87.8 | 8.43 |
| 8 | 128 | 786.3 | 212.0 | 12.20 |
| 16 | 128 | 1069.8 | 281.7 | 17.54 |
| 32 | 128 | 1428.1 | 385.2 | 24.90 |
| 64 | 128 | 1812.7 | 826.7 | 49.51 |
| 128 | 256 | 2379.8 | 1592.0 | 73.78 |
| 256 | 512 | 3113.4 | 3045.9 | 122.08 |

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
| 14 | Grouped expert GEMMs | 209.0 | [`3f93e68`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/3f93e6858c94c6eee506cd5331e23bde65f6e38b) |
| 15 | Wider grouped experts | 209.9 | [`a2b480a`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/a2b480a3526a3a235f7113b630650516733f658f) |
| 16 | Sharded draft sampling | 211.3 | [`e36008c`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/e36008c9f6a7cd5c1cb9f55e8082ede7341164e5) |
| 17 | Medium grouped experts | 211.7 | [`ba418b9`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/ba418b932c779f5224cc2657e52a769112cf0917) |
| 18 | Cold-start guards | 209.3 | [`62b3bf7`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/62b3bf717ce8447cd20c62b06a2bfbe781b6836d) |
| 19 | Counter-safe collectives | 211.4 | [`b4be2dd`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/b4be2ddcfaf547f6f042d91aac6a104dfabae3a2) |
| 20 | Compact two-way experts | 228.9 | This revision |
