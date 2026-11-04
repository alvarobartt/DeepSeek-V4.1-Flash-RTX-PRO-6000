# 8 x NVIDIA RTX PRO 6000

The shared [patch](../diff.patch) is cumulative against `vllm/vllm-openai:v0.32.0` and selects kernels for TP4 and TP8 at runtime.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 23. Paired GPU collectives

- Skip publishing the local-owned unreduced chunk because its values are read directly from device memory.
- Exchange packets with the accessible paired GPU through CUDA IPC to reduce mapped-host traffic while preserving rank-ordered FP32 accumulation and BF16 rounding.
- Pass each communicator's GPU buffers as launch arguments so another communicator cannot replace its mapping.

Requirements: see the shared [Dockerfile](../Dockerfile).

```bash
export VLLM_SM120_EXPERT_TP=8
export VLLM_SM120_COMPACT_EXPERTS=1
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
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "revision": "1bae0e1088993b9b82de08f59de5be5a34e42170", "enable_adaptive_verification": true, "use_local_argmax_reduction": true}' \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.965 \
    --max-num-batched-tokens 3840 \
    --compilation-config '{"max_cudagraph_capture_size":768,"cudagraph_capture_sizes":[1,2,4,6,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,640,768]}'
```

| Concurrency | Prompts | Output t/s | p95 TTFT (ms) | p95 TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 284.6 | 122.1 | 4.54 |
| 4 | 128 | 686.2 | 93.8 | 7.60 |
| 8 | 128 | 892.9 | 132.8 | 10.75 |
| 16 | 128 | 1141.3 | 296.2 | 17.11 |
| 32 | 128 | 1532.2 | 674.3 | 23.78 |
| 64 | 128 | 1974.3 | 836.4 | 34.95 |
| 128 | 256 | 2545.0 | 1573.4 | 77.19 |
| 256 | 512 | 3179.3 | 3379.2 | 122.31 |

The full 1,048,576-token context and 1,024 scheduler slots are retained, with 24,425,790 KV tokens.

All 160 correctness requests and six API checks pass, and all sweep throughput and latency percentiles were independently recomputed.

The CUDA kernels are unchanged in the shared recipe; GPU measurements use source overlays on the base image, and the built image is separately checked for matching source hashes and compiled libraries.

Against the preceding accepted recipe, c32 p99 TTFT rises by 34.0% and p99 TPOT by 20.1%, while p95 end-to-end latency improves there by 2.9%.

The table records the linked accepted milestone, and the shared recipe also passes a fresh full concurrency sweep, 160 correctness requests, six API fallback checks, and original-weight checks on TP8.

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
| 20 | Compact two-way experts | 228.9 | [`cf53cb6`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/cf53cb67cf3f69c26acbbe9afd07ec428d766c60) |
| 21 | Target vocabulary-parallel sampling | 232.8 | [`7152d0f`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/7152d0f13f8aebd75b3f55638f26c270c16b4643) |
| 22 | Compact TP8 cooperative collectives | 280.5 | [`c00f229`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/c00f229e9578cea673e1367190ef2b4315b1ed2a) |
| 23 | Paired GPU collectives | 284.6 | [`1122ce0`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/1122ce0fc4b811d1e1c22dd96d2b760f17c25901) |
