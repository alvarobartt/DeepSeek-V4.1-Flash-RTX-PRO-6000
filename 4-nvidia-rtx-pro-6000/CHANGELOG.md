# 4 x NVIDIA RTX PRO 6000

The patch is cumulative against `vllm/vllm-openai:v0.32.0`; every working milestone contains its complete recipe.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 9. Direct dense CUDA kernels

- Use direct FP8 kernels for small QKV and shared-expert projections to reduce projection latency.
- Use BF16 register MMA for five through eight WO-A rows while retaining the existing smaller-row kernel and original weights.
- Limit BLAS and compiler thread pools to one to reproduce the matched measurements and avoid thread exhaustion.
- Retain the preceding serve arguments, full context, and scheduler capacity to isolate the kernel changes.

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
patch --batch --forward -p1 -d "$vllm_package_dir/.." < 4-nvidia-rtx-pro-6000/diff.patch
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
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC --cudart shared -rdc=true -DHOST_RANKS=4 -DPACKET_LANES=16 \
    -gencode arch=compute_120a,code=sm_120a \
    "$vllm_package_dir/distributed/device_communicators/sm120_host_all_reduce.cu" \
    -o /usr/local/lib/libvllm-sm120-host-ll.so
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC --cudart shared \
    -gencode arch=compute_120a,code=sm_120a \
    "$vllm_package_dir/models/deepseek_v4/nvidia/ops/sm120_wo_a_mma.cu" \
    -o /usr/local/lib/libvllm-sm120-g2-wo-a-mma.so
export NCCL_TUNER_PLUGIN=/usr/local/lib/libvllm-sm120-tuner.so
export VLLM_SM120_EXPERT_TP=4
export VLLM_SM120_COMPACT_EXPERTS=1
export VLLM_SM120_TARGET_GREEDY=1
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC --cudart shared \
    -gencode arch=compute_120a,code=sm_120a \
    "$vllm_package_dir/models/deepseek_v4/nvidia/ops/sm120_wo_a_register.cu" \
    -o /usr/local/lib/libvllm-sm120-wo-a-register.so
export VLLM_SM120_DIRECT_DENSE=1
export VLLM_SM120_WOA_REGISTER=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
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
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"revision":"1bae0e1088993b9b82de08f59de5be5a34e42170","enable_adaptive_verification":true,"use_local_argmax_reduction":true}' \
    --gpu-memory-utilization 0.97 \
    --max-num-batched-tokens 4096 \
    --compilation-config '{"max_cudagraph_capture_size":768,"cudagraph_capture_sizes":[1,2,4,6,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,528,544,560,576,592,608,624,640,656,672,688,704,720,736,752,768]}'
```

| Concurrency | Prompts | Output t/s | p95 TTFT (ms) | p95 TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 231.3 | 118.5 | 5.25 |
| 4 | 128 | 534.4 | 100.0 | 9.20 |
| 8 | 128 | 752.1 | 153.3 | 13.45 |
| 16 | 128 | 1018.1 | 279.7 | 19.43 |
| 32 | 128 | 1363.1 | 677.1 | 28.03 |
| 64 | 128 | 1666.3 | 854.7 | 46.81 |
| 128 | 256 | 2241.0 | 1732.4 | 84.52 |
| 256 | 512 | 2882.4 | 3298.9 | 121.63 |

The full 1,048,576-token context and 1,024 scheduler slots are retained, with 3,115,530 KV tokens.

All 160 correctness requests and six API checks pass, and all sweep throughput and latency percentiles were independently recomputed.

The Docker recipe builds and reproduces the measured source hashes; GPU benchmarks used matching source overlays on the base image rather than the built image itself.

Against the matched control, c1 improves by 5.2%, c32 and c64 throughput change by -0.3% and -0.5%, and c4 p99 TPOT rises from 10.0 to 12.0 ms; reversed cache-matched c32 repeats show comparable median tails.

KV capacity falls from 3,116,608 to 3,115,530 tokens while retaining the full context and scheduler limits.

## Accepted progress

| Stage | Change | c1 t/s | Revision |
|---:|---|---:|---|
| 0 | Four-GPU baseline | 68.9 | [`98e3fb6`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/98e3fb632bb37dea066d2f7d9cb6f1b4a436cb4d) |
| 1 | DSpark decoding | 131.3 | [`dee02d3`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/dee02d3567a50684bf4abb462d28390b3c3c9460) |
| 2 | Ninety-six local experts | 161.5 | [`26ae9ab`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/26ae9abd68d901a24ec36c582ae2c354a34eb81c) |
| 3 | Four-rank host collectives | 173.8 | [`630b1eb`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/630b1eb24926b2c70072dbe1d0c3a8af4adb4226) |
| 4 | Grouped BF16 projections | 178.5 | [`253839f`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/253839f710753836599f93bdcca65e822f0ac618) |
| 5 | Aligned TP2 and EP2 experts | 192.2 | [`d0fbce8`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/d0fbce80bfdab5115723369ed99b7ee5881d2c99) |
| 6 | Five-draft TP4 experts | 213.9 | [`1520928`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/1520928d7a23dbef2399128cb1ec9286d971d93e) |
| 7 | Cooperative collectives through 64 rows | 216.9 | [`c00f229`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/c00f229e9578cea673e1367190ef2b4315b1ed2a) |
| 8 | Paired GPU collectives | 224.0 | [`1122ce0`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/1122ce0fc4b811d1e1c22dd96d2b760f17c25901) |
| 9 | Direct dense CUDA kernels | 231.3 | This revision |
