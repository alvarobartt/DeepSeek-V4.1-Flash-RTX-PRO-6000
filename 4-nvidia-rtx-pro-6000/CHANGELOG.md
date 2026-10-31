# 4 x NVIDIA RTX PRO 6000

The patch is cumulative against `vllm/vllm-openai:v0.32.0`; every working milestone contains its complete recipe.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 5. Aligned TP2 and EP2 experts

- Use two-way expert tensor parallelism and two expert groups so 1,152-channel shards retain the native aligned FP8 and MXFP4 kernels.
- Adapt the grouped kernels to 192 target experts and 64 draft experts per bank to reuse their small- and medium-batch paths.
- Set the batching budget to 6,144 tokens to improve concurrent throughput while retaining the full context.
-    --gpu-memory-utilization 0.97

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
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC --cudart shared \
    -gencode arch=compute_120a,code=sm_120a \
    "$vllm_package_dir/distributed/device_communicators/sm120_host_all_reduce.cu" \
    -o /usr/local/lib/libvllm-sm120-host-ll.so
nvcc -O3 -std=c++17 -shared -Xcompiler=-fPIC --cudart shared \
    -gencode arch=compute_120a,code=sm_120a \
    "$vllm_package_dir/models/deepseek_v4/nvidia/ops/sm120_wo_a_mma.cu" \
    -o /usr/local/lib/libvllm-sm120-g2-wo-a-mma.so
export NCCL_TUNER_PLUGIN=/usr/local/lib/libvllm-sm120-tuner.so
export VLLM_SM120_EXPERT_TP=2
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
    --gpu-memory-utilization 0.97 \
    --max-num-batched-tokens 6144
```

| Concurrency | Prompts | Output t/s | p95 TTFT (ms) | p95 TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 192.2 | 131.5 | 6.15 |
| 4 | 128 | 441.8 | 102.2 | 10.49 |
| 8 | 128 | 628.9 | 228.9 | 15.40 |
| 16 | 128 | 861.3 | 283.1 | 21.06 |
| 32 | 128 | 1231.1 | 462.7 | 29.89 |
| 64 | 128 | 1504.3 | 889.7 | 50.18 |
| 128 | 256 | 2059.7 | 1618.5 | 80.12 |
| 256 | 512 | 2737.3 | 2908.4 | 151.46 |

## Accepted progress

| Stage | Change | c1 t/s | Revision |
|---:|---|---:|---|
| 0 | Four-GPU baseline | 68.9 | [`98e3fb6`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/98e3fb632bb37dea066d2f7d9cb6f1b4a436cb4d) |
| 1 | DSpark decoding | 131.3 | [`dee02d3`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/dee02d3567a50684bf4abb462d28390b3c3c9460) |
| 2 | Ninety-six local experts | 161.5 | [`26ae9ab`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/26ae9abd68d901a24ec36c582ae2c354a34eb81c) |
| 3 | Four-rank host collectives | 173.8 | [`630b1eb`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/630b1eb24926b2c70072dbe1d0c3a8af4adb4226) |
| 4 | Grouped BF16 projections | 178.5 | [`253839f`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/253839f710753836599f93bdcca65e822f0ac618) |
| 5 | Aligned TP2 and EP2 experts | 192.2 | This revision |
