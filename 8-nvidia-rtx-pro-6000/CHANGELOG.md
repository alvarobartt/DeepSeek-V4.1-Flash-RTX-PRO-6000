# 8 x NVIDIA RTX PRO 6000

The patch is cumulative against `vllm/vllm-openai:v0.32.0`; every working milestone contains its complete recipe.

FlashInfer 0.7.1 replaces the bundled 0.7.0.post1 for runtime page-size support, and both the sm120 compatibility patch and `--block-size 64` are required.

## 3. Three-token drafts

- Reduce the draft from five to three tokens because later positions are rarely accepted and still add verification work.
-    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"revision":"1bae0e1088993b9b82de08f59de5be5a34e42170"}'

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
    --block-size 64 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":3,"revision":"1bae0e1088993b9b82de08f59de5be5a34e42170"}'
```

| Concurrency | Prompts | Output t/s | p95 TTFT (ms) | p95 TPOT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 128 | 155.8 | 136.2 | 7.61 |
| 4 | 128 | 383.7 | 116.2 | 12.26 |
| 8 | 128 | 528.1 | 225.2 | 17.13 |
| 16 | 128 | 681.7 | 284.8 | 26.60 |
| 32 | 128 | 909.9 | 427.0 | 39.72 |
| 64 | 128 | 1227.4 | 872.2 | 50.68 |
| 128 | 256 | 1677.5 | 1451.8 | 81.97 |
| 256 | 512 | 2288.8 | 2785.3 | 143.19 |

## Accepted progress

| Stage | Change | c1 t/s | Revision |
|---:|---|---:|---|
| 1 | sm120 support | 85.0 | [`0cd46b2`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/0cd46b25086a61436eb0979abf115f66442e889e) |
| 2 | DSpark decoding | 159.2 | [`e5ac344`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/tree/e5ac344074d1e9141c50e3178cde4e2a3680da8b) |
| 3 | Three-token drafts | 155.8 | This revision |
