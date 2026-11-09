# llm-inference-bench

Release `0.8.0`, source [`52216d7`](https://github.com/local-inference-lab/llm-inference-bench/tree/52216d70865a015b34689626d04a80b89c9dc45f).

Use the [shared setup](../README.md#setup) before these commands.

Pinned client, 30-second sustained decode cells, and contexts of 0 and 16,384 tokens.

```bash
curl -fL https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/52216d70865a015b34689626d04a80b89c9dc45f/llm_decode_bench.py \
    -o /tmp/llm_decode_bench.py
LLM_BENCH_NO_UPDATE_CHECK=1 uv run --python 3.12 --with httpx==0.28.1 --with rich==14.1.0 --with psutil==7.0.0 \
    python /tmp/llm_decode_bench.py \
    --host "$URL" --model "$MODEL" --concurrency 1,8,32 --contexts 0,16384 \
    --duration 30 --max-tokens 4096 --temperature 0 --skip-prefill --display-mode plain \
    --output "$OUTPUT/llm-inference-bench.json"
```

## Extended matrix

Use the optimized server with `--gpu-memory-utilization 0.95` for the extended sweep; this run is separate from the baseline comparisons below.

The pinned client caps standalone prefill at 128k, so [`prefill.patch`](prefill.patch) lifts that cap to the server context limit and labels the client `0.8.0+prefill-window`.

```bash
patch --batch --fuzz=0 -p1 -d /tmp < benchmarks/llm-inference-bench/prefill.patch
LLM_BENCH_NO_UPDATE_CHECK=1 uv run --python 3.12 --with httpx==0.28.1 --with rich==14.1.0 --with psutil==7.0.0 \
    python /tmp/llm_decode_bench.py \
    --host "$URL" --model "$MODEL" \
    --contexts 8192,16384,32768,65536,131072,262144,524288,1043456 \
    --concurrency 1,2,4,8,16,32,64,128 \
    --prefill-contexts 8192,16384,32768,65536,131072,262144,524288,1043456,1048512 \
    --duration 30 --decode-warmup-seconds 3 --cell-warmup-timeout-seconds 300 \
    --standalone-prefill --prefill-duration 30 --max-tokens 4096 --temperature 0 \
    --no-resume --display-mode plain --output "$OUTPUT/llm-inference-bench.json"
```

Context labels are targets, with actual prompt lengths recorded in the reports; oversized combinations are skipped and capacity or occupancy flags are retained.

The [published matrices](../localmaxxing/#results) include nine cold-prefill points per topology, measured using client TTFT and unique request prefixes.

## Results

Aggregate sustained decode throughput uses the measured 30-second window after each cell warms its context; context 0 still includes the benchmark prompt.

Measured on 2026-11-05 with the [recorded revisions](../provenance.json); all reported cells completed without request errors.

### 8 x NVIDIA RTX PRO 6000

| Target context tokens | Concurrency | Baseline + MTP t/s | Optimized t/s | Mean TTFT ms, baseline / optimized |
|---:|---:|---:|---:|---:|
| 0 | 1 | 155.9 | **312.5 (+100.4%)** | 66.3 / 47.7 |
| 16384 | 1 | 153.6 | **281.7 (+83.4%)** | 130.4 / 99.6 |
| 0 | 8 | 555.8 | **1049.3 (+88.8%)** | 243.2 / 102.7 |
| 16384 | 8 | 551.1 | **1035.7 (+87.9%)** | 293.5 / 185.7 |
| 0 | 32 | 1247.5 | **1929.4 (+54.7%)** | 366.0 / 349.6 |
| 16384 | 32 | 1203.1 | **1879.0 (+56.2%)** | 855.3 / 699.2 |

### 4 x NVIDIA RTX PRO 6000

| Target context tokens | Concurrency | Baseline + MTP t/s | Optimized t/s | Mean TTFT ms, baseline / optimized |
|---:|---:|---:|---:|---:|
| 0 | 1 | 134.6 | **241.6 (+79.5%)** | 69.6 / 53.3 |
| 16384 | 1 | 135.7 | **252.0 (+85.7%)** | 118.8 / 101.7 |
| 0 | 8 | 543.3 | **915.1 (+68.4%)** | 247.7 / 127.2 |
| 16384 | 8 | 537.5 | **884.3 (+64.5%)** | 313.6 / 282.6 |
| 0 | 32 | 1341.2 | **1838.4 (+37.1%)** | 366.9 / 343.0 |
| 16384 | 32 | 1305.0 | **1815.8 (+39.1%)** | 779.2 / 734.8 |
