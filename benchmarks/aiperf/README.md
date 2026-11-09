# AIPerf

Release `0.13.0`, source [`794f8bb`](https://github.com/ai-dynamo/aiperf/tree/794f8bb75f8582f22e412d7e650fc71ca2a3d21a).

Use the [shared setup](../README.md#setup) before these commands.

ShareGPT output-length targets, seed 0, and 64 requests or two waves of concurrency, whichever is larger.

```bash
for concurrency in 1 4 8 16 32 64 128 256; do
    requests=$((concurrency < 32 ? 64 : 2 * concurrency))
    uvx --python 3.12 aiperf==0.13.0 profile \
        --url "$URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
        --endpoint-type chat --streaming --use-server-token-count \
        --public-dataset sharegpt --num-dataset-entries 128 --random-seed 0 \
        --concurrency "$concurrency" --request-count "$requests" --warmup-request-count "$concurrency" \
        --extra-inputs temperature:0 \
        --output-artifact-dir "$OUTPUT/aiperf-c$concurrency" --ui-type none
done
```

## Results

Some high-concurrency results have higher p95 TTFT.

Aggregate output throughput includes reasoning tokens and uses the profiling observation window; the public ShareGPT loader uses its output-length targets.

Measured on 2026-11-05 with the [recorded revisions](../provenance.json); all reported cells completed without request errors.

### 8 x NVIDIA RTX PRO 6000

| Concurrency | Baseline + MTP t/s | Optimized t/s | p95 TTFT ms, baseline / optimized |
|---:|---:|---:|---:|
| 1 | 163.7 | **291.7 (+78.2%)** | 139.5 / 99.6 |
| 4 | 374.0 | **645.9 (+72.7%)** | 191.4 / 115.5 |
| 8 | 470.3 | **802.1 (+70.6%)** | 249.4 / 156.1 |
| 16 | 582.9 | **1029.5 (+76.6%)** | 290.2 / 280.6 |
| 32 | 756.6 | **1323.8 (+75.0%)** | 2117.4 / 433.4 |
| 64 | 1312.1 | **2111.7 (+60.9%)** | 848.3 / 743.0 |
| 128 | 1548.3 | **2493.2 (+61.0%)** | 1670.8 / 2890.3 |
| 256 | 2295.1 | **2636.9 (+14.9%)** | 2462.1 / 6157.4 |

### 4 x NVIDIA RTX PRO 6000

| Concurrency | Baseline + MTP t/s | Optimized t/s | p95 TTFT ms, baseline / optimized |
|---:|---:|---:|---:|
| 1 | 135.7 | **236.6 (+74.3%)** | 141.2 / 106.8 |
| 4 | 295.6 | **521.2 (+76.3%)** | 165.1 / 148.3 |
| 8 | 423.4 | **672.1 (+58.7%)** | 225.6 / 157.4 |
| 16 | 571.8 | **894.5 (+56.4%)** | 302.7 / 299.1 |
| 32 | 784.5 | **1210.2 (+54.3%)** | 389.9 / 423.3 |
| 64 | 1295.5 | **1733.5 (+33.8%)** | 846.5 / 808.5 |
| 128 | 1895.6 | **2363.0 (+24.7%)** | 1675.1 / 1809.0 |
| 256 | 2328.8 | **2776.1 (+19.2%)** | 2510.1 / 2958.8 |

The c32 pair was rerun on one matched lease with `--connection-reuse-strategy never` after one client connection reset; the failed measurement is excluded.
