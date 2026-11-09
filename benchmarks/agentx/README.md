# AgentX

Release `1.0.13`, source [`60a094f`](https://github.com/SemiAnalysisAI/agentx-harness/tree/60a094f256b42f55fde1efb3d45672e1d9a44566).

Use the [shared setup](../README.md#setup) before these commands.

This unofficial comparison uses 15 minutes of profiling and traces with peak context up to 131,072 tokens; the full InferenceX recipe uses one hour.

Each concurrency lane runs a session tree with its subagents.

Restart the server before each concurrency point, then run with `CONCURRENCY=1` and `CONCURRENCY=8` separately.

For c8, restart the optimized server with `--gpu-memory-utilization 0.95` to leave room for DeepGEMM prefill workspaces.

```bash
CONCURRENCY=1
uvx --python 3.12 --from agentx-harness==1.0.13 agentx profile \
    --url "$URL" --endpoint /v1/chat/completions --model "$MODEL" --tokenizer "$TOKENIZER" \
    --public-dataset semianalysis_cc_traces_weka_062126 --max-context-length 131072 \
    --concurrency "$CONCURRENCY" --random-seed 42 --extra-inputs temperature:0 \
    --benchmark-duration 900 --benchmark-grace-period 900 --agentic-warmup-grace-period 900 \
    --output-artifact-dir "$OUTPUT/agentx-c$CONCURRENCY" --ui-type none
```

## Results

Throughput includes reasoning tokens over the profiling window, and optimized runs have higher p95 TTFT.

Measured on 2026-11-05 with the [recorded revisions](../provenance.json); all reported cells completed without request errors.

The c8 pairs were rerun on matched leases with `--gpu-memory-utilization 0.95` on the optimized servers after the original settings ran out of memory during prefill.

### 8 x NVIDIA RTX PRO 6000

| Concurrency | Baseline + MTP t/s | Optimized t/s | p95 TTFT ms, baseline / optimized | Requests, baseline / optimized |
|---:|---:|---:|---:|---:|
| 1 | 128.4 | **163.1 (+27.1%)** | 982.4 / 1114.4 | 134 / 173 |
| 8 | 292.7 | **303.5 (+3.7%)** | 1286.4 / 1569.8 | 360 / 374 |

### 4 x NVIDIA RTX PRO 6000

| Concurrency | Baseline + MTP t/s | Optimized t/s | p95 TTFT ms, baseline / optimized | Requests, baseline / optimized |
|---:|---:|---:|---:|---:|
| 1 | 111.6 | **149.3 (+33.8%)** | 1030.9 / 1096.9 | 94 / 156 |
| 8 | 279.4 | **303.2 (+8.5%)** | 1439.5 / 1502.4 | 338 / 369 |
