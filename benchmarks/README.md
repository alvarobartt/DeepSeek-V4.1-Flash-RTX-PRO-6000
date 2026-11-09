# Benchmarks

Compare baseline + MTP (DSpark) with the optimized recipe on the same GPUs, using the same client commands and one server at a time.

| GPUs | Baseline + MTP recipe | Draft tokens |
|---:|---|---:|
| 4 | [`dee02d3`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/blob/dee02d3567a50684bf4abb462d28390b3c3c9460/4-nvidia-rtx-pro-6000/CHANGELOG.md) | 3 |
| 8 | [`e5ac344`](https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000/blob/e5ac344074d1e9141c50e3178cde4e2a3680da8b/8-nvidia-rtx-pro-6000/CHANGELOG.md) | 5 |

Each baseline revision contains its Dockerfile, cumulative patch, and serving command; use the root [README](../README.md) for the optimized container.

Build a baseline from its revision without changing the working tree, then use its documented serving arguments:

```bash
GPU_COUNT=8
BASELINE_REVISION=e5ac344074d1e9141c50e3178cde4e2a3680da8b
BASELINE_DIR=$(mktemp -d)
git archive "$BASELINE_REVISION" | tar -x -C "$BASELINE_DIR"
docker build -f "$BASELINE_DIR/$GPU_COUNT-nvidia-rtx-pro-6000/Dockerfile" \
    -t "vllm:baseline-mtp-$BASELINE_REVISION" "$BASELINE_DIR"
```

## Setup

Requires `uv` and `curl`; run from the repository root and change `OUTPUT` for each hardware configuration and recipe.

```bash
MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
URL=http://127.0.0.1:8000
TOKENIZER=/tmp/deepseek-v41-tokenizer
OUTPUT=results/8-gpu/optimized
uvx --python 3.12 --from huggingface-hub hf download "$MODEL" tokenizer.json tokenizer_config.json config.json \
    --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir "$TOKENIZER"
mkdir -p "$OUTPUT"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
```

## Clients

| Benchmark | Workload |
|---|---|
| [AIPerf](aiperf/) | ShareGPT at c1 through c256 |
| [AgentX](agentx/) | Coding-agent traces at c1 and c8 |
| [llm-inference-bench](llm-inference-bench/) | Sustained decode comparisons and an extended context/concurrency matrix |
| [LocalMaxxing](localmaxxing/) | Canonical prompts at c1 and published speed tests and matrices |
| [Quality checks](quality/) | GSM8K, executable HumanEval tests and generated-program validation |

Each directory contains its pinned client commands and compact results, with shared [provenance](provenance.json); raw logs and request traces stay outside Git.

The clients run without benchmark-service accounts; publishing on LocalMaxxing requires an account and `lmx auth login`.

Compare results within each tool because their workloads and throughput denominators differ.

Some high-concurrency ShareGPT and AgentX results have higher p95 TTFT, as shown alongside throughput in the tables.

The historical throughput comparisons use temperature 0; the output-quality investigation uses the checkpoint's recommended temperature 1.0 and checks final answers separately.
