# LocalMaxxing

Release `0.1.60`, source [`f02df4a`](https://github.com/LottoLottoLotto/localmaxxing-cli/tree/f02df4ae88e3bf8b1fbf6fca767bb96c032f2a34).

Use the [shared setup](../README.md#setup) before these commands.

Install the Linux x86-64 release and verify its checksum.

```bash
mkdir -p /tmp/localmaxxing
curl -fL https://github.com/LottoLottoLotto/localmaxxing-cli/releases/download/v0.1.60/lmx-linux-amd64.tar.gz \
    -o /tmp/localmaxxing/lmx-linux-amd64.tar.gz
(cd /tmp/localmaxxing && \
    echo '0cf00caca43f2c14ed21922a96bd9fc5b478c2c2374292b30a308e3e3119b7f3  lmx-linux-amd64.tar.gz' | sha256sum --check && \
    tar -xzf lmx-linux-amd64.tar.gz)
```

Generate hardware metadata on the server host and check its GPU count against the Docker allocation.

```bash
/tmp/localmaxxing/lmx hardware --out "$OUTPUT/hardware.json"
```

## Speed test

CLI 0.1.60 sends temperature but omits top_p in remote speed tests, so these measurements pin the server default with `--override-generation-config '{"temperature":1.0,"top_p":1.0}'`.

```bash
/tmp/localmaxxing/lmx speed-test run vllm --mode remote --base-url "$URL" \
    --hf-id "$MODEL" --served-model "$MODEL" --quantization 'MXFP4+FP8' \
    --engine-version 0.32.0 --model-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --engine-repository https://github.com/alvarobartt/DeepSeek-V4.1-Flash-RTX-PRO-6000 \
    --engine-commit 889b538fc5098835454a5e592492c80db0fe3102 \
    --engine-build ghcr.io/alvarobartt/vllm@sha256:ebdd9351069705b01e49d3fe080a427729c5680c5fd87d13c7fef163d041838a \
    --hardware "$OUTPUT/hardware.json" \
    --spec-method MTP --spec-num-tokens 5 --canonical-prompt code-v1 \
    --temperature 1 --max-tokens 262144 --warmup 2 --iterations 5 \
    --out "$OUTPUT/speed-test.json" --runs-dir "$OUTPUT/speed-test-runs"
```

Repeat with `--canonical-prompt reasoning-v1` for the reasoning test; use five draft tokens for either optimized recipe and three for the four-GPU baseline + MTP recipe.

Check every timed response for a completed final answer and functional correctness before submitting; `--enable-log-requests --enable-log-outputs --no-enable-log-deltas --max-log-len 1048576` retains complete vLLM responses for that audit.

## Results

| GPUs | Prompt | Median decode t/s | Median TTFT ms |
|---:|---|---:|---:|
| 4 | code-v1 | [268.2](https://www.localmaxxing.com/en/models/deepseek-ai/DeepSeek-V4.1-Flash-1017?run=cmw4r18p00c8qoa017nckg5bb) | 74.93 |
| 4 | reasoning-v1 | [208.0](https://www.localmaxxing.com/en/models/deepseek-ai/DeepSeek-V4.1-Flash-1017?run=cmw4r18y60c8toa01xt1zu44m) | 80.97 |
| 8 | code-v1 | [340.7](https://www.localmaxxing.com/en/models/deepseek-ai/DeepSeek-V4.1-Flash-1017?run=cmw4r19730c8woa01mw67rblr) | 60.73 |
| 8 | reasoning-v1 | [266.7](https://www.localmaxxing.com/en/models/deepseek-ai/DeepSeek-V4.1-Flash-1017?run=cmw4r19k30c8zoa01ltcmcoco) | 70.62 |

The November 6 rerun uses temperature 1, top_p 1, two warmups, five timed samples, a 262,144-token output cap and an explicit 1,048,576-token context, with image revision `889b538` and GPU memory utilization 0.95; every request's effective sampling settings were checked in vLLM logs.

Rates include reasoning tokens and exclude prefill, and differences from earlier sampled runs are not attributed to new optimization.

All seven code programs per GPU count passed eight isolated functional cases each, covering malformed records, timestamp filtering, timezone offsets, percentiles, empty input, invalid record shapes, missing latency and singleton percentiles.

Reasoning answers completed all four parts with consistent budget arithmetic, and the battery question permits pooled-budget and equal-share interpretations, so this review is not a formal accuracy score.

The previews show final-answer excerpts, and the [quality investigation](../quality/) records the unresolved HumanEval gap; these checks do not establish production readiness.

Earlier temperature-0 capped runs produced unfinished reasoning and are excluded from this table; their public reports carry investigation notes.

The [four-GPU](https://www.localmaxxing.com/en/inference-benchmarks/cmw41hm250c39oa01nud4y8yd) and [eight-GPU](https://www.localmaxxing.com/en/inference-benchmarks/cmw41hn3g0c5goa01z8a4ylt4) matrices retain the November 5 load-only protocol at temperature 0; they were not rerun with the canonical tests.

The matrices target eight decode contexts from 8k to 1,019k, eight concurrencies from 1 to 128, and nine cold-prefill points; four GPUs measured 36 decode cells and eight GPUs measured 58, retaining native skips, underfilled cells and three capacity-limited cells on eight GPUs.

Actual maximum prefill prompts were approximately 1,027k tokens, prefill timing uses client TTFT, and matrix token rates do not validate semantic accuracy.

## Publishing

Sign in once with `/tmp/localmaxxing/lmx auth login` to publish under your account.

Before submitting with CLI 0.1.60, keep `outputSample` within 4,000 characters, retain the complete output and hash, use flat hardware metadata (`gpuName`, `gpuCount`, `vramGb`), and attach the actual serving command and sampling settings.

```bash
/tmp/localmaxxing/lmx speed-test dry-run "$OUTPUT/speed-test.json"
/tmp/localmaxxing/lmx speed-test submit "$OUTPUT/speed-test.json"
```

Import the [llm-inference-bench](../llm-inference-bench/) matrix using hardware metadata from the measured server, then validate and submit it:

```bash
/tmp/localmaxxing/lmx speed-test matrix import "$OUTPUT/llm-inference-bench.json" \
    --hf-id "$MODEL" --quantization 'MXFP4+FP8' \
    --model-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --engine-version 0.32.0 --hardware "$OUTPUT/hardware.json" \
    --out "$OUTPUT/matrix.json"
/tmp/localmaxxing/lmx speed-test matrix dry-run "$OUTPUT/matrix.json"
/tmp/localmaxxing/lmx speed-test matrix submit "$OUTPUT/matrix.json"
```

Use the [extended matrix command](../llm-inference-bench/#extended-matrix) before importing; matrices and canonical prompt speed tests are separate measurements.
