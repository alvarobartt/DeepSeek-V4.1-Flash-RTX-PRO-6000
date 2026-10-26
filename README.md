# DeepSeek-V4.1-Flash-RTX-PRO-6000

<table>
    <tr><td>Model</td><td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017">DeepSeek-V4.1-Flash-1017</a></td></tr>
    <tr><td>Revision</td><td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170"><code>1bae0e1</code></a></td></tr>
    <tr><td>Hardware</td><td>NVIDIA RTX PRO 6000 (96 GB each)</td></tr>
    <tr><td>Architecture</td><td>sm120</td></tr>
    <tr><td>Precision</td><td>BF16 activations with native FP8 and MXFP4 weights</td></tr>
    <tr><td>Image</td><td><code>vllm/vllm-openai:v0.32.0</code></td></tr>
</table>

Validated serving recipes for four and eight GPUs, using the checkpoint's original weights and precision.

## 8 x NVIDIA RTX PRO 6000

| Concurrency | Baseline t/s [^8gpu-baseline] | Baseline + DSpark t/s | Current t/s |
|---:|---:|---:|---:|
| 1 | 85.0 | 159.2 (+87.3%) | **211.4 (+32.8%)** |
| 4 | 253.2 | 380.4 (+50.2%) | **523.3 (+37.6%)** |
| 8 | 375.4 | 496.2 (+32.2%) | **736.2 (+48.4%)** |
| 16 | 553.1 | 650.5 (+17.6%) | **1000.1 (+53.7%)** |
| 32 | 657.4 | 954.1 (+45.1%) | **1319.1 (+38.3%)** |
| 64 | 866.2 | 1185.0 (+36.8%) | **1647.7 (+39.0%)** |
| 128 | 1242.4 | 1471.9 (+18.5%) | **2239.7 (+52.2%)** |
| 256 | 1814.1 | 1726.3 (-4.8%) | **2983.6 (+72.8%)** |

![Accepted ShareGPT c1 throughput](8-nvidia-rtx-pro-6000/throughput.png)

![ShareGPT throughput by concurrency](8-nvidia-rtx-pro-6000/concurrency.png)

See the detailed changelog in [`8-nvidia-rtx-pro-6000/CHANGELOG.md`](8-nvidia-rtx-pro-6000/CHANGELOG.md).

## 4 x NVIDIA RTX PRO 6000

| Concurrency | Baseline t/s [^4gpu-baseline] | Baseline + DSpark t/s | Current t/s |
|---:|---:|---:|---:|
| 1 | 68.9 | 131.3 (+90.6%) | **161.5 (+23.0%)** |
| 4 | 216.8 | 305.4 (+40.9%) | **398.5 (+30.5%)** |
| 8 | 317.3 | 450.0 (+41.8%) | **582.3 (+29.4%)** |
| 16 | 425.8 | 646.7 (+51.9%) | **798.6 (+23.5%)** |
| 32 | 553.8 | 929.3 (+67.8%) | **1126.5 (+21.2%)** |
| 64 | 735.3 | 1216.4 (+65.4%) | **1354.5 (+11.4%)** |
| 128 | 1147.5 | 1779.2 (+55.1%) | **1850.7 (+4.0%)** |
| 256 | 1683.6 | 2090.1 (+24.1%) | **2506.8 (+19.9%)** |

![Accepted ShareGPT c1 throughput](4-nvidia-rtx-pro-6000/throughput.png)

![ShareGPT throughput by concurrency](4-nvidia-rtx-pro-6000/concurrency.png)

See the detailed changelog in [`4-nvidia-rtx-pro-6000/CHANGELOG.md`](4-nvidia-rtx-pro-6000/CHANGELOG.md).

Run the requirements and full `vllm serve` command in the hardware changelog; each directory also includes a Dockerfile that retains the image's default entrypoint without model or parallelism defaults.

Measurements use `vllm bench serve` with ShareGPT, seed 0, temperature 0, dataset output lengths, and matching-concurrency warmups, with 128 measured prompts through c64, 256 at c128, and 512 at c256.

`benchmark.py` only sends benchmark requests to an existing server; run `python benchmark.py --help` for its model, dataset, endpoint, and output options.

Throughput is aggregate generated output tokens per second, including reasoning tokens; results depend on prompt lengths, cache state, and concurrency.

The measured model revision retains a 1,048,576-token context and 1,024 scheduler slots; the ShareGPT sweep does not measure performance at the full context length.

The recipes target this model, sm120 hardware, and vLLM version; the [Apache 2.0 license](LICENSE) covers this repository, and model weights retain their upstream terms.

[^8gpu-baseline]: Baseline uses sm120 compatibility fixes, FlashInfer 0.7.1, and `--block-size 64`; stock vLLM failed before inference.
[^4gpu-baseline]: Baseline includes sm120 compatibility fixes, native FP8 KV-cache handling, grouped FP4 experts, and expert parallelism with DSpark disabled; the DSpark reference adds three adaptive draft tokens.
