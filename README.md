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
| 1 | **85.0** | Not measured | **85.0** |
| 4 | **253.2** | Not measured | **253.2** |
| 8 | **375.4** | Not measured | **375.4** |
| 16 | **553.1** | Not measured | **553.1** |
| 32 | **657.4** | Not measured | **657.4** |
| 64 | **866.2** | Not measured | **866.2** |
| 128 | **1242.4** | Not measured | **1242.4** |
| 256 | **1814.1** | Not measured | **1814.1** |

![Accepted ShareGPT c1 throughput](8-nvidia-rtx-pro-6000/throughput.png)

![ShareGPT throughput by concurrency](8-nvidia-rtx-pro-6000/concurrency.png)

See the detailed changelog in [`8-nvidia-rtx-pro-6000/CHANGELOG.md`](8-nvidia-rtx-pro-6000/CHANGELOG.md).

Run the requirements and full `vllm serve` command in the hardware changelog; each directory also includes a Dockerfile that retains the image's default entrypoint without model or parallelism defaults.

Measurements use `vllm bench serve` with ShareGPT, seed 0, temperature 0, dataset output lengths, and matching-concurrency warmups, with 128 measured prompts through c64, 256 at c128, and 512 at c256.

`benchmark.py` only sends benchmark requests to an existing server; run `python benchmark.py --help` for its model, dataset, endpoint, and output options.

Throughput is aggregate generated output tokens per second, including reasoning tokens; results depend on prompt lengths, cache state, and concurrency.

The measured model revision retains a 1,048,576-token context and 1,024 scheduler slots; the ShareGPT sweep does not measure performance at the full context length.

The recipes target this model, sm120 hardware, and vLLM version; the [Apache 2.0 license](LICENSE) covers this repository, and model weights retain their upstream terms.

[^8gpu-baseline]: Baseline uses sm120 compatibility fixes, FlashInfer 0.7.1, and `--block-size 64`; stock vLLM failed before inference.
