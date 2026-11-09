# Output quality

The historical LocalMaxxing speed tests used temperature 0 and an 8,192-token cap, and their saved samples contain unfinished reasoning, including a repetition loop on eight GPUs.

LocalMaxxing combines reasoning and final content in its preview, and its verified-run label validates the benchmark submission rather than the generated program.

The [pinned checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md) recommends temperature 1, top_p 1 and at least 262,144 output tokens; a completed answer to the saved prompt required 11,896 tokens in our sampled control.

## Regression checks

| Configuration | DSpark | GSM8K /128 | HumanEval /164 |
|---|---|---:|---:|
| 8-GPU compatibility reference | On | 124 | 159 |
| 8-GPU compatibility reference | Off | 126 | 157 |
| 8-GPU optimized | On | 125 | 155 |
| 8-GPU optimized | Off | 124 | 158 |
| 4-GPU optimized | On | 125 | 155 |
| 4-GPU optimized | Off | 125 | 159 |

One completion per problem, seed 42, temperature 1, top_p 1, concurrency 8 and GPU memory utilization 0.95; model, image and source revisions are recorded in [provenance.json](../provenance.json).

The four-task coding gap with DSpark remains unresolved; these sampled results establish neither quality parity nor production readiness.

- GSM8K selects the first 128 test indices sorted by the SHA256 bytes of `deepseek-sm120-quality-audit-v2:{index}`, appends `\nGive your final answer as #### <number>.`, uses reasoning effort 50 and an 8,192-token cap, and compares the final numeric answer with the original label.
- HumanEval uses all 164 tasks, thinking disabled and an 8,192-token cap, requests the complete function and imports with the prefix in provenance, and executes the official tests against that returned implementation with a 10-second timeout.
- The HumanEval evaluator passed all 164 canonical solutions and rejected all 164 empty solutions before scoring model answers.
- The original seven scenario prompts completed, with strict JSON checks and independent LRU eviction, update and concurrent-access tests on both optimized GPU counts.
- Thinking-enabled `logtally` answers passed malformed-record and timestamp-filter tests, while thinking-disabled answers counted invalid-latency records, so disabling thinking is not a validated workaround.

These are chat-adapted regression checks rather than reproductions of the producer's published scores.

## Diagnosis

Frozen-state fast/native sampler comparisons matched valid tokens through at least 2,048 graph-mode calls and 3,072 eager-mode calls, which checks the sampler on observed inputs without establishing equivalence of every kernel.

Both eager reference and optimized final programs passed malformed-record, timestamp-filter, timezone-offset and percentile checks; the separately tested upstream graph fix did not cure greedy looping and was not shipped as a cure.

The producer's [DeepSWE reproduction guide](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/evaluation/README.md) supplies the agent harness and task environments for an agent-level evaluation; that benchmark has not been reproduced here.
