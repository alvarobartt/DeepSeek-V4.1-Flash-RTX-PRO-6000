# 8 x NVIDIA RTX PRO 6000

## 0. Baseline

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --tokenizer-mode deepseek_v41 --tensor-parallel-size 8 --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 --mm-encoder-tp-mode data
```

- Failed with `Assertion error (csrc/apis/attention.hpp:411): block_kv == 32 or block_kv == 64`.
