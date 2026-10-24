import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128, 256])
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error("dataset must be a ShareGPT JSON file")
    if min(args.concurrency) < 1 or (args.num_prompts is not None and args.num_prompts < 1):
        parser.error("request counts and concurrency must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ | {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for concurrency in args.concurrency:
        num_prompts = args.num_prompts or max(128, 2 * concurrency)
        command = [
            "vllm", "bench", "serve",
            "--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
            "--base-url", args.base_url, "--model", args.model,
            "--tokenizer-mode", args.tokenizer_mode,
            "--dataset-name", "sharegpt", "--dataset-path", str(args.dataset),
            "--num-prompts", str(num_prompts), "--seed", "0",
            "--max-concurrency", str(concurrency), "--request-rate", "inf",
            "--num-warmups", str(concurrency), "--temperature", "0",
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
            "--save-result", "--save-detailed", "--disable-tqdm",
            "--result-dir", str(args.output), "--result-filename", f"c{concurrency}.json",
        ]
        if args.tokenizer:
            command += ["--tokenizer", args.tokenizer]
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
