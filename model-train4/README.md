# Qwen3.5 GSPO Smoke Stack

This directory contains a realistic throughput smoke pipeline for pairwise virality training on a local `Qwen3.5-9B-Base` checkpoint.

Layout:

- core training scripts live in the root of `model-train4/`
- optional helper utilities live in `model-train4/tools/`
- raw source data now lives in `model-train4/data/raw/`
- prepared manifests for S3 utilities live in `model-train4/data/manifests/`
- runtime artifacts stay in `output/`, `cache/`, `s3_cache/`, and `wandb/`

What it checks:

- raw video pair input
- 2 fps / 45 sec video regime
- offline full transcript generation
- offline audio-summary generation
- Unsloth GSPO training with a minimal reward
- post-train pairwise inference from the saved LoRA adapter

Important implementation notes:

- The smoke setup keeps the input load close to the intended final training setup.
- The simplification is only in the reward and in the number of training pairs.
- The scripts expect the base model in `~/models/Qwen3.5-9B-Base` by default.
- Qwen3.5 RL is wired through Unsloth inference (`fast_inference=False`), not through `vLLM 0.16.0`.
- The default training mode is now regular bf16 LoRA. `--load_in_4bit` is still available as an explicit fallback if you want to compare it against QLoRA.

Quickstart:

```bash
cd model-train/model-train4
bash setup_env.sh
bash run_gspo_qwen35_smoke.sh
```

Override the local model path if needed:

```bash
MODEL_PATH=/absolute/path/to/Qwen3.5-9B-Base bash run_gspo_qwen35_smoke.sh
```
