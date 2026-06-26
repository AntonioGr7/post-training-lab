"""Generate summaries with the PPO-tuned policy — compare before/after RLHF.

    uv run python generate.py --model_path outputs/ppo-tldr
    uv run python generate.py --model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  # the SFT baseline

Run it on both the SFT baseline and your PPO checkpoint to *see* what RLHF bought you (and watch for
reward-hacked artifacts: truncation, length gaming, degenerate phrasing). PPOTrainer also logs sample
generations during training — this is for poking at the final model yourself.
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--num_examples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=53)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    prompts = load_dataset("trl-lib/tldr", split="validation").select(range(args.num_examples))
    for ex in prompts:
        inputs = tokenizer(ex["prompt"], return_tensors="pt", truncation=True, max_length=512).to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        summary = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 80)
        print(ex["prompt"][-500:])
        print(f"\n>>> SUMMARY: {summary.strip()}")


if __name__ == "__main__":
    main()
