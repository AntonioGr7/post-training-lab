"""Solve GSM8K problems with the GRPO-trained model — and measure accuracy before/after.

    uv run python generate.py --model_path Qwen/Qwen3-4B                      # baseline
    uv run python generate.py --model_path outputs/qwen3-4b-gsm8k-grpo        # after GRPO

Reports exact-match accuracy on a slice of the GSM8K test set and prints a few worked solutions, so
you can *see* the reasoning GRPO bought you (longer chains, more self-correction). This is the same
verifier used as the training reward — eval and reward are the same check, which is the point of RLVR.
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewards import SYSTEM_PROMPT, extract_answer, gsm8k_gold, _normalize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--num_examples", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--show", type=int, default=3, help="How many worked solutions to print.")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    data = load_dataset("openai/gsm8k", "main", split="test").select(range(args.num_examples))
    correct = 0
    for i, ex in enumerate(data):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        pred, gold = extract_answer(text), gsm8k_gold(ex["answer"])
        ok = pred is not None and pred == _normalize(gold)
        correct += ok
        if i < args.show:
            print("\n" + "=" * 80)
            print(f"Q: {ex['question']}")
            print(f"\n{text.strip()}")
            print(f"\n>>> predicted={pred} | gold={gold} | {'✓' if ok else '✗'}")

    print(f"\nGSM8K exact-match accuracy: {correct}/{len(data)} = {correct / len(data):.1%}")


if __name__ == "__main__":
    main()
