"""Chat with the DPO-aligned model — compare it against the base/SFT model.

    uv run python generate.py --model_path outputs/qwen3-4b-ultrafeedback-dpo
    uv run python generate.py --model_path Qwen/Qwen3-4B          # the un-aligned baseline

Run it on BOTH the baseline and your DPO checkpoint, side by side, to *see* what preference
alignment bought you: helpfulness, formatting, refusal calibration, instruction-following. DPO is
offline, so it can only sharpen behaviors already present in the pair distribution — watch for that
ceiling (it motivates the online methods in folders 05/06).

LoRA checkpoint? Point --model_path at the adapter dir; it loads via the base model automatically.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain the difference between DPO and RLHF with PPO in two sentences.",
    "Write a short, encouraging note to someone learning to code.",
    "What are three common mistakes when fine-tuning an LLM on preferences?",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        reply = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 80)
        print(f">>> PROMPT: {prompt}")
        print(f">>> REPLY:  {reply.strip()}")


if __name__ == "__main__":
    main()
