"""Talk to your fine-tuned model — the qualitative half of 'did SFT work?'.

    uv run python generate.py --model_path outputs/qwen3-4b-tulu3-sft
    uv run python generate.py --model_path outputs/qwen3-4b-tulu3-sft \
        --prompt "Explain gradient checkpointing in two sentences."

Loss curves tell you optimization worked; this tells you the model actually learned to follow
instructions and to STOP (emits EOS) instead of rambling. Compare against the base model to see
what SFT bought you.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Write a haiku about gradient descent.",
    "Give me three tips for debugging a CUDA out-of-memory error.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to the SFT checkpoint (output_dir).")
    p.add_argument("--prompt", default=None, help="Single prompt; omit to run the default set.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        # add_generation_prompt=True appends the assistant turn marker so the model knows to speak.
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 80)
        print(f"USER: {prompt}")
        print(f"ASSISTANT: {completion.strip()}")


if __name__ == "__main__":
    main()
