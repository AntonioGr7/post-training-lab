"""Use your trained reward model to score responses — the qualitative check.

A reward model is only useful if it ranks good answers above bad ones on inputs it never saw.
This script loads the RM and scores one or more candidate responses to a prompt, printing the
scalar reward for each and which one the RM prefers.

    uv run python score.py --model_path outputs/qwen3-4b-ultrafeedback-rm \
        --prompt "Explain why the sky is blue." \
        --responses "Rayleigh scattering makes shorter (blue) wavelengths scatter more." \
                    "Because blue is the ocean's color reflecting upward."

With no --responses, a built-in good/bad pair is scored so you can sanity-check direction.
Try probing for reward hacking: does the RM prefer longer or more confident answers regardless
of correctness? (See README §"Reward hacking".)
"""

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_PROMPT = "What is the capital of France, and why is it significant?"
DEFAULT_RESPONSES = [
    "Paris is the capital of France. It's the political, cultural, and economic center of the "
    "country, home to institutions like the Louvre and the seat of government.",
    "i think its lyon or maybe marseille, not totally sure honestly",
]


def score(model, tokenizer, prompt: str, response: str) -> float:
    """Reward of a single (prompt, response) pair under the RM's chat template."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits  # shape [1, 1] — a single scalar reward
    return logits.squeeze().item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to the trained reward model.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--responses", nargs="+", default=DEFAULT_RESPONSES,
                   help="One or more candidate responses to rank.")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    print(f"\nPROMPT: {args.prompt}\n" + "-" * 80)
    scored = [(r, score(model, tokenizer, args.prompt, r)) for r in args.responses]
    scored.sort(key=lambda x: x[1], reverse=True)
    for rank, (resp, reward) in enumerate(scored, 1):
        print(f"#{rank}  reward={reward:+.3f}  | {resp[:100]}{'...' if len(resp) > 100 else ''}")
    print(f"\nRM prefers: {scored[0][0][:80]!r}")


if __name__ == "__main__":
    main()
