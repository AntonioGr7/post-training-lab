"""A lightweight eval harness — reward-model-as-judge win-rate between two policies.

This is the course's tying-together piece: it reuses a reward model (folder 02) to *evaluate* the
policies that folders 04/06 produced. Give it two model paths (e.g. the base/SFT model and your
Online-DPO checkpoint); it generates on a held-out prompt set, scores both with the RM, and reports:

  • mean reward per model           — did alignment raise the RM score?
  • win-rate (B beats A)            — head-to-head, the metric that matters
  • mean completion length          — a REWARD-HACKING tripwire: if B's reward rose mostly because it
                                      got longer (not better), suspect length gaming, not alignment.

    uv run python eval.py \
        --model_a Qwen/Qwen3-4B \
        --model_b outputs/qwen3-4b-online-dpo \
        --reward_model ../02-reward-modeling/outputs/qwen3-4b-ultrafeedback-rm

A reward model is itself a proxy (folder 02 §6), so a rising RM score is *necessary, not sufficient*.
Always pair this with reading completions and, ideally, a held-out judge the policy wasn't trained on.
"""

import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def generate(model_path: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    outs = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
            )
        outs.append(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip())
    del model
    torch.cuda.empty_cache()
    return outs


def score(reward_model_path: str, prompts: list[str], completions: list[str]) -> list[float]:
    tok = AutoTokenizer.from_pretrained(reward_model_path)
    rm = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, num_labels=1, dtype=torch.bfloat16, device_map="auto"
    )
    rm.eval()
    scores = []
    for prompt, completion in zip(prompts, completions):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
            tokenize=False,
        )
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(rm.device)
        with torch.no_grad():
            scores.append(rm(**inputs).logits[0, 0].item())
    del rm
    torch.cuda.empty_cache()
    return scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_a", required=True, help="Baseline (e.g. base/SFT model).")
    p.add_argument("--model_b", required=True, help="Candidate (e.g. Online-DPO checkpoint).")
    p.add_argument("--reward_model", required=True, help="Scalar RM used as the judge (folder 02).")
    p.add_argument("--dataset", default="trl-lib/ultrafeedback-prompt")
    p.add_argument("--split", default="test")
    p.add_argument("--num_examples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=256)
    args = p.parse_args()

    ds = load_dataset(args.dataset, split=args.split).select(range(args.num_examples))
    prompts = [row["prompt"][-1]["content"] if isinstance(row["prompt"], list) else row["prompt"]
               for row in ds]

    comp_a = generate(args.model_a, prompts, args.max_new_tokens)
    comp_b = generate(args.model_b, prompts, args.max_new_tokens)
    score_a = score(args.reward_model, prompts, comp_a)
    score_b = score(args.reward_model, prompts, comp_b)

    wins = sum(b > a for a, b in zip(score_a, score_b))
    n = len(prompts)
    len_a = sum(len(c) for c in comp_a) / n
    len_b = sum(len(c) for c in comp_b) / n

    print("\n" + "=" * 70)
    print(f"Eval over {n} prompts | judge RM: {args.reward_model}")
    print("-" * 70)
    print(f"  A ({args.model_a}): mean reward {sum(score_a)/n:+.3f} | mean len {len_a:.0f} chars")
    print(f"  B ({args.model_b}): mean reward {sum(score_b)/n:+.3f} | mean len {len_b:.0f} chars")
    print(f"  win-rate (B beats A): {wins}/{n} = {wins/n:.1%}")
    if len_b > 1.5 * len_a:
        print("  ⚠ B is much longer than A — check whether the reward gain is real or length gaming.")
    print("=" * 70)


if __name__ == "__main__":
    main()
