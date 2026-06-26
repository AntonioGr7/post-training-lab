"""Did distillation work? Put the small student next to the big teacher and read the answers.

    uv run python generate.py --student outputs/qwen3-0.6b-gkd --teacher Qwen/Qwen3-4B

The whole point of distillation is a SMALL model that punches above its size. So the qualitative test
is comparative: for each prompt, see the teacher's answer, the distilled student's answer, and — if
you pass --base — the *undistilled* student of the same size, to isolate what distillation bought
(student-after vs student-before, at equal inference cost).

Loss/JSD curves say the optimization ran; this says the student actually absorbed the teacher's
behavior — coherent, instruction-following, and knowing when to STOP (emits EOS).
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "Explain why the sky is blue to a 10-year-old, in two sentences.",
    "Write a haiku about knowledge distillation.",
    "A train leaves at 3pm going 60mph. Another leaves at 4pm going 80mph in the same direction. "
    "When does the second catch the first? Show your reasoning.",
]


def load(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tok, model


def answer(tok, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature or None,
            top_p=0.9,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True, help="Distilled student checkpoint (output_dir).")
    p.add_argument("--teacher", default=None, help="Teacher model for side-by-side comparison.")
    p.add_argument("--base", default=None, help="Undistilled student (same size) — the before/after.")
    p.add_argument("--prompt", default=None, help="Single prompt; omit to run the default set.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    # Label → loaded (tok, model). Order: teacher first (the target), then before/after students.
    models = {}
    if args.teacher:
        models["TEACHER " + args.teacher] = load(args.teacher)
    if args.base:
        models["STUDENT (base) " + args.base] = load(args.base)
    models["STUDENT (distilled) " + args.student] = load(args.student)

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    for prompt in prompts:
        print("\n" + "=" * 90)
        print(f"PROMPT: {prompt}")
        for label, (tok, model) in models.items():
            reply = answer(tok, model, prompt, args.max_new_tokens, args.temperature)
            print(f"\n[{label}]\n{reply}")


if __name__ == "__main__":
    main()
