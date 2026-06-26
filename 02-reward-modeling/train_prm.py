"""Process Reward Model (PRM) training — the advanced section.

Where the outcome RM (`train.py`) scores a *whole* response with one number, a PRM scores *each
reasoning step* as correct/incorrect. It is a token-classification model trained on stepwise
supervision: a `prompt`, a list of `completions` (steps), and a list of `labels` (one per step).

We use `trl-lib/math_shepherd` (math reasoning, step-level labels) and Qwen3-0.6B-Base — a PRM does
not need to be large to be useful, and step-level data is expensive, so PRMs are typically small.

    uv run python train_prm.py --config configs/prm.yaml
    # quick smoke: append CLI overrides
    uv run python train_prm.py --config configs/prm.yaml --max_train_samples 256 --max_steps 5

NOTE: `PRMTrainer` lives in `trl.experimental` and is subject to change — it's a research API.
Unlike the other trainers it wants an already-instantiated `AutoModelForTokenClassification`.
"""

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from trl import ModelConfig, TrlParser, get_peft_config
from trl.experimental.prm import PRMConfig, PRMTrainer


@dataclass
class LabArguments:
    dataset_name: str = field(default="trl-lib/math_shepherd")
    dataset_train_split: str = field(default="train")
    max_train_samples: int | None = field(default=None)


def main() -> None:
    parser = TrlParser((LabArguments, PRMConfig, ModelConfig))
    lab_args, training_args, model_args = parser.parse_args_and_config()

    dtype = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    # num_labels=2 → each step is classified as incorrect (0) or correct (1).
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,
        dtype=dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    train_dataset = load_dataset(lab_args.dataset_name, split=lab_args.dataset_train_split)
    if lab_args.max_train_samples is not None:
        n = min(lab_args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(n))

    print(f"Train examples (multi-step solutions): {len(train_dataset):,}")

    trainer = PRMTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
