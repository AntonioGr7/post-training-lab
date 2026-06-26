"""End-to-end PPO for RLHF — a deliberately *light* demo.

PPO is the classic RLHF algorithm (InstructGPT). This script runs the canonical, proven
TL;DR-summarization reference setup so you can watch the RLHF loop actually work on a single GPU:
a policy generates summaries, a reward model scores them, and PPO pushes the policy toward higher
reward while a KL penalty keeps it near the SFT reference.

Why this setup instead of our Qwen3 stack? PPO needs FOUR models in memory at once — policy (trained),
value model (trained), reference policy (frozen), reward model (frozen). At 4B scale that blows past
one 80GB GPU. PPO's value here is *conceptual/historical* (see README §"Honest verdict"), so we use
the small, reproducible pythia-1b reference to teach the mechanics without the operational pain.
To wire it to YOUR models, point `sft_model_path` / `reward_model_path` at folders 01 & 02's outputs
(and expect to need LoRA and/or multi-GPU — see the README).

    uv run python train.py --config configs/smoke.yaml       # quick loop check
    uv run python train.py --config configs/default.yaml      # the demo run
    uv run accelerate launch train.py --config configs/default.yaml   # multi-GPU (recommended for PPO)

NOTE: PPOTrainer lives in `trl.experimental.ppo` — TRL classifies PPO as experimental, which is itself
a signal about its current standing.
"""

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl import ModelConfig, TrlParser, get_peft_config
from trl.experimental.ppo import PPOConfig, PPOTrainer


@dataclass
class LabArguments:
    dataset_name: str = field(default="trl-lib/tldr")
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(default="validation")
    max_prompt_length: int = field(
        default=512, metadata={"help": "Truncate prompts to this many tokens (controls rollout cost)."}
    )
    max_train_samples: int | None = field(default=None)


def main() -> None:
    parser = TrlParser((LabArguments, PPOConfig, ModelConfig))
    lab_args, training_args, model_args = parser.parse_args_and_config()

    # Left padding: required so generated tokens are appended at the right edge during rollouts.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    load_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "dtype": dtype,
        "attn_implementation": model_args.attn_implementation,
    }

    # The four models of PPO ----------------------------------------------------------------
    # policy + value are TRAINED; ref + reward are FROZEN. value & reward share the RM backbone
    # (a scalar-head sequence classifier), policy & ref share the SFT backbone (a causal LM).
    policy = AutoModelForCausalLM.from_pretrained(training_args.sft_model_path, **load_kwargs)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, num_labels=1, **load_kwargs
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, num_labels=1, **load_kwargs
    )

    # With PEFT, the reference is the policy with its adapter disabled, so we pass ref_model=None.
    peft_config = get_peft_config(model_args)
    ref_policy = None if peft_config is not None else AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, **load_kwargs
    )

    # Dataset: PPO consumes tokenized PROMPTS (`input_ids`); completions are generated on the fly.
    dataset = load_dataset(lab_args.dataset_name)
    train_dataset = dataset[lab_args.dataset_train_split]
    eval_dataset = (
        dataset[lab_args.dataset_test_split] if lab_args.dataset_test_split in dataset else None
    )
    if lab_args.max_train_samples is not None:
        n = min(lab_args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(n))

    def tokenize(element):
        ids = tokenizer(
            element["prompt"], padding=False, truncation=True, max_length=lab_args.max_prompt_length
        )["input_ids"]
        return {"input_ids": ids}

    train_dataset = train_dataset.map(
        tokenize, batched=True, remove_columns=train_dataset.column_names
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize, batched=True, remove_columns=eval_dataset.column_names
        )

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
