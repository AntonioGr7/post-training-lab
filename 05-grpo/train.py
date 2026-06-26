"""End-to-end GRPO / RLVR on GSM8K — the DeepSeek-R1-style reasoning recipe, in TRL.

GRPO (Group Relative Policy Optimization) is online RL with two ideas that fix PPO's pain (folder 03):
  1. **No value model.** Instead of a learned critic estimating each token's advantage, GRPO samples
     a GROUP of G completions per prompt and uses the group's mean reward as the baseline:
        Â_i = (r_i − mean(r)) / std(r).
     Above-average completions get pushed up, below-average down. This deletes PPO's 4th model.
  2. **Verifiable rewards (RLVR).** The reward comes from a programmatic VERIFIER (is the math answer
     correct?), not a learned reward model — so there is nothing to over-optimize against.

So GRPO holds at most TWO models (policy + optional frozen reference for the KL term), generates
on-policy like PPO, but is far lighter and more stable. Generation is the bottleneck → we use vLLM.

    uv run python train.py --config configs/smoke.yaml                       # quick loop check (no vLLM)
    uv run accelerate launch train.py --config configs/default.yaml          # headline run (vLLM colocate)

Reward functions live in `rewards.py` (correctness + format). The dataset must expose a `prompt`
column; any other column (here `ground_truth`) is forwarded to the reward functions as a kwarg.
"""

from dataclasses import dataclass, field

from datasets import load_dataset

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    TrlParser,
    get_peft_config,
)

from rewards import SYSTEM_PROMPT, correctness_reward, format_reward, gsm8k_gold


@dataclass
class LabArguments:
    dataset_name: str = field(default="openai/gsm8k")
    dataset_config: str | None = field(default="main")
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(default="test")
    max_train_samples: int | None = field(
        default=None, metadata={"help": "Subsample train prompts to fit a single-GPU time budget."}
    )


def main() -> None:
    parser = TrlParser((LabArguments, GRPOConfig, ModelConfig))
    lab_args, training_args, model_args = parser.parse_args_and_config()

    # GRPOConfig.model_init_kwargs are forwarded to <Arch>.from_pretrained when `model` is a string.
    dtype = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
    }

    peft_config = get_peft_config(model_args)

    # --- Dataset --------------------------------------------------------------------------
    # GRPO consumes PROMPTS only — completions are generated on-policy each step. We build a
    # conversational prompt (system + question) and carry the gold answer in `ground_truth`, which
    # the trainer passes straight through to the reward functions.
    dataset = load_dataset(lab_args.dataset_name, name=lab_args.dataset_config)
    train_dataset = dataset[lab_args.dataset_train_split]
    eval_dataset = dataset.get(lab_args.dataset_test_split)

    def to_prompt(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "ground_truth": gsm8k_gold(example["answer"]),
        }

    train_dataset = train_dataset.map(to_prompt, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(to_prompt, remove_columns=eval_dataset.column_names)

    if lab_args.max_train_samples is not None:
        n = min(lab_args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(n))

    print(f"Train prompts: {len(train_dataset):,} | num_generations(G)={training_args.num_generations}")

    # --- Train ----------------------------------------------------------------------------
    # Multiple reward functions are SUMMED (or weighted via GRPOConfig.reward_weights). Correctness
    # is the real signal; format is a small shaping term — keep its scale low (see rewards.py).
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[correctness_reward, format_reward],
        args=training_args,
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
