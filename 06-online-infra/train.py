"""End-to-end Online DPO — DPO's loss, but the preference pairs are generated ON-POLICY each step.

This closes the loop that offline DPO (folder 04) left open. Offline DPO trains on a *fixed* pair
dataset sampled from some other model; as the policy drifts during training, those pairs go stale and
off-policy. Online DPO (a.k.a. OAIF — Online AI Feedback) fixes this:

  each step:  sample TWO completions from the CURRENT policy → score both (reward model / judge)
              → the higher-scored one is `chosen`, the other `rejected` → take a DPO step on that pair

So you get DPO's simplicity (no value model, no PPO clip) with PPO's on-policy freshness (you train on
what the model produces *now*). The cost: generation is back in the loop → vLLM matters (see README).

The scorer here is a REWARD MODEL (folder 02's output, or any scalar RM). Swapping it for an LLM
judge is a one-line change conceptually — see the README. Because we optimize a *learned* proxy
on-policy, **reward hacking is back on the table** (folder 02 §6) — the README's mitigation section
is the point of this folder.

    uv run python train.py --config configs/smoke.yaml                          # quick loop check (no vLLM)
    uv run accelerate launch train.py --config configs/default.yaml             # headline run (vLLM colocate)

NOTE: OnlineDPOTrainer lives in `trl.experimental.online_dpo` — TRL flags the whole online-preference
family (Online DPO, NashMD, XPO) as experimental.
"""

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

from trl import ModelConfig, TrlParser, get_peft_config
from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer


@dataclass
class LabArguments:
    dataset_name: str = field(
        default="trl-lib/ultrafeedback-prompt",
        metadata={"help": "PROMPT-ONLY dataset — completions are generated on-policy each step."},
    )
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(default="test")
    max_train_samples: int | None = field(default=None)
    max_eval_samples: int | None = field(default=None)


def main() -> None:
    parser = TrlParser((LabArguments, OnlineDPOConfig, ModelConfig))
    lab_args, training_args, model_args = parser.parse_args_and_config()

    dtype = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
    }

    # The scorer: a frozen scalar reward model. By default we point at folder 02's RM (build it first)
    # so the course ties together: 02 trained this RM on UltraFeedback prefs → here it judges the
    # policy's on-policy generations. `reward_model_path` is set in the config; swap a released RM if
    # you haven't run folder 02. (For an LLM-as-judge instead, pass a PairwiseJudge — see README.)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        num_labels=1,
        dtype=dtype,
        trust_remote_code=model_args.trust_remote_code,
    )

    peft_config = get_peft_config(model_args)

    # Prompt-only dataset (no chosen/rejected — those are produced on-policy).
    dataset = load_dataset(lab_args.dataset_name)
    train_dataset = dataset[lab_args.dataset_train_split]
    eval_dataset = dataset.get(lab_args.dataset_test_split)

    if lab_args.max_train_samples is not None:
        n = min(lab_args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(n))
    if eval_dataset is not None and lab_args.max_eval_samples is not None:
        n = min(lab_args.max_eval_samples, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(n))

    print(f"Train prompts: {len(train_dataset):,} | scorer: {training_args.reward_model_path}")

    trainer = OnlineDPOTrainer(
        model=model_args.model_name_or_path,
        ref_model=None,                      # created from the policy (frozen); adapter-disabled with LoRA
        reward_funcs=reward_model,           # could also be a path str, a custom callable, or a judge
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
