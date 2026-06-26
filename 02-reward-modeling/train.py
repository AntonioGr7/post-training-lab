"""End-to-end Outcome Reward Model (ORM) training with TRL.

Trains a Bradley-Terry reward model on a real preference dataset
(`trl-lib/ultrafeedback_binarized`). The model is a causal LM repurposed as a *sequence classifier*
with a single scalar output head: given (prompt, response) it returns one number — the reward.
Training maximizes the margin between chosen and rejected responses.

Config-driven via TRL's `TrlParser`:

    uv run python train.py --config configs/smoke.yaml      # ~1-2 min sanity check
    uv run python train.py --config configs/default.yaml     # single A100/H100 headline run
    uv run accelerate launch train.py --config configs/default.yaml   # multi-GPU

Note: `RewardTrainer` loads the model with `AutoModelForSequenceClassification` and forces
`num_labels=1` for you. The classification ("score") head is randomly initialized — that's expected;
it's what we train.
"""

from dataclasses import dataclass, field

from datasets import load_dataset

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class LabArguments:
    dataset_name: str = field(
        default="trl-lib/ultrafeedback_binarized",
        metadata={"help": "Hub preference dataset (chosen/rejected). Conversational or standard."},
    )
    dataset_config: str | None = field(default=None)
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(
        default="test",
        metadata={"help": "Held-out split for preference accuracy. If absent, we carve one out."},
    )
    max_train_samples: int | None = field(
        default=None, metadata={"help": "Subsample train to fit a single-GPU time budget."}
    )
    max_eval_samples: int | None = field(default=None)


def main() -> None:
    parser = TrlParser((LabArguments, RewardConfig, ModelConfig))
    lab_args, training_args, model_args = parser.parse_args_and_config()

    # --- Model loading kwargs (forwarded to AutoModelForSequenceClassification) ----------
    quantization_config = get_quantization_config(model_args)
    dtype = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
        "device_map": get_kbit_device_map() if quantization_config is not None else None,
        "quantization_config": quantization_config,
    }

    # When using PEFT on a causal LM turned classifier, the randomly-initialized reward head
    # ("score") must be trained in full and saved — otherwise you'd freeze it at random init.
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        save = list(getattr(peft_config, "modules_to_save", None) or [])
        if "score" not in save:
            save.append("score")
        peft_config.modules_to_save = save

    # --- Dataset --------------------------------------------------------------------------
    dataset = load_dataset(lab_args.dataset_name, name=lab_args.dataset_config)
    train_dataset = dataset[lab_args.dataset_train_split]

    # Eval split: prefer a real one; otherwise hold out 2% of train.
    if lab_args.dataset_test_split in dataset:
        eval_dataset = dataset[lab_args.dataset_test_split]
    else:
        split = train_dataset.train_test_split(test_size=0.02, seed=training_args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]

    if lab_args.max_train_samples is not None:
        n = min(lab_args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(n))
    if lab_args.max_eval_samples is not None:
        n = min(lab_args.max_eval_samples, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(n))

    print(f"Train pairs: {len(train_dataset):,} | Eval pairs: {len(eval_dataset):,}")

    # --- Train ----------------------------------------------------------------------------
    trainer = RewardTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    metrics = trainer.evaluate()
    print(f"Final eval preference accuracy: {metrics.get('eval_accuracy'):.4f}")

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
