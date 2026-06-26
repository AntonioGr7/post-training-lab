"""End-to-end Direct Preference Optimization (DPO) with TRL.

DPO is "PPO's RLHF objective (folder 03) solved in closed form": the optimal policy for the
KL-constrained reward-maximization objective has an analytic form, so we can train *directly on
preference pairs* with a simple classification-style loss — no reward model, no value model, no
rollouts, no KL babysitting. It is an OFFLINE method (a fixed pair dataset; no generation in the
loop) — the foil to PPO's online loop.

One script, many variants: `loss_type` is a plain config field, so the same code runs vanilla DPO
(`sigmoid`), IPO (`ipo`), SimPO-style length-normalized DPO (`sigmoid_norm`), `hinge`, and more.
See the README's decision table for when to reach for each.

    uv run python train.py --config configs/smoke.yaml       # ~1-2 min sanity check
    uv run python train.py --config configs/default.yaml      # single A100/H100 headline run
    uv run accelerate launch train.py --config configs/default.yaml   # multi-GPU

The reference model:
  - Full fine-tune (default config): pass `ref_model=None` and the trainer freezes a copy of the
    INITIAL policy as the reference. So policy + ref = TWO 4B models resident at once (~58 GB
    static in bf16) — fits one 80 GB GPU with gradient checkpointing + a modest `max_length`.
  - LoRA (uncomment `use_peft` in the config): the reference is just the policy with its adapter
    disabled, so there is no second model — `ref_model=None` still does the right thing.
"""

from dataclasses import dataclass, field

from datasets import load_dataset

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class LabArguments:
    dataset_name: str = field(
        default="trl-lib/ultrafeedback_binarized",
        metadata={"help": "Hub preference dataset (prompt/chosen/rejected). Conversational or standard."},
    )
    dataset_config: str | None = field(default=None)
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(
        default="test",
        metadata={"help": "Held-out split for reward accuracy/margins. If absent, we carve one out."},
    )
    max_train_samples: int | None = field(
        default=None, metadata={"help": "Subsample train to fit a single-GPU time budget."}
    )
    max_eval_samples: int | None = field(default=None)


def main() -> None:
    parser = TrlParser((LabArguments, DPOConfig, ModelConfig))
    lab_args, training_args, model_args = parser.parse_args_and_config()

    # --- Model loading kwargs (forwarded to <Arch>.from_pretrained by DPOTrainer) ---------
    # DPOConfig.model_init_kwargs are passed straight to from_pretrained when `model` is a string.
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

    # With LoRA there is no separate reference model: the trainer disables the adapter to get the
    # reference log-probs (so ref_model stays None). With full-FT, ref_model=None makes the trainer
    # freeze a copy of the initial policy as the reference.
    peft_config = get_peft_config(model_args)

    # --- Dataset --------------------------------------------------------------------------
    # Same UltraFeedback preference pairs as folder 02 — on purpose. Contrast the two recipes:
    # 02 trains an explicit reward model on these prefs; here we optimize the policy on them directly.
    dataset = load_dataset(lab_args.dataset_name, name=lab_args.dataset_config)
    train_dataset = dataset[lab_args.dataset_train_split]

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

    print(
        f"Train pairs: {len(train_dataset):,} | Eval pairs: {len(eval_dataset):,} "
        f"| loss_type={training_args.loss_type} | beta={training_args.beta}"
    )

    # --- Train ----------------------------------------------------------------------------
    trainer = DPOTrainer(
        model=model_args.model_name_or_path,
        ref_model=None,           # full-FT: frozen copy of init policy | LoRA: adapter disabled
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    metrics = trainer.evaluate()
    # rewards/accuracies: fraction of eval pairs where the implicit reward ranks chosen > rejected.
    acc = metrics.get("eval_rewards/accuracies")
    margin = metrics.get("eval_rewards/margins")
    print(f"Final eval reward accuracy: {acc:.4f} | reward margin: {margin:.4f}")

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
