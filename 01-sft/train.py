"""End-to-end Supervised Fine-Tuning (SFT) with TRL.

This script instruction-tunes a *base* language model on a real conversational dataset
(`allenai/tulu-3-sft-mixture`) using TRL's `SFTTrainer`. It is intentionally thin: TRL does the
heavy lifting (chat-template application, packing, loss masking). Our job is to wire up the config
and the dataset the way a practitioner would.

It is config-driven via TRL's `TrlParser`, so everything is set in a YAML file:

    # Single GPU (or the smoke test):
    uv run python train.py --config configs/smoke.yaml

    # Single A100/H100 headline run:
    uv run python train.py --config configs/default.yaml

    # Multi-GPU — same script, no code changes (DDP / FSDP via accelerate or torchrun):
    uv run accelerate launch train.py --config configs/default.yaml
    uv run torchrun --nproc_per_node=4 train.py --config configs/default.yaml

You can override any field on the CLI, e.g. `... --config configs/default.yaml --learning_rate 1e-5`.
"""

from dataclasses import dataclass, field

from datasets import load_dataset

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class LabArguments:
    """SFT-lab–specific knobs that aren't already covered by SFTConfig / ModelConfig.

    Kept deliberately small. Dataset *training* behaviour (packing, max_length, assistant-only
    loss, chat template, ...) lives in SFTConfig; model/LoRA/quantization lives in ModelConfig.
    """

    dataset_name: str = field(
        default="allenai/tulu-3-sft-mixture",
        metadata={"help": "Hub dataset id. Must be in a conversational ('messages') format."},
    )
    dataset_config: str | None = field(
        default=None, metadata={"help": "Dataset config/subset name, if any."}
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Split to train on."})
    max_train_samples: int | None = field(
        default=None,
        metadata={"help": "Subsample the train split to this many examples (None = use all). "
                          "We subsample so the headline run fits a single-GPU time budget."},
    )
    eval_ratio: float = field(
        default=0.0,
        metadata={"help": "Fraction of the (subsampled) train split to hold out for evaluation. "
                          "tulu-3 ships only a 'train' split, so we carve eval out of it."},
    )


def main() -> None:
    parser = TrlParser((LabArguments, SFTConfig, ModelConfig))
    lab_args, training_args, model_args = parser.parse_args_and_config()

    # --- Model loading kwargs -------------------------------------------------------------
    # We forward dtype / attention impl / quantization to the trainer, which calls
    # `from_pretrained` for us. `dtype` (formerly `torch_dtype`) was renamed in transformers v5;
    # we read whichever attribute this TRL build exposes.
    quantization_config = get_quantization_config(model_args)
    dtype = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
        # QLoRA needs the base weights sharded onto the visible device(s).
        "device_map": get_kbit_device_map() if quantization_config is not None else None,
        "quantization_config": quantization_config,
    }

    # --- Dataset --------------------------------------------------------------------------
    # tulu-3-sft-mixture is already in conversational format: each row has a `messages` list of
    # {role, content} dicts. SFTTrainer detects this and applies the chat template automatically —
    # we do NOT pre-format anything by hand.
    dataset = load_dataset(
        lab_args.dataset_name, name=lab_args.dataset_config, split=lab_args.dataset_train_split
    )
    if lab_args.max_train_samples is not None:
        n = min(lab_args.max_train_samples, len(dataset))
        dataset = dataset.shuffle(seed=training_args.seed).select(range(n))

    eval_dataset = None
    if lab_args.eval_ratio > 0:
        split = dataset.train_test_split(test_size=lab_args.eval_ratio, seed=training_args.seed)
        dataset, eval_dataset = split["train"], split["test"]

    print(f"Train examples: {len(dataset):,}"
          + (f" | Eval examples: {len(eval_dataset):,}" if eval_dataset is not None else ""))

    # --- Train ----------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),  # None unless `use_peft: true` in the config
    )

    trainer.train()

    # Saves model + processing class (tokenizer/chat template) so it reloads cleanly.
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
