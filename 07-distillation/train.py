"""End-to-end Generalized Knowledge Distillation (GKD) — compress a big teacher into a small student.

Distillation transfers what a large model *knows* into a smaller, cheaper one. The naive way is
SUPERVISED KD: take the teacher's outputs as fixed targets and SFT the student on them. That has the
same blind spot offline RL did (folders 03/04/06): the targets are off-policy. The student is graded
on sequences IT would never produce, then at inference must walk a path it never practiced — the
train/inference distribution mismatch (exposure bias).

GKD (Agarwal et al. 2023, "On-Policy Distillation") fixes this exactly like online RL fixed it for
alignment: train the student on ITS OWN on-policy generations, scored token-by-token by the teacher.

  each step:  with prob `lmbda`, the student GENERATES a completion (on-policy);
              the teacher provides its full token-level distribution over that sequence;
              minimise a generalized Jensen-Shannon divergence (interpolated by `beta`) between
              student and teacher distributions on every token.

Two dials carry the whole method:
  • lmbda — fraction of on-policy (student-generated) data. 0.0 = pure supervised KD on the dataset's
            targets; 1.0 = pure on-policy. The paper's headline: higher lmbda wins.
  • beta  — interpolates the divergence. 0.0 ≈ forward KL (mode-covering, "mimic everything the
            teacher might say"); 1.0 ≈ reverse KL (mode-seeking, "commit to the teacher's best mode"
            — better when the small student lacks the capacity to cover the full distribution).

    uv run python train.py --config configs/smoke.yaml                  # quick loop check (tiny T+S)
    uv run accelerate launch train.py --config configs/default.yaml     # headline (Qwen3-4B → 0.6B)

NOTE: GKDTrainer lives in `trl.experimental.gkd` (TRL flags it 🧪). It's a wrapper around SFTTrainer
that adds a teacher model and the on-policy generation + JSD loss.
"""

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import ModelConfig, TrlParser, get_peft_config
from trl.experimental.gkd import GKDConfig, GKDTrainer


@dataclass
class LabArguments:
    dataset_name: str = field(
        default="allenai/tulu-3-sft-mixture",
        metadata={"help": "Chat dataset (messages format). Prompts drive on-policy generation; the "
                          "assistant turns are the off-policy targets used when lmbda < 1."},
    )
    dataset_train_split: str = field(default="train")
    dataset_test_split: str | None = field(default=None)
    max_train_samples: int | None = field(default=None)
    max_eval_samples: int | None = field(default=None)


def main() -> None:
    # GKDConfig carries the teacher (`teacher_model_name_or_path`) + GKD dials (lmbda/beta/seq_kd)
    # on top of all of SFTConfig's fields. ModelConfig configures the STUDENT (the model we train).
    parser = TrlParser((LabArguments, GKDConfig, ModelConfig))
    lab_args, training_args, model_args = parser.parse_args_and_config()

    # `dtype` is transformers>=5; older snapshots used `torch_dtype`. Read defensively.
    dtype = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
    }
    # The student is instantiated from `model_name_or_path` via SFTConfig.model_init_kwargs;
    # the teacher from `teacher_model_name_or_path` via teacher_model_init_kwargs. Same dtype/attn.
    training_args.model_init_kwargs = init_kwargs
    training_args.teacher_model_init_kwargs = init_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # IMPORTANT: teacher and student must share a tokenizer/vocab — GKD aligns their per-token
    # distributions position-by-position. Distilling across model families means a vocab mismatch;
    # the clean path is a teacher and student from the SAME family (here: Qwen3 4B → 0.6B).
    peft_config = get_peft_config(model_args)

    dataset = load_dataset(lab_args.dataset_name)
    train_dataset = dataset[lab_args.dataset_train_split]
    eval_dataset = dataset.get(lab_args.dataset_test_split) if lab_args.dataset_test_split else None

    if lab_args.max_train_samples is not None:
        n = min(lab_args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(n))
    if eval_dataset is not None and lab_args.max_eval_samples is not None:
        n = min(lab_args.max_eval_samples, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(n))

    print(
        f"Student: {model_args.model_name_or_path} | Teacher: "
        f"{training_args.teacher_model_name_or_path}\n"
        f"Train: {len(train_dataset):,} | lmbda={training_args.lmbda} (on-policy frac) "
        f"beta={training_args.beta} seq_kd={training_args.seq_kd}"
    )

    trainer = GKDTrainer(
        model=model_args.model_name_or_path,                       # the STUDENT (trained)
        teacher_model=training_args.teacher_model_name_or_path,    # the TEACHER (frozen)
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
