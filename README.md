# Post-Training Lab

A **hands-on, SOTA course on post-training and reinforcement learning for LLMs**, for AI
practitioners who want to specialize. Learn-by-doing: we don't reimplement algorithms from scratch —
we teach the **theory, the when/why/how, and the practitioner judgment** for each technique using the
real tools people ship today (Hugging Face **TRL ≥ 1.0**, **transformers ≥ 5**, PEFT, vLLM).

## How this course works

- **Every folder is an independent project.** Copy any one out and it stands alone — its own
  `uv` environment, lockfile, lesson, code, and configs.
- **Each runs on a single A100/H100 (80 GB)**, with a tiny `smoke` config for sanity checks anywhere.
- **Each folder has:** a practitioner-deep `README.md` lesson, a runnable end-to-end example on a
  **real dataset**, and configs. Training code is `torchrun`/`accelerate`-ready for multi-GPU.

### Running any folder
```bash
cd 01-sft
uv sync
uv run python train.py --config configs/smoke.yaml      # quick pipeline check
uv run python train.py --config configs/default.yaml     # the real run
```

## Curriculum

| # | Folder | Technique | Status |
|---|---|---|---|
| 01 | [`01-sft`](01-sft) | **Supervised Fine-Tuning** — the foundation; instruction-tune a base model | ✅ ready to test |
| 02 | [`02-reward-modeling`](02-reward-modeling) | **Reward Modeling** — Bradley-Terry + Process RMs | ✅ ready to test |
| 03 | `03-ppo` | **PPO** — classic RLHF, explained + honest "superseded" verdict | ☐ planned |
| 04 | `04-dpo` | **DPO & variants** (IPO/KTO/ORPO/SimPO) — the modern default | ☐ planned |
| 05 | `05-grpo` | **GRPO / RLVR** — verifiable-reward RL for reasoning (DeepSeek-R1 era) | ☐ planned |
| 06 | `06-online-infra` | **Online DPO, vLLM generation, reward hacking, eval harness** | ☐ planned |
| 07 | `07-distillation` | **Knowledge distillation** (GKD / on-policy) — the R1 "cold start" lineage | ☐ planned |

The numbering is a suggested learning order; folders remain independent. See [`PLAN.md`](PLAN.md)
for scope, conventions, and build status.

## Taxonomy map: where each technique sits

The course is a *learning progression*, but every technique is also one of TRL's trainers. This map
bridges the two — find any TRL trainer here and see which folder teaches the idea behind it. The
**online vs offline** split (do you generate fresh samples from the current policy during training,
or train on a fixed dataset?) is the single most important conceptual axis in post-training RL, and
we return to it in every folder.

Flags follow TRL: ⚡️ = vLLM-accelerated · 🧪 = experimental in TRL (a production-readiness signal).

| TRL category | Trainer | Taught in | Notes |
|---|---|---|---|
| **Offline** | `SFTTrainer` | `01-sft` | the foundation; behavioral cloning |
| | `DPOTrainer` | `04-dpo` | the stable, modern preference default |
| | `KTOTrainer`🧪 `ORPOTrainer`🧪 `CPOTrainer`🧪 `BCOTrainer`🧪 | `04-dpo` | DPO-variant landscape (all experimental → secondary) |
| **Reward modeling** | `RewardTrainer` | `02-reward-modeling` | Bradley-Terry outcome RM |
| | `PRMTrainer`🧪 | `02-reward-modeling` | process reward models (advanced section) |
| **Online** | `GRPOTrainer`⚡️ | `05-grpo` | RLVR / reasoning flagship |
| | `RLOOTrainer`⚡️ | `05-grpo` / `06` | lighter-weight on-policy sibling |
| | `OnlineDPOTrainer`🧪⚡️ | `06-online-infra` | online preference optimization |
| | `PPOTrainer`🧪 | `03-ppo` | classic RLHF — explained, **superseded** verdict |
| | `NashMDTrainer`🧪⚡️ `XPOTrainer`🧪⚡️ | `06-online-infra` | frontier online methods (map/mention) |
| **Distillation** | `GKDTrainer`🧪 | `07-distillation` | on-policy / generalized KD |
| | `MiniLLMTrainer`🧪 | `07-distillation` | reverse-KL variant (mention) |

> **Reading the flags:** that PPO and *all* the DPO-variants are 🧪 in TRL is itself instructive —
> it corroborates the course's verdicts (PPO is superseded; DPO is the stable default, its variants
> secondary). We surface these honestly rather than hiding them.

## Prerequisites
- Comfort with PyTorch and the Hugging Face ecosystem.
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- A single A100/H100 (80 GB) for the headline runs; a Hugging Face account/token for models & data.
