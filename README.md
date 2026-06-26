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
| 03 | [`03-ppo`](03-ppo) | **PPO** — classic RLHF, explained + honest "superseded" verdict | ✅ ready to test |
| 04 | [`04-dpo`](04-dpo) | **DPO & variants** (IPO/KTO/ORPO/SimPO) — the modern default | ✅ ready to test |
| 05 | [`05-grpo`](05-grpo) | **GRPO / RLVR** — verifiable-reward RL for reasoning (DeepSeek-R1 era) | ✅ ready to test |
| 06 | [`06-online-infra`](06-online-infra) | **Online DPO, vLLM generation, reward hacking, eval harness** | ✅ ready to test |
| 07 | [`07-distillation`](07-distillation) | **Knowledge distillation** (GKD / on-policy) — the R1 "cold start" lineage | ✅ ready to test |

The numbering is a suggested learning order; folders remain independent. See [`PLAN.md`](PLAN.md)
for scope, conventions, and build status.

## What modern labs actually run in production (2026)

A course can teach seven techniques and still leave you unsure which ones a real frontier lab *uses*.
So here is the blunt version. This is opinionated and current as of mid-2026 — treat it as the map,
not gospel; each folder's **Honest verdict** section argues the call in detail.

### The actual pipeline

A modern post-training stack is **not** "pick one algorithm." It's a sequence, and the same shape
recurs across DeepSeek, Qwen, Llama, Tulu, and the closed labs:

```
   base model
       │
   ┌───▼────────────────────────────────────────────────────────────┐
   │ 1. SFT cold-start            (01)  — teach format + behavior;    │
   │    often on DISTILLED teacher traces (07) for reasoning seed     │
   └───┬────────────────────────────────────────────────────────────┘
       │
   ┌───▼────────────────────────────────────────────────────────────┐
   │ 2. The RL / alignment stage — pick by the SIGNAL you have:       │
   │                                                                  │
   │   • Verifiable reward (math, code, tool use)?                    │
   │        → GRPO / RLVR  (05)        ← the reasoning-era workhorse   │
   │                                                                  │
   │   • Only taste / preference (no checkable answer)?               │
   │        → DPO  (04)                ← the offline default          │
   │        → Online DPO / RLHF  (06)  ← when staleness bites & you    │
   │                                     can afford on-policy gen      │
   └───┬────────────────────────────────────────────────────────────┘
       │
   ┌───▼────────────────────────────────────────────────────────────┐
   │ 3. Distill to a deployable size   (07)  — ship the small model,  │
   │    not the teacher. The R1-Distill lineage.                      │
   └─────────────────────────────────────────────────────────────────┘
```

In one line: **SFT (often on distilled traces) → GRPO where rewards are verifiable, DPO where they're
not → distill for deployment.** Real recipes mix-and-iterate (e.g. SFT → DPO → a GRPO pass → distill),
but those are the load-bearing stages.

### What's actually used vs. what faded

| Technique | Status in 2026 production | Why |
|---|---|---|
| **SFT** (`01`) | **Universal, mandatory** | Every recipe's first stage; the on-ramp to everything else. The frontier moved to *data curation*, not the algorithm. |
| **GRPO / RLVR** (`05`) | **The reasoning-era default; most active area** | Verifiable rewards + no critic. How R1-class reasoning was trained. Where the research energy is. |
| **DPO** (`04`) | **The default for *preference* alignment** | PPO's objective in closed form — no RM, no rollouts, no value model. The stable workhorse when there's no verifier. |
| **Distillation** (`07`) | **Production-essential** | How a capable model becomes a *deployable* one. The offline cold-start (SFT-on-traces) is the scalable form; GKD is the on-policy upgrade. |
| **Online DPO / RLHF** (`06`) | **Used, but not the default** | On-policy preference training; operationally heavy (3 models + generation). Reach for it when offline staleness measurably costs you. |
| **Process Reward Models** (`02`) | **Live frontier** — reasoning, search, reranking | Standalone *outcome* RMs largely faded for chat, but PRMs are an active research/production tool for reasoning. |
| **PPO** (`03`) | ⚰️ **Largely superseded** | DPO deleted its preference use case; GRPO dropped its value model. Taught in full because the *concepts* (KL leash, trust region, advantage) live on in everything after it. |
| **Standalone outcome RMs for chat** (`02`) | ⚰️ **Mostly folded away** | DPO absorbs the preference signal directly; explicit chat RMs survive mainly as *scorers* inside online RLHF (`06`) and as a teaching foundation. |
| **Actor-critic value models** | ⚰️ **Dropped** | GRPO/RLOO replace the learned critic with a group/leave-one-out baseline — less memory, fewer moving parts. |

The throughline behind every row: the field keeps **deleting machinery** (the critic, the reward
model, the RL loop itself) while keeping the *signal*. PPO had four models; GRPO has one policy + a
verifier. That simplification trend is the story of modern post-training — and the reason the "honest
verdict" in each folder matters as much as the how-to.

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
