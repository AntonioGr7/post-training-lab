# 01 · Supervised Fine-Tuning (SFT)

> **The foundation of all post-training.** Before a model can be aligned with preferences (DPO),
> trained with rewards (PPO/GRPO), or anything else in this course, it almost always passes through
> SFT first. Get this right and everything downstream gets easier; get it wrong and no amount of RL
> will save you.

**You will:** instruction-tune a base model (`Qwen3-4B-Base`) on a real, SOTA instruction dataset
(`allenai/tulu-3-sft-mixture`) end-to-end on a single A100/H100 using Hugging Face **TRL**, then
chat with the result.

---

## Table of contents
1. [Where SFT sits in the pipeline](#1-where-sft-sits-in-the-pipeline)
2. [What SFT actually optimizes](#2-what-sft-actually-optimizes)
3. [Data is the whole game: formats & chat templates](#3-data-is-the-whole-game-formats--chat-templates)
4. [Loss masking: full vs completion-only vs assistant-only](#4-loss-masking-full-vs-completion-only-vs-assistant-only)
5. [Packing (and why it needs FlashAttention)](#5-packing-and-why-it-needs-flashattention)
6. [Full fine-tune vs LoRA vs QLoRA](#6-full-fine-tune-vs-lora-vs-qlora)
7. [Fitting it on one 80 GB GPU: the memory arithmetic](#7-fitting-it-on-one-80gb-gpu-the-memory-arithmetic)
8. [Hyperparameters that actually matter](#8-hyperparameters-that-actually-matter)
9. [How to know it worked (evaluation)](#9-how-to-know-it-worked-evaluation)
10. [When SFT is enough — and when it isn't](#10-when-sft-is-enough--and-when-it-isnt)
11. [Honest verdict: is SFT still SOTA?](#11-honest-verdict-is-sft-still-sota)
12. [Run it](#12-run-it)
13. [Common failure modes](#13-common-failure-modes)
14. [Exercises](#14-exercises)
15. [References](#15-references)

---

## 1. Where SFT sits in the pipeline

A modern instruct/chat model is built in stages:

```
   Pretraining            Post-training
 ┌──────────────┐   ┌─────────────────────────────────────────────────┐
 │  Base model  │ → │  SFT  →  Preference opt. (DPO) / RL (PPO, GRPO)  │ → Aligned model
 └──────────────┘   └─────────────────────────────────────────────────┘
   next-token on        imitate good          optimize a preference /
   trillions of         demonstrations        reward signal beyond
   web tokens           (this folder)         imitation
```

A **base model** is a brilliant autocomplete engine with no notion of "a user asked me something
and I should help." It has the *knowledge* but not the *behavior*. **SFT teaches behavior by
imitation**: you show it thousands of `(instruction → good response)` demonstrations and train it to
reproduce them. After SFT the model follows instructions, adopts a chat format, and knows when to
stop talking.

Everything later in this course (folders `02`–`06`) assumes you start from an SFT'd model. SFT is
the cheapest, highest-leverage step in post-training — and the one people most often rush.

> **Mental model:** SFT is *behavioral cloning*. It can only teach the model to imitate responses
> you can already write down. It cannot teach it to be *better than your demonstrations*. That gap
> is exactly what preference optimization and RL exist to close — and why this course continues past
> this folder.

---

## 2. What SFT actually optimizes

SFT is plain **next-token prediction** (causal language modeling), restricted to the tokens you
care about. For a target sequence `y = (y_1, …, y_T)` the loss is token-level cross-entropy:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t})$$

This is the **same objective as pretraining**. The only differences are:

1. **The data** — curated instruction/response demonstrations instead of raw web text.
2. **The masking** — you typically compute the loss only on the *response* tokens, not the prompt
   (see §4).

Two consequences worth internalizing:

- **Teacher forcing.** During training the model always conditions on the *ground-truth* previous
  tokens (`y_{<t}`), never on its own samples. This is what makes SFT stable and cheap — no
  generation loop, no reward model, no rollouts. It's also the root of *exposure bias*: at inference
  the model conditions on *its own* (possibly wrong) tokens, a distribution it never saw in training.
  RL methods (later folders) train on the model's own samples and so don't have this gap.
- **It's MLE.** SFT is maximum-likelihood estimation of your demonstration distribution. It pushes
  probability mass onto "responses that look like the dataset." It has no concept of *how wrong* a
  wrong answer is — every non-target token is equally penalized. (Preference methods add that notion
  of relative quality.)

> 💡 TRL's `SFTTrainer` defaults to `loss_type="chunked_nll"` — mathematically identical to standard
> cross-entropy, but it computes the `lm_head` projection and cross-entropy in chunks so peak
> activation memory doesn't scale with `vocab_size × seq_len`. Free memory win; leave it on.

---

## 3. Data is the whole game: formats & chat templates

> If you remember one thing from this folder: **in SFT, data quality and formatting dominate
> everything else.** A smaller, cleaner dataset routinely beats a larger, noisier one. Tulu-3's main
> contribution was a *carefully curated mixture*, not a new algorithm.

### Dataset formats TRL understands

TRL auto-detects these and applies the chat template for you — **do not pre-render strings by hand**:

| Format | Shape | Use when |
|---|---|---|
| Conversational language-modeling | `{"messages": [{role, content}, …]}` | multi-turn chat SFT (**our case**) |
| Conversational prompt-completion | `{"prompt": [...], "completion": [...]}` | you want loss only on the completion |
| Standard language-modeling | `{"text": "..."}` | you've already rendered the template yourself |
| Standard prompt-completion | `{"prompt": "...", "completion": "..."}` | non-chat, single-turn |

`allenai/tulu-3-sft-mixture` is **conversational** (`messages`), so `SFTTrainer` applies the chat
template and we touch nothing.

### Chat templates — the silent killer

A **chat template** is a Jinja template (stored on the tokenizer) that turns a list of messages into
a single string with role markers and special tokens, e.g.:

```
<|im_start|>user
What color is the sky?<|im_end|>
<|im_start|>assistant
It is blue.<|im_end|>
```

This is where most SFT bugs live:

- **Train/inference template mismatch.** If you train with one template and serve with another, the
  model sees out-of-distribution formatting and quality collapses. *Rule: train and serve with the
  exact same template.*
- **Base models often have no (usable) chat template.** Their tokenizer either lacks one or carries a
  placeholder. You must *give* them one during SFT. In this example we instruction-tune
  `Qwen3-4B-Base` and set `chat_template_path: HuggingFaceTB/SmolLM3-3B` — TRL clones that template
  onto the tokenizer, resizes embeddings for any new special tokens, and aligns the EOS token. (We
  borrow SmolLM3's template specifically because it includes the `{% generation %}` tags that
  assistant-only loss needs — see §4.)
- **The EOS token must match the template.** If the model never learns to emit the template's
  end-of-turn token, it won't stop generating at inference (the classic "model rambles forever" bug).
  For Qwen-family instruct templates that token is `<|im_end|>`.

> **Base vs Instruct, which do I start from?** For *learning* SFT, start from a **base** model — you
> see the behavior appear, and there's no prior alignment to fight. For *production*, teams sometimes
> continue-SFT from an existing **instruct** model to add a capability; that's easier (template
> already present) but you risk overwriting the vendor's alignment.

---

## 4. Loss masking: full vs completion-only vs assistant-only

You almost never want to train on *every* token. Three regimes:

| Mode | Loss computed on | TRL setting | When |
|---|---|---|---|
| Full sequence | prompt **and** response | (default for `{"text": ...}`) | rarely; only for pure continued pretraining |
| Completion-only | the completion, not the prompt | `completion_only_loss=True` (default for prompt-completion data) | single-turn instruction tuning |
| **Assistant-only** | **only assistant turns** in a multi-turn chat | `assistant_only_loss=True` | **multi-turn chat SFT (our case)** |

**Why mask the prompt?** Training the model to predict the *user's* question is wasted capacity at
best and harmful at worst — you don't want it learning to generate user turns. You want it to learn
the *mapping* from instruction to response. Masking the prompt focuses the gradient there.

**Why assistant-only for multi-turn?** A conversation has multiple user and assistant turns. You
want loss on *every assistant turn* but on *no user turn*. `assistant_only_loss=True` does exactly
this.

> ⚠️ **Gotcha:** assistant-only loss requires the chat template to mark assistant spans with
> `{% generation %} … {% endgeneration %}` tags. Most stock templates don't. TRL auto-patches known
> families (e.g. Qwen3); for others you must use a template that has them. This is the *real reason*
> we clone `HuggingFaceTB/SmolLM3-3B`'s template here — it ships those tags. If you swap models/
> templates and assistant-only loss silently trains on everything (or errors), this is why.

---

## 5. Packing (and why it needs FlashAttention)

Instruction examples vary wildly in length. If you pad every example to `max_length`, most of your
GPU's FLOPs go into computing attention over `<pad>` tokens — pure waste.

**Packing** concatenates multiple short examples into one full-length sequence, so (almost) every
token is real. On instruction data this is commonly a **2–5× throughput win**. Enable with
`packing=True`.

**The catch:** once you pack example A and example B into one sequence, you must stop tokens in B
from attending to tokens in A — otherwise you've created spurious cross-example dependencies that
corrupt training. Correct packing therefore needs **variable-length ("varlen") attention** that
respects document boundaries, which is provided by **FlashAttention-2** (`attn_implementation:
flash_attention_2`). With plain `sdpa`, packing can leak attention across examples.

**Practical rule used in this folder:**
- Headline run (`default.yaml`): `packing: true` + `attn_implementation: flash_attention_2`.
- Can't build flash-attn? Set `packing: false` + `attn_implementation: sdpa`. Correct, just slower.
  (The smoke config does exactly this so it runs anywhere.)

Install flash-attn with `uv sync --extra flash`.

---

## 6. Full fine-tune vs LoRA vs QLoRA

| | Full FT | LoRA | QLoRA |
|---|---|---|---|
| What trains | all weights | small low-rank adapters | adapters on a 4-bit-frozen base |
| Quality ceiling | highest | ~full for most SFT | ~LoRA, tiny extra hit from 4-bit |
| VRAM | high (see §7) | low | lowest |
| Mergeable / portable | n/a | adapter is a few MB | adapter is a few MB |
| Typical LR | `~2e-5` | `~1e-4` | `~1e-4` |
| Best for | you have the GPU and want max quality | most practitioners, most of the time | big models on small GPUs |

**LoRA** freezes the base weights and learns two small matrices `A, B` per target layer such that
the effective weight is `W + (α/r)·BA`, with rank `r ≪ d`. You train <1% of the parameters, the
optimizer state shrinks proportionally, and you ship a few-MB adapter instead of a full model copy.
For SFT specifically, LoRA gets you most of the way to full-FT quality.

**QLoRA** goes further: it holds the *base* weights in 4-bit (NF4) and trains LoRA adapters on top in
bf16. This is how people SFT a 70B on a single 80 GB card.

**This folder's headline run uses full fine-tuning** — Qwen3-4B fits comfortably in 80 GB (§7) and
full-FT is the cleanest thing to learn first, with no adapter/merge caveats. The code supports the
other two out of the box: add to your config

```yaml
use_peft: true
lora_r: 16
lora_alpha: 32
lora_target_modules: all-linear
learning_rate: 1.0e-4        # higher LR for adapters
# QLoRA: also add ↓ and install with `uv sync --extra quant`
load_in_4bit: true
```

`get_peft_config` / `get_quantization_config` in `train.py` pick these up automatically.

---

## 7. Fitting it on one 80 GB GPU: the memory arithmetic

The number that surprises newcomers: **the model weights are usually the *small* part.** For full
fine-tuning with Adam in mixed precision, rough per-parameter cost:

| Component | Bytes / param | For a 4B model |
|---|---|---|
| Weights (bf16) | 2 | 8 GB |
| Gradients (bf16) | 2 | 8 GB |
| Adam state: momentum + variance (fp32) | 8 | 32 GB |
| (fp32 master weights, if kept) | 4 | 16 GB |
| **Static subtotal** | **~12–16** | **~48–64 GB** |
| Activations | depends on batch × seq_len, ↓ by checkpointing | the rest |

So a 4B full-FT lands around ~50–65 GB static, leaving headroom on an 80 GB card for activations —
hence Qwen3-4B is a comfortable headline choice. Levers when you're tight:

- **`gradient_checkpointing: true`** (default here) — recompute activations in the backward pass
  instead of storing them. Trades ~20–30% compute for a large activation-memory cut. The single most
  important knob for fitting bigger batches/sequences.
- **Smaller `per_device_train_batch_size`, larger `gradient_accumulation_steps`** — same effective
  batch, less peak memory.
- **Shorter `max_length`** — activations scale with sequence length.
- **8-bit optimizer** (e.g. `optim: adamw_bnb_8bit`) — roughly halves the 32 GB Adam state.
- **LoRA / QLoRA** (§6) — collapses the gradient + optimizer terms because you train <1% of params.
- **Multi-GPU sharding** — FSDP/DeepSpeed-ZeRO split weights/grads/optimizer across GPUs (see §12).

> The point of this table isn't the exact GB — it's the *shape*: optimizer state dominates, and
> that's precisely what LoRA/QLoRA and ZeRO attack.

---

## 8. Hyperparameters that actually matter

Ordered by how often they make or break an SFT run:

1. **Data quality & mixture** — see §3. Dominates everything below.
2. **Epochs (1–3).** SFT overfits fast. Start with **1 epoch**; 2–3 only for small/narrow datasets.
   More epochs → memorization, degraded generalization, and "stylistic collapse." Watch eval loss
   turn back up.
3. **Learning rate.** Full-FT: **~1–2e-5**. LoRA/QLoRA: **~1e-4**. Too high → the model forgets its
   pretraining (capability loss); too low → underfits.
4. **`max_length`.** Set it to actually cover your data (truncating responses teaches the model to be
   cut off). Longer = more memory. 4096 is a sane default for general chat.
5. **Warmup + scheduler.** A short warmup (`warmup_ratio: 0.03`) + cosine decay is the standard,
   stable recipe.
6. **Effective batch size** (`per_device_batch × grad_accum × num_gpus`). Larger = smoother
   gradients, more stability; bounded by memory. 32–128 sequences is typical for SFT.
7. **Masking & packing** — §4, §5. Correctness, not just speed.

`SFTConfig` ships practitioner defaults that differ from raw `TrainingArguments`: `bf16=True`,
`gradient_checkpointing=True`, `learning_rate=2e-5`, `logging_steps=10`. We mostly accept them.

---

## 9. How to know it worked (evaluation)

SFT has no single score; triangulate:

- **Train/eval loss & `mean_token_accuracy`** (TRL logs both). Loss should fall and plateau; if eval
  loss climbs while train loss falls, you're overfitting — cut epochs/LR. `mean_token_accuracy` is
  an intuitive companion (fraction of next-tokens predicted correctly on non-masked positions).
- **It stops talking.** Generate and confirm the model emits EOS and ends turns cleanly. Runaway
  generation = EOS/template problem (§3), not a "smarter model" problem.
- **Qualitative generations.** Run `generate.py` on held-out prompts and *read* the outputs. Compare
  to the base model to see what SFT bought you. This catches format/refusal/repetition issues that
  loss hides.
- **Held-out benchmarks** (the rigorous bar). For general instruct models practitioners track
  instruction-following and capability suites (e.g. IFEval, MMLU, GSM8K, MT-Bench/AlpacaEval-style
  judgments). We build a proper eval harness in folder `06`; for now, loss + qualitative is enough to
  validate the pipeline.

> Loss going down proves *optimization* worked. Only generation + benchmarks prove the *model* got
> better. Always look at samples.

---

## 10. When SFT is enough — and when it isn't

**SFT alone is enough when:**
- You can *write down* examples of exactly the behavior you want (format conversion, extraction,
  domain Q&A, tone/style, tool-call syntax).
- There's a clear "correct" output to imitate.
- You don't need the model to exceed your demonstrations' quality.

**SFT is *not* enough when:**
- "Good" is easier to *rank* than to *write* ("which of these two answers is more helpful?"). →
  **Preference optimization (DPO, folder `04`).**
- You have a programmatic notion of correctness (unit tests pass, math answer matches, format
  validates). → **RL with verifiable rewards (GRPO, folder `05`).**
- You need to *suppress* behaviors (unsafe, sycophantic, verbose). Imitation can't easily teach "do
  less of X"; preference/RL can.
- You hit the **behavioral-cloning ceiling**: the model is as good as your best demonstrations and
  stops improving no matter how much SFT data you add.

This is the throughline of the whole course: **SFT gets you a competent instruct model; the rest of
post-training pushes it past what you could demonstrate by hand.**

---

## 11. Honest verdict: is SFT still SOTA?

**Yes — unambiguously, and it is not going away.** Unlike PPO (folder `03`), which has been largely
displaced in practice, SFT remains a mandatory, universal first stage of essentially every modern
post-training recipe (Tulu-3, Qwen, Llama, DeepSeek, etc.). What has evolved is *around* it:

- **Data-centric SFT.** The frontier moved from "more data" to *curated mixtures*, decontamination,
  difficulty balancing, and synthetic data with verification. Tulu-3 is a canonical example.
- **Long-context & multi-turn** SFT as standard, with packing + varlen attention as the default
  efficiency stack (§5).
- **Reasoning SFT / distillation.** A huge recent shift: SFT on *long chain-of-thought traces*
  (often distilled from a stronger reasoning model) is now a standard way to bootstrap reasoning
  before RL — DeepSeek-R1's "cold start" is exactly this. SFT is the on-ramp to the GRPO work in
  folder `05`.
- **Efficiency.** LoRA/QLoRA, FlashAttention, Liger kernels, 8-bit optimizers, FSDP/ZeRO — the
  *algorithm* is unchanged; the *systems* around it got much better.

So: master SFT not as a legacy step but as the load-bearing foundation it remains.

---

## 12. Run it

### Setup
```bash
cd 01-sft
uv sync                 # base deps
uv sync --extra flash   # + FlashAttention-2 for the headline run's packing (optional but recommended)
uv sync --extra quant   # + bitsandbytes, only if you want the QLoRA path
```

You'll need to be logged in to the Hub to pull the model/dataset, and to wandb for tracking
(wandb is the course-wide default tracker):
```bash
uv run hf auth login                       # or: export HF_TOKEN=...
uv run wandb login                         # or: export WANDB_API_KEY=...
export WANDB_PROJECT=post-training-lab     # groups all the lab's runs in one project
```
(Don't want wandb? Set `report_to: none` in the config, or `export WANDB_MODE=offline`.)

### 1) Smoke test first (always)
~1–2 minutes, tiny model, 5 steps — proves the pipeline before you spend GPU-hours.
```bash
uv run python train.py --config configs/smoke.yaml
```

### 2) The headline run (single A100/H100)
```bash
uv run python train.py --config configs/default.yaml
```
Subsamples Tulu-3 to 50k examples, 1 epoch, full fine-tune of Qwen3-4B-Base. Outputs land in
`outputs/qwen3-4b-tulu3-sft/`. Watch live metrics (loss, `mean_token_accuracy`, grad norm, LR) in
the wandb run it prints a link to.

> ⏱ **Expected runtime (rough):** ~**3–5 h** on a single A100-80GB (~1.5–2.5 h on H100), excluding
> first-run downloads. Scales ~linearly with `max_train_samples` — halve it for a quick pass. These
> are FLOP-based estimates, not measured; replace with your own once you've run it.

### 3) Talk to your model
```bash
uv run python generate.py --model_path outputs/qwen3-4b-tulu3-sft
```

### Multi-GPU — no code changes
The script is launcher-agnostic. For data-parallel (DDP):
```bash
uv run accelerate launch train.py --config configs/default.yaml
# or
uv run torchrun --nproc_per_node=4 train.py --config configs/default.yaml
```
For models too big to replicate, shard with FSDP/DeepSpeed-ZeRO via an accelerate config:
```bash
uv run accelerate config            # choose FSDP or DeepSpeed once
uv run accelerate launch train.py --config configs/default.yaml
```
Remember `per_device_train_batch_size` is *per GPU*: effective batch = per_device × grad_accum × #GPUs.

---

## 13. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Model never stops generating | EOS not aligned with chat template | ensure template's end-of-turn token is the EOS; for Qwen set `eos_token: "<\|im_end\|>"` |
| Output format looks wrong at inference | train/serve template mismatch | serve with the *exact* template you trained with (`apply_chat_template`) |
| Eval loss rises while train loss falls | overfitting | fewer epochs, lower LR, more/cleaner data |
| Model got "dumber" after SFT | LR too high → catastrophic forgetting | drop LR (full-FT ~1–2e-5), fewer epochs, or use LoRA |
| OOM at start | optimizer state (§7) | `gradient_checkpointing`, smaller batch + more grad-accum, 8-bit optim, or LoRA/QLoRA |
| OOM partway through | a long batch hit `max_length` activations | lower `max_length` or batch size |
| `assistant_only_loss` errors / trains on everything | template lacks `{% generation %}` tags | use a template that has them (we clone SmolLM3's) |
| Packing seems to hurt quality | cross-example attention leak | use `flash_attention_2` with packing, or disable packing |
| Loss is exactly 0 / NaN early | all labels masked, or fp16 overflow | check masking; prefer `bf16` over `fp16` |

---

## 14. Exercises

1. **Ablate the masking.** Train once with `assistant_only_loss: true` and once `false`. Compare eval
   loss *and* generations. Where does the difference show up?
2. **Epoch sweep.** Run 1 vs 3 epochs on the 50k subset. Find the point where eval loss turns up and
   generations start to feel "memorized."
3. **LoRA vs full-FT.** Flip to the LoRA config (§6). Compare quality, VRAM, and wall-clock. Was the
   quality gap worth the memory you saved?
4. **Break it on purpose.** Set `attn_implementation: sdpa` *with* `packing: true` and look for
   subtle quality degradation — feel the cross-example attention leak from §5.
5. **Data > scale.** Train on 10k vs 50k examples. Does 5× the data give 5× the improvement? (It
   won't — internalize the diminishing returns that motivate data curation.)

---

## 15. References

- TRL `SFTTrainer` docs — https://huggingface.co/docs/trl/sft_trainer
- TRL dataset formats — https://huggingface.co/docs/trl/dataset_formats
- Tülu 3 (open post-training recipe & data) — https://arxiv.org/abs/2411.15124
- QLoRA — https://arxiv.org/abs/2305.14314
- LoRA — https://arxiv.org/abs/2106.09685
- FlashAttention-2 — https://arxiv.org/abs/2307.08691
- Chat templates (transformers) — https://huggingface.co/docs/transformers/chat_templating

---

**Next:** [`02-reward-modeling`](../02-reward-modeling) — once you can imitate good answers, the next
question is how to *rank* them. We build the Bradley-Terry reward model that classic RLHF (folder
`03`) optimizes against.
