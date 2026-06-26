# 04 · DPO (Direct Preference Optimization) & its variants

> **PPO's RLHF objective (folder `03` §2) — solved in closed form.** DPO proves that the optimal
> policy for the KL-constrained reward-maximization objective has an *analytic* form. So instead of
> training a reward model and then running an RL loop against it, you train the policy **directly on
> preference pairs** with a single classification-style loss. **No reward model, no value model, no
> rollouts, no KL babysitting.** This is why most of PPO's use case evaporated, and why DPO is the
> modern default for preference alignment.

**You will:** align Qwen3-4B on real preference data (the *same* UltraFeedback pairs folder `02`
trained a reward model on), understand the math that collapses RLHF into one loss, watch the implicit
reward separate chosen from rejected, and learn the whole **variant landscape** (IPO, KTO, ORPO,
CPO/SimPO, BCO) well enough to pick the right one for a given dataset.

> **Online/offline lens:** DPO is the archetypal **offline** method — a *fixed* dataset of preference
> pairs, no generation in the training loop. It is the exact foil to PPO's online rollouts (folder
> `03`). That is its great strength (cheap, stable, one forward+backward) and its hard ceiling: **it
> can only sharpen behaviors already represented in the pair distribution — it cannot explore.** That
> ceiling is precisely what motivates the online methods in folders `05`/`06`.

---

## Table of contents
1. [Where DPO sits: deleting the RL loop](#1-where-dpo-sits-deleting-the-rl-loop)
2. [The derivation: from PPO's objective to one loss](#2-the-derivation-from-ppos-objective-to-one-loss)
3. [The implicit reward & what `beta` really does](#3-the-implicit-reward--what-beta-really-does)
4. [The reference model: why two models, and the LoRA shortcut](#4-the-reference-model-why-two-models-and-the-lora-shortcut)
5. [Hyperparameters & what to watch](#5-hyperparameters--what-to-watch)
6. [The variant landscape (the signature of this folder)](#6-the-variant-landscape-the-signature-of-this-folder)
7. [Decision framework: which method, when](#7-decision-framework-which-method-when)
8. [Honest verdict: is DPO SOTA?](#8-honest-verdict-is-dpo-sota)
9. [Run it](#9-run-it)
10. [Common failure modes](#10-common-failure-modes)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. Where DPO sits: deleting the RL loop

```
 SFT (01) ──► [classic RLHF path] ──► Reward Model (02) ──► PPO (03): RL loop against the RM
        │
        └───► [DPO path, this folder] ──► train DIRECTLY on the preference pairs
                                            (no RM, no rollouts — one supervised-style pass)
```

Classic RLHF is three stages: SFT → reward model → PPO. DPO's insight is that **stages 2 and 3
collapse into a single training step**. The preference pairs that you *would have* used to train a
reward model are used to train the policy directly. The reward model still exists — but only
*implicitly*, baked into the policy's own log-probabilities (§3).

The practical consequence is enormous. Compare the operational reality against folder `03`:

| | PPO (folder `03`) | DPO (this folder) |
|---|---|---|
| Models in memory | 4 (policy, value, reference, reward) | 2 (policy + reference) — or **1** with LoRA |
| Generation in the loop? | yes (online rollouts) | **no** (offline, fixed pairs) |
| Separate reward model? | yes (folder `02`) | **no** (implicit) |
| Value/critic model? | yes | **no** |
| KL control | manual `kl_coef` you must babysit | folded into `beta`, set-and-forget |
| Stability | fragile, hard to reproduce | stable, ~SFT-like |
| Compute | generation-bound, heavy | one forward+backward over pairs |

DPO keeps the *goal* of RLHF (align to preferences while staying near the reference) and throws away
the *machinery* (the loop, the critic, the explicit reward, the sampling).

---

## 2. The derivation: from PPO's objective to one loss

This is the heart of the lesson — the closed-form trick. Start from the **exact same objective PPO
optimizes** (folder `03` §2): maximize reward minus a KL leash to the reference policy.

$$\max_{\pi}\; \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(\cdot\mid x)}\big[\, r(x,y)\,\big] \;-\; \beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_\text{ref}(\cdot\mid x)\big)$$

**Step 1 — the optimal policy has a closed form.** This KL-constrained objective is not something you
*have* to solve with RL. It has a known analytic maximizer:

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_\text{ref}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}\,r(x,y)\Big)$$

The optimal policy is just the reference reweighted by the exponentiated reward. The catch is the
partition function `Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β)` — a sum over *all possible sequences*,
hopelessly intractable. This is exactly why classic RLHF gave up on the closed form and used PPO to
approximate it.

**Step 2 — invert it to express the reward in terms of the policy.** Take logs and rearrange:

$$r(x,y) = \beta\,\log\frac{\pi^*(y\mid x)}{\pi_\text{ref}(y\mid x)} + \beta\,\log Z(x)$$

The reward equals a **log-ratio between the optimal policy and the reference**, plus a term that
depends only on `x` (not on `y`).

**Step 3 — the Bradley-Terry model cancels `Z(x)`.** Preferences are modeled (folder `02` §2) by
Bradley-Terry: `P(y⁺ ≻ y⁻ | x) = σ(r(x,y⁺) − r(x,y⁻))`. This is a *difference of two rewards for the
same prompt `x`* — so the intractable `β log Z(x)` term, identical for both completions, **cancels
exactly**. Substituting the reward expression from Step 2:

$$\mathcal{L}_\text{DPO}(\theta) = -\,\mathbb{E}_{(x,y^+,y^-)}\Big[\log \sigma\Big(\beta\log\frac{\pi_\theta(y^+\mid x)}{\pi_\text{ref}(y^+\mid x)} - \beta\log\frac{\pi_\theta(y^-\mid x)}{\pi_\text{ref}(y^-\mid x)}\Big)\Big]$$

**That's it.** A binary-cross-entropy-style loss over preference pairs. No reward model to train, no
`Z(x)` to estimate, no sampling. Everything you need is four log-probabilities per pair — the
policy's and the reference's, on the chosen and rejected completions. You are still optimizing PPO's
objective; you've just removed the need to do it the hard way.

> **The one-sentence version to remember:** *DPO makes the language model its own reward model* — the
> reward is implicitly `β·log(π_θ/π_ref)`, and training the policy to rank chosen above rejected under
> that implicit reward is equivalent to the full RLHF objective.

---

## 3. The implicit reward & what `beta` really does

DPO never builds a reward model, but it defines one implicitly. The **implicit reward** of a
completion is:

$$\hat{r}(x,y) = \beta\,\log\frac{\pi_\theta(y\mid x)}{\pi_\text{ref}(y\mid x)}$$

TRL logs this directly: `rewards/chosen`, `rewards/rejected`, `rewards/margins` (chosen − rejected),
and `rewards/accuracies` (fraction of pairs where chosen's implicit reward beats rejected's). These
are your primary training signals — there is no external RM accuracy to watch like in folder `02`;
the implicit-reward margin *is* the thing.

**`beta` is the single most important DPO knob** — it is the KL strength from the original objective,
now living inside the loss:

- **Low `beta` (e.g. 0.01–0.05):** weak leash. The policy is free to move far from the reference to
  fit the preferences hard. Stronger preference accuracy, but more drift → more risk of degeneration,
  reduced diversity, and overfitting to the (possibly noisy) pair labels.
- **High `beta` (e.g. 0.3–0.5):** strong leash. The policy stays close to the reference. Conservative,
  safe, but may barely move the behavior.
- **`beta = 0.1`** is the field default and a sane starting point. Tune it like you'd tune `kl_coef`
  in PPO — because it *is* the same dial.

> **A crucial subtlety — DPO often works by *suppression*, not promotion.** Empirically, DPO usually
> *lowers* the log-prob of the rejected completion faster than it *raises* the chosen one. The margin
> grows, accuracy climbs — but watch `logps/chosen`: if it's falling in absolute terms, the model is
> getting *less* likely to produce the good answer too, just less so than the bad one. Past a point
> this hurts generation quality even as the DPO metrics look great. This is DPO's version of reward
> hacking, and it's why you must judge by *generations*, not just `rewards/margins`.

---

## 4. The reference model: why two models, and the LoRA shortcut

DPO's loss needs `π_ref` — the frozen starting policy — to compute every log-ratio. How you supply it
determines your memory footprint.

**Full fine-tune (the default config).** Pass `ref_model=None` and TRL freezes a copy of the
**initial policy** as the reference. So you hold **two 4B models** at once: the trainable policy (with
optimizer state) + the frozen reference (forward-only). In bf16 that's ~58 GB static at 4B — it fits
one 80 GB GPU **with gradient checkpointing and a modest `max_length`** (our default uses both). This
is the headline run.

> **What should the reference be?** Properly, your *SFT model* (folder `01`'s output) — DPO assumes a
> policy that already produces reasonable completions; it sharpens preferences, it doesn't teach the
> task. We use the released `Qwen/Qwen3-4B` instruct model so the folder stands alone, but point
> `model_name_or_path` at `../01-sft/outputs/qwen3-4b-tulu3-sft` to do it the canonical way.

**LoRA (uncomment `use_peft` in the config).** Here the reference is *free*: it's just the policy with
its adapter **disabled**. One set of base weights serves as both policy (adapter on) and reference
(adapter off), so you still pass `ref_model=None` but there is **no second model**. This roughly
halves memory and is the standard way to DPO larger models on smaller GPUs. The trade: raise the LR
to **~1e-5** (adapters need a higher LR than full-FT — see §5).

> `train.py` always passes `ref_model=None`; TRL does the right thing in both cases. To DPO an
> *existing* LoRA adapter further, load the `PeftModel` yourself and pass it as `model` without
> `peft_config` (see the TRL docs' "continue training your PeftModel" snippet).

---

## 5. Hyperparameters & what to watch

**Knobs that matter (in rough priority):**

| Param | Default here | What it does / how to tune |
|---|---|---|
| `learning_rate` | 5e-7 (full-FT) | **The #1 footgun.** DPO LRs are *much* smaller than SFT — ~5e-7 full-FT, ~1e-5 LoRA. Too high → the policy collapses (rewards diverge, generations degrade). If one thing breaks your run, it's this. |
| `beta` | 0.1 | Implicit-reward temperature == KL strength (§3). Lower = fit prefs harder + drift more; higher = stay near reference. |
| `loss_type` | `sigmoid` | Selects the variant: `sigmoid` (vanilla), `ipo`, `sigmoid_norm` (SimPO-style), `hinge`, … (§6). A plain config field — no code change. |
| `max_length` / `max_prompt_length` | 1536 / 768 | Truncation budget. Bigger = more memory (you score chosen+rejected for *two* models). Long pairs that truncate the completion silently corrupt the signal. |
| `num_train_epochs` | 1 | DPO overfits fast. 1–3 epochs is typical; watch eval margins stop improving. |
| `gradient_checkpointing` | true | Required to fit policy+reference at 4B on one GPU. |

**Metrics to watch (TRL logs all of these):**

- **`rewards/accuracies`** — fraction of eval pairs where chosen's implicit reward > rejected's.
  Should climb toward (but rarely reach) 1.0. *The* headline signal.
- **`rewards/margins`** — mean implicit-reward gap (chosen − rejected). Should grow steadily.
- **`rewards/chosen` and `rewards/rejected`** — watch *both*. Healthy DPO: rejected falls, chosen
  holds roughly flat or rises slightly. **Danger: chosen also falling sharply** → the suppression
  pathology of §3; your model is getting worse at the good answer too.
- **`logps/chosen`** — absolute log-prob of chosen. If this craters, generation quality is degrading
  even if margins look great. Judge by `generate.py`, not metrics alone.
- **`loss`** — should decrease smoothly. Spikes/NaNs ⇒ LR too high.

> Unlike PPO you do **not** babysit a ratio, a KL estimate, and clip fractions while generation runs.
> DPO's whole pitch is that this table is short and the run is stable. That stability is the product.

---

## 6. The variant landscape (the signature of this folder)

DPO spawned a family. Two kinds of variants: **(a) alternative losses** you select with one
`loss_type` field on the *same* `DPOTrainer`, and **(b) separate trainers** for fundamentally
different data/setup assumptions. Knowing which to reach for is the practitioner skill.

### (a) Same trainer, swap `loss_type`

These all run through `train.py` unchanged — just edit `loss_type` in the YAML:

| `loss_type` | Method | Why / when |
|---|---|---|
| `sigmoid` | **Vanilla DPO** | The default. Start here. |
| `ipo` | **IPO** | Fixes DPO's tendency to **overfit to deterministic preferences**: when a pair is "always chosen ≻ rejected," vanilla DPO can push the margin to ∞ and degrade. IPO uses an identity (not log-sigmoid) transform with a margin target, regularizing this away. Reach for it when preferences are clean/near-deterministic or you see margin runaway. |
| `sigmoid_norm` | **SimPO-style** | Length-normalizes the log-ratios, attacking DPO's **length bias** (the tendency to prefer longer completions). SimPO proper is also *reference-free* (see `CPOTrainer` below); this loss is the length-normalization piece. |
| `hinge` | **SLiC/RSO hinge** | Hinge loss on the margin instead of logistic; `beta` becomes the reciprocal of the margin. |
| `robust` | **Robust DPO** | Models label noise via `label_smoothing` ∈ [0, 0.5). Use when your preference labels are known to be noisy. |
| `exo_pair`, `nca_pair`, `bco_pair`, `aot`, `apo_zero/down`, `sppo_hard`, `discopop` | research losses | Various theoretical refinements (reverse-KL, optimal transport, anchored objectives, …). Mostly for research; `sigmoid`/`ipo` cover the practical 95%. |

> TRL also supports **combining** losses with weights (`loss_type=["sigmoid","bco_pair","sft"]`,
> `loss_weights=[...]`) — this is **MPO** (Mixed Preference Optimization). Adding a small `sft` term
> is a common trick to counteract the chosen-log-prob decay of §3.

### (b) Separate trainers (all 🧪 experimental in TRL)

These change the *data contract* or *model setup*, so they're distinct trainers, not a `loss_type`:

| Trainer | What's different | Use when |
|---|---|---|
| **`KTOTrainer`** 🧪 | **Unpaired** data: a single good/bad (thumbs-up/down) label per sample, no pairs. Based on prospect theory. | You have **binary feedback**, not preference pairs — e.g. production 👍/👎 logs. Often easier to collect at scale than pairs. |
| **`ORPOTrainer`** 🧪 | **Reference-free** *and* **SFT-free**: folds preference alignment into the SFT stage itself (a log-odds penalty added to the SFT loss). **No reference model, no separate SFT run.** | You want a one-stage pipeline from a base model, or can't afford a second model in memory. |
| **`CPOTrainer`** 🧪 | **Reference-free** DPO-style loss (drops `π_ref`). **SimPO** is reached via its `loss_type`. | Memory is tight (no reference model) and you accept slightly less principled KL control. |
| **`BCOTrainer`** 🧪 | Binary classifier whose logit is the reward; handles **unpaired** data with a running reward shift. | Unpaired data, as an alternative to KTO. |

**The honesty signal (per the course's taxonomy):** **`DPOTrainer` is the *only non-experimental*
offline preference trainer in TRL.** KTO/ORPO/CPO/BCO are *all* marked 🧪. That is TRL telling you, in
its own labeling, exactly what this course argues: **DPO is the stable default; the variants are
secondary** — reach for them only when your data or memory constraints force the issue.

> ORPO config sketch (reference-free, no SFT stage), for orientation — it's a different trainer, so a
> separate script, but the lab keeps the build DPO-centric:
> ```python
> from trl import ORPOConfig, ORPOTrainer
> trainer = ORPOTrainer(model="Qwen/Qwen3-4B", args=ORPOConfig(beta=0.1, ...),
>                       train_dataset=ds)   # no ref_model, run directly on a base model
> ```

---

## 7. Decision framework: which method, when

Walk these questions in order:

1. **Is your task verifiable (math, code, has a checkable answer)?** → You don't want preference
   alignment at all; go to **GRPO / RLVR (folder `05`)**. DPO is for *taste*, not *correctness*.
2. **Do you have preference *pairs* (chosen vs rejected for the same prompt)?**
   - **Yes** → **DPO** (`loss_type=sigmoid`). The default. Clean/deterministic prefs or margin
     runaway? → `ipo`. Length bias? → `sigmoid_norm`/SimPO. Noisy labels? → `robust`.
   - **No, I have binary 👍/👎 (unpaired)** → **KTO** (or BCO).
3. **Can you afford a separate SFT stage + a reference model in memory?**
   - **No, I want one stage from a base model** → **ORPO** (reference-free + SFT-free).
   - **No reference model, but I'll still SFT first** → **CPO/SimPO** (reference-free).
   - **Yes** → **DPO** (the principled default).
4. **Do you need the model to exceed the quality present in your pairs (explore new behaviors)?** →
   DPO *cannot* (it's offline; see §8). Go **online: Online DPO (`06`) or GRPO (`05`)**.

> Default answer for "align a chat model on human/AI preferences in 2026": **DPO with `sigmoid`,
> `beta=0.1`, a tiny LR, initialized from your SFT model.** Everything else is a targeted deviation.

---

## 8. Honest verdict: is DPO SOTA?

**Yes — DPO is the de facto standard for preference alignment, and you should reach for it first.**
It delivers ~all of PPO-RLHF's preference-alignment benefit at a fraction of the cost and complexity,
it's stable and reproducible, and it's the *only* non-experimental offline preference trainer TRL
ships. Most open chat models since ~2024 use DPO or a close relative in their alignment stage.

**But be precise about its ceiling — this is the lesson that sets up the rest of the course:**

- **DPO is offline. It cannot explore.** It only ever sees the fixed pairs you give it, so it can
  *sharpen* and *re-rank* behaviors already present in that distribution — it cannot discover better
  responses than the ones in the dataset. PPO/GRPO, by generating fresh samples, *can*. When your
  pairs cap out the quality you need, you've hit the offline wall → **online methods (`05`, `06`)**.
- **It's only as good as the pairs.** Garbage/noisy/biased preferences → a garbage/biased policy,
  with none of an explicit reward model's interpretability (folder `02`'s probing) to catch it.
- **The suppression pathology (§3)** means DPO can quietly degrade generation quality while its own
  metrics improve. You must validate on real generations, not just `rewards/margins`.
- **It still needs a good SFT base.** DPO aligns; it doesn't teach the task. Skip folder `01` and DPO
  has nothing solid to sharpen.

> **The arc of folders 03→04→05:** PPO (online, 4 models, fragile) → DPO (offline, ≤2 models, stable,
> but can't explore) → GRPO (online again, but only 1 extra model and verifiable rewards — buys back
> exploration without PPO's critic and KL pain). DPO isn't the end of the story; it's the stable
> center of it.

---

## 9. Run it

### Setup
```bash
cd 04-dpo
uv sync
uv run hf auth login
uv run wandb login && export WANDB_PROJECT=post-training-lab
```

### 1) Smoke test first
```bash
uv run python train.py --config configs/smoke.yaml
```
Qwen3-0.6B, 256 pairs, 5 steps (~1–2 min). Confirms the pipeline loads policy+reference, tokenizes
pairs, and steps — before you spend GPU-hours.

### 2) The headline run
```bash
uv run python train.py --config configs/default.yaml
```
Qwen3-4B full fine-tune on 30k UltraFeedback pairs. Watch **`rewards/accuracies`** and
**`rewards/margins`** climb in wandb — and keep an eye on `rewards/chosen` (§3, §5). At the end the
script prints the final eval reward accuracy and margin.

> ⏱ **Expected runtime (rough):** ~**2–4 h** on a single A100-80GB for 30k pairs, 1 epoch. **Estimate,
> not measured** — FLOP-based, similar profile to folder `02`'s RM run. H100 ≈ 0.5–0.6×. Scales ~linearly
> with `max_train_samples` and epochs. Replace with a real number once you've run it.

### 3) Compare before/after
```bash
uv run python generate.py --model_path Qwen/Qwen3-4B                          # un-aligned baseline
uv run python generate.py --model_path outputs/qwen3-4b-ultrafeedback-dpo     # after DPO
```
Read them side by side. DPO's effect is on *taste* (helpfulness, format, calibration) — and remember
the offline ceiling (§8) when judging.

### LoRA path (smaller GPUs)
Uncomment the `use_peft` block in `configs/default.yaml` **and raise the LR to ~1e-5**. The reference
becomes the adapter-disabled policy (no second model), roughly halving memory.

### Multi-GPU
`train.py` is `accelerate`-ready (TRL `DPOTrainer` over `Trainer`):
```bash
uv run accelerate config        # DDP, or DeepSpeed ZeRO-2/3 to shard the 4B + reference
uv run accelerate launch train.py --config configs/default.yaml
```

### Try a variant — no code change
```bash
# IPO (regularizes overfitting to deterministic prefs):
uv run python train.py --config configs/default.yaml --loss_type ipo
# SimPO-style length-normalized DPO:
uv run python train.py --config configs/default.yaml --loss_type sigmoid_norm
```

---

## 10. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss spikes / NaN; generations degrade fast | **LR too high** (the classic DPO mistake) | drop to ~5e-7 full-FT / ~1e-5 LoRA; check you didn't copy an SFT LR |
| `rewards/margins` flat, accuracy ~0.5 | LR too low, `beta` too high, or bad pairs | raise LR slightly, lower `beta`, sanity-check the dataset |
| Margins look great but model got *worse* | suppression pathology — `rewards/chosen` falling (§3) | lower LR/`beta`, fewer epochs, add a small `sft` loss term (MPO), validate on generations |
| OOM at 4B | policy + reference don't fit | enable `gradient_checkpointing` (on by default), lower `max_length`, or switch to **LoRA** (`ref_model` free) |
| Outputs got longer & blander | DPO **length bias** | use `loss_type: sigmoid_norm` (SimPO) or shorten `max_length` |
| Margin runs away then quality drops | overfitting deterministic prefs | use `loss_type: ipo`; reduce epochs |
| Truncation warnings everywhere | pairs longer than `max_length` | raise `max_length`/`max_prompt_length` (watch memory) or filter long pairs |
| No improvement vs base at all | base model too weak (no SFT) | DPO sharpens, doesn't teach — start from a real SFT model (folder `01`) |

---

## 11. Exercises

1. **Feel `beta`.** Run `beta = 0.01`, `0.1`, `0.5`. Plot `rewards/margins` and `rewards/chosen`, and
   eyeball generations. Find where the policy stops moving (high `beta`) and where it starts degrading
   (low `beta`).
2. **Catch the suppression.** Log `rewards/chosen` and `rewards/rejected` separately. Confirm DPO
   mostly works by pushing `rejected` *down*. Then add a small `sft` loss (MPO) and see `chosen` hold up.
3. **Vanilla vs IPO.** Train `sigmoid` and `ipo` on the same data; compare margin behavior and final
   generation quality. When does IPO's regularization help?
4. **Length bias.** Measure mean output length before/after vanilla DPO, then with `sigmoid_norm`
   (SimPO). Quantify the bias and the fix.
5. **DPO vs PPO (cross-folder).** Align the *same* SFT model on the *same* UltraFeedback prefs with
   PPO (folder `03`) and DPO. Compare quality, engineering effort, and wall-clock. Decide what you'd ship.
6. **RM vs DPO (cross-folder).** Folder `02` trained a *reward model* on these exact pairs; here you
   trained a *policy* on them. Use folder `02`'s RM to score this DPO model's generations. Do the
   implicit reward (DPO) and the explicit reward (RM) agree?
7. **Hit the offline wall.** Find a prompt type underrepresented in UltraFeedback. Confirm DPO can't
   improve it (no pairs to learn from) — the motivation for online RL (`05`/`06`).

---

## 12. References

- DPO (Rafailov et al. 2023) — https://arxiv.org/abs/2305.18290
- IPO (Azar et al. 2023) — https://arxiv.org/abs/2310.12036
- KTO (Ethayarajh et al. 2024) — https://arxiv.org/abs/2402.01306
- ORPO (Hong et al. 2024) — https://arxiv.org/abs/2403.07691
- SimPO (Meng et al. 2024) — https://arxiv.org/abs/2405.14734
- CPO (Xu et al. 2024) — https://arxiv.org/abs/2401.08417
- SLiC-HF / hinge (Zhao et al. 2023) — https://arxiv.org/abs/2305.10425
- MPO (mixed preference optimization) — https://arxiv.org/abs/2411.10442
- TRL `DPOTrainer` docs — https://huggingface.co/docs/trl/dpo_trainer
- TRL dataset formats — https://huggingface.co/docs/trl/dataset_formats

---

**Next:** [`05-grpo`](../05-grpo) — GRPO / RLVR. DPO can't exceed its preference distribution because
it never generates. When you have **verifiable** rewards (math, code) and need the model to *explore*
better answers, you go back online — but GRPO buys exploration back without PPO's critic or KL pain
(one extra model, group-relative advantages). The flagship folder.
