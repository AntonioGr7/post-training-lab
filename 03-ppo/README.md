# 03 · PPO (Proximal Policy Optimization)

> **The algorithm that defined the RLHF era — and the one you'll probably never reach for today.**
> PPO is how InstructGPT and the first ChatGPT were aligned. Understanding it is essential: it's the
> conceptual parent of everything in folders `04`–`06`, and the failure modes it exposes (reward
> hacking, KL control, instability) shaped every method that replaced it. But you should leave this
> folder understanding *why* the field largely moved on.

**You will:** run a real, reproducible RLHF-with-PPO loop on TL;DR summarization (single GPU), watch
a policy improve against a reward model, and develop a clear-eyed view of PPO's costs vs its
successors.

> **This folder is intentionally a *light build*.** The lesson is the deliverable; the demo exists so
> you can watch the loop turn, not to produce a SOTA model. TRL itself now ships `PPOTrainer` under
> `trl.experimental` — a strong signal about PPO's current standing.

> **Online/offline lens:** PPO is the archetypal **online, on-policy** method. Every update is
> computed from *fresh samples the current policy just generated*. That's the source of both its
> power (it optimizes the actual thing you deploy) and its pain (generation in the training loop →
> slow, memory-heavy, unstable). Contrast this constantly with the *offline* DPO of folder `04`.

---

## Table of contents
1. [Where PPO sits: closing the RLHF loop](#1-where-ppo-sits-closing-the-rlhf-loop)
2. [The RL objective: reward minus a KL leash](#2-the-rl-objective-reward-minus-a-kl-leash)
3. [The four models of PPO](#3-the-four-models-of-ppo)
4. [From policy gradient to the clipped surrogate](#4-from-policy-gradient-to-the-clipped-surrogate)
5. [The PPO loop, step by step](#5-the-ppo-loop-step-by-step)
6. [Hyperparameters & what to watch](#6-hyperparameters--what-to-watch)
7. [Why PPO is operationally brutal](#7-why-ppo-is-operationally-brutal)
8. [Honest verdict: is PPO still used?](#8-honest-verdict-is-ppo-still-used)
9. [Run it](#9-run-it)
10. [Common failure modes](#10-common-failure-modes)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. Where PPO sits: closing the RLHF loop

```
 SFT (01) ──► Reward Model (02) ──► PPO (this folder): optimize the policy against the RM
                                         │
                  ┌──────────────────────┴───────────────────────┐
                  │  repeat:                                       │
                  │   1. policy generates responses to prompts     │  ← online rollouts
                  │   2. reward model scores them                  │
                  │   3. PPO nudges policy toward higher reward    │
                  │      while a KL penalty keeps it near the SFT  │
                  └────────────────────────────────────────────────┘
```

This is **stage 3 of classic RLHF**. SFT gave us a competent policy; the RM gave us a way to score
responses; PPO uses reinforcement learning to push the policy toward high-reward outputs. Crucially,
it optimizes on the model's *own generations* — so unlike SFT it can push past the
behavioral-cloning ceiling. The price is that you must generate during training and keep the whole
thing from collapsing.

---

## 2. The RL objective: reward minus a KL leash

PPO for RLHF maximizes expected reward **minus a KL penalty** that keeps the policy `π_θ` close to
the frozen reference (SFT) policy `π_ref`:

$$\max_{\theta}\; \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot\mid x)}\Big[\, r_\phi(x,y) \;-\; \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_\text{ref}(\cdot\mid x)\big)\Big]$$

Two terms, two jobs:

- **`r_φ(x,y)`** — the reward model's score (folder `02`). Pulls the policy toward what the RM likes.
- **`β · KL(π_θ ‖ π_ref)`** — the leash. Without it, the policy would sprint to whatever degenerate
  text maximizes the RM (gibberish that the RM happens to score high) — **reward hacking** (folder
  `02` §6). The KL term keeps generations in the region where the RM is actually trustworthy. **`β`
  (`kl_coef`) is the single most important RLHF knob**: too small → reward hacking and incoherent
  text; too large → the policy never moves.

In TRL's logs this decomposes exactly: `objective/scores` is the raw RM score,
`objective/non_score_reward` is `−β·KL`, and `objective/rlhf_reward = scores − non_score_reward` is
what PPO actually optimizes. *That last number going up is the whole game.*

---

## 3. The four models of PPO

This is PPO's defining operational fact — and the root of its cost. You hold **four models**:

| Model | Role | Trained? | Class in `train.py` |
|---|---|---|---|
| **Policy** `π_θ` | the model being aligned; generates responses | ✅ trained | `AutoModelForCausalLM` |
| **Reference** `π_ref` | frozen SFT snapshot; the KL anchor | ❄️ frozen | `AutoModelForCausalLM` |
| **Reward model** `r_φ` | scores responses | ❄️ frozen | `AutoModelForSequenceClassification` |
| **Value model** `V_ψ` | estimates expected future reward (the critic) | ✅ trained | `AutoModelForSequenceClassification` |

The **value model (critic)** is the part people forget. PPO is an *actor-critic* method: the value
model predicts "from this partial generation, how much reward do I expect?" so we can compute
**advantages** (did this token do better or worse than expected?). It's a second full-size trained
model with its own optimizer state — which is a big reason PPO's memory footprint is brutal (§7), and
exactly the piece GRPO (folder `05`) deletes.

> With LoRA you can collapse policy+reference into one set of weights (reference = adapter disabled),
> which is why `train.py` sets `ref_model=None` when PEFT is on. But the value model remains.

---

## 4. From policy gradient to the clipped surrogate

The "PP" in PPO is the engineering that makes policy-gradient RL stable on huge models.

**Policy gradient (the starting point).** Increase the probability of actions (tokens) that led to
high advantage `A`: `∇J = E[ A · ∇log π_θ(a|s) ]`. Pure policy gradient is high-variance and
unforgiving: one big update on a bad batch can wreck the policy irrecoverably.

**Advantages via GAE.** We estimate each token's advantage with **Generalized Advantage Estimation**
(`gamma`, `lam`), using the value model as the baseline. GAE trades off bias vs variance — `lam=0.95`
is the standard.

**The clipped surrogate objective.** PPO constrains how far each update can move the policy by
clipping the probability ratio `ρ = π_θ(a|s) / π_θ_old(a|s)`:

$$\mathcal{L}^{\text{clip}} = \mathbb{E}\Big[\min\big(\rho\,A,\; \text{clip}(\rho,\,1-\epsilon,\,1+\epsilon)\,A\big)\Big]$$

Intuition: if an update would change a token's probability by more than ±`ε` (`cliprange`, default
0.2), the objective flattens — there's no gradient reward for moving further. This is a cheap
**trust region**: stay near the old policy, take many small safe steps instead of one reckless leap.
(This exact ratio-clipping mechanism is what folder `05`'s GRPO inherits — and what the DPPO paper
in the roadmap argues is *itself* the flaw to fix.)

**Total loss** = clipped policy loss + `vf_coef`·value loss (train the critic to predict returns) −
entropy bonus, with the KL penalty folded into the reward.

---

## 5. The PPO loop, step by step

Each iteration:

1. **Rollout.** Sample a batch of prompts; the policy *generates* responses (`response_length`
   tokens). ← the expensive, online part.
2. **Score.** The reward model scores each (prompt, response). Apply the **EOS trick**
   (`missing_eos_penalty`) to punish responses that never terminate.
3. **Evaluate & compute advantages.** Run the value model; compute per-token advantages via GAE;
   record the old policy's log-probs (the `π_θ_old` for the ratio).
4. **Optimize.** For `num_ppo_epochs` passes over `num_mini_batches` minibatches: compute the clipped
   surrogate + value loss, backprop, update policy and value. Clipping keeps each step in the trust
   region.
5. **Repeat** until `total_episodes` rollouts are consumed.

Steps 1–3 are pure inference on (up to) four models; only step 4 trains. Generation dominates
wall-clock — which is why production PPO setups bolt on fast inference engines and why this is all
*much* heavier than the single forward+backward of SFT/DPO.

---

## 6. Hyperparameters & what to watch

**Knobs that matter (in rough priority):**

| Param | Default | What it does / how to tune |
|---|---|---|
| `kl_coef` (β) | 0.05 | KL-leash strength. The master dial. Reward hacking + gibberish → raise it; policy won't move → lower it. |
| `learning_rate` | 3e-6 | PPO needs a *much* smaller LR than SFT/RM. Too high → collapse. |
| `cliprange` (ε) | 0.2 | Trust-region width. Rarely change from 0.2. |
| `num_ppo_epochs` | 4 | Optimization passes per rollout. Higher = more sample-efficient but more off-policy drift. |
| `response_length` | 53 | Max new tokens per rollout. Longer = costlier + more room to hack. |
| `whiten_rewards` | false | Normalize rewards per batch for stability. |
| `missing_eos_penalty` | — | The "EOS trick": penalize non-terminating completions. Very useful. |
| `vf_coef`, `gamma`, `lam` | 0.1, 1.0, 0.95 | Value-loss weight and GAE params. Defaults are good. |

**Metrics to watch (TRL logs all of these):**

- **`objective/rlhf_reward`** — *the* success signal. Should climb steadily. If it doesn't, training
  isn't working.
- **`objective/kl`** — divergence from the reference. Should grow *slowly and stay bounded*. A KL
  that explodes = the policy is running off to hack the RM; raise `kl_coef`.
- **`val/ratio`** — the PPO probability ratio. Should hover near **1.0**. If it's 2.0, 0.1, 1000 →
  updates are too drastic; something's wrong (LR, batch sizes, numerics).
- **`policy/clipfrac_avg`** — fraction of updates being clipped. Moderate is healthy; near-1.0 means
  every update is slamming the trust-region wall.
- **`objective/scores` vs true quality** — watch the sample-generation tables (`num_sample_generations`).
  Rising RM score with *falling* readable quality = reward over-optimization (folder `02` §6).

> The fact that you must babysit `val/ratio`, `objective/kl`, and clip fractions — and that any one
> going sideways can silently ruin a run — is precisely the operational burden that motivated PPO's
> successors.

---

## 7. Why PPO is operationally brutal

A candid inventory of the costs that drove the field away:

1. **Four models in memory.** Policy + value (both trained, both with optimizer state) + reference +
   reward (frozen). Two *trained* full-size models is roughly double an SFT job's footprint — which
   is exactly why we use a small reference setup here and why a Qwen3-4B PPO run won't fit one 80GB
   GPU without LoRA and/or sharding.
2. **Generation inside the training loop.** Every step waits on autoregressive sampling. Slow, and it
   makes the systems engineering (batching, KV cache, fast-inference backends) a project in itself.
3. **Hyperparameter fragility.** `kl_coef`, LR, clip range, batch sizing, reward scale — many
   interacting dials, and a bad combination doesn't just underperform, it *diverges*. PPO is
   notorious for being hard to reproduce.
4. **A whole extra trained model to maintain.** The value/critic doubles the trainable-parameter
   bookkeeping and is itself a source of instability if its loss isn't balanced (`vf_coef`).
5. **Reward hacking is ever-present.** Optimizing a learned proxy hard (folder `02` §6) demands
   constant vigilance: KL tuning, reward normalization, early stopping on *true* eval.

None of this means PPO is wrong — it's a sound, powerful algorithm. It means the *cost-to-benefit*
ratio for LLM alignment is poor compared to what came next.

---

## 8. Honest verdict: is PPO still used?

**Mostly no, for LLM alignment — and that's the key lesson of this folder.** Be precise about why:

- **DPO (folder `04`) removed the RL loop entirely** for preference alignment. It showed you can get
  the same RLHF objective by training *directly on preference pairs* — no reward model, no value
  model, no rollouts, no KL babysitting. For "align a chat model on preferences," DPO and its
  relatives are now the default. This deleted PPO's main use case.
- **GRPO (folder `05`) removed the value model** for reward-based RL. By estimating advantages from a
  *group* of sampled responses (their mean reward as the baseline) instead of a learned critic, GRPO
  keeps online RL's benefits at roughly half the model footprint and far less fragility. For
  reasoning/RLVR — the place you genuinely still want online RL — GRPO and friends (RLOO, etc.)
  largely replaced PPO.
- **TRL classifies `PPOTrainer` as experimental** (`trl.experimental.ppo`) — the maintainers'
  implicit statement that it's no longer the recommended path.

**Where PPO (or close kin) still shows up:** some large industrial RLHF pipelines that already
invested heavily in PPO infrastructure; certain control/agentic RL settings; and as the theoretical
baseline every newer method is compared against. If you're starting fresh in 2026, you'd reach for
DPO (preferences) or GRPO (verifiable/reward RL) first — and you'd understand both *because* you
understand PPO.

> **Why teach it at all, then?** Because PPO is the Rosetta Stone of RL post-training. KL-to-reference,
> advantages, the clipped trust region, reward hacking, the actor-critic split — these concepts are
> *load-bearing* in every successor. DPO is "PPO's objective, solved in closed form." GRPO is "PPO
> minus the critic." You cannot deeply understand them without this folder.

---

## 9. Run it

### Setup
```bash
cd 03-ppo
uv sync
uv run hf auth login
uv run wandb login && export WANDB_PROJECT=post-training-lab
```

### 1) Smoke test first
```bash
uv run python train.py --config configs/smoke.yaml
```
A few hundred episodes — just confirms the rollout→score→optimize loop spins.

### 2) The demo run
```bash
uv run python train.py --config configs/default.yaml
```
Watch **`objective/rlhf_reward`** climb and **`objective/kl`** stay bounded in the wandb run, and read
the periodic sample-generation tables. (We shrink `total_episodes` to 30k vs the published 1M, so the
model won't fully converge — the point is to *see the loop work*.)

> ⏱ **Expected runtime (rough):** ~**1–3 h** on a single A100-80GB for 30k episodes — but PPO is
> *generation-bound*, not FLOP-bound, so this is the least reliable estimate in the course (assume
> ±2×). Scales ~linearly with `total_episodes`. Multi-GPU helps here more than anywhere else.

### 3) Compare before/after
```bash
uv run python generate.py --model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr   # SFT baseline
uv run python generate.py --model_path outputs/ppo-tldr                                  # after PPO
```

### Multi-GPU (recommended for PPO)
PPO is the folder where multi-GPU matters most — sharding the four models with DeepSpeed ZeRO is the
standard way to make real runs feasible:
```bash
uv run accelerate config        # choose DeepSpeed ZeRO-2/3
uv run accelerate launch train.py --config configs/default.yaml
```

### Wiring it to your own Qwen3 stack
Point the paths at folders 01 & 02 in the config:
```yaml
sft_model_path: ../01-sft/outputs/qwen3-4b-tulu3-sft
reward_model_path: ../02-reward-modeling/outputs/qwen3-4b-ultrafeedback-rm
```
**Expect it not to fit one 80GB GPU at 4B** (two trained models + two frozen). Use LoRA (`use_peft:
true`, which also lets `ref_model=None`) and/or multi-GPU ZeRO-3. This memory wall is itself the
lesson — and the motivation for GRPO.

---

## 10. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `objective/rlhf_reward` flat | LR too low, KL too high, or reward scale off | raise LR slightly, lower `kl_coef`, check RM outputs |
| `objective/kl` explodes; text degenerates | reward hacking; `kl_coef` too small | raise `kl_coef`; enable `whiten_rewards`; shorten `response_length` |
| `val/ratio` far from 1.0 | updates too drastic | lower LR, fewer `num_ppo_epochs`, check batch sizing |
| OOM immediately | four models don't fit | smaller model, LoRA (`ref_model=None`), DeepSpeed ZeRO-3 |
| Responses never stop | no EOS | set `missing_eos_penalty`, `stop_token: eos` |
| Reward rises but quality drops | over-optimization | early-stop on true eval, raise KL, retrain/ensemble RM |
| Wildly irreproducible runs | PPO's hyperparameter fragility | fix seeds, change one knob at a time, mirror known-good configs |

---

## 11. Exercises

1. **Feel the KL leash.** Run with `kl_coef: 0.0` vs `0.05` vs `0.3`. Watch `objective/kl` and the
   sample generations. Find where reward hacking kicks in and where the policy stops moving.
2. **Watch the ratio.** Crank `learning_rate` 5–10×. Observe `val/ratio` and `clipfrac` blow up, and
   `rlhf_reward` collapse. This is *why* the clipped trust region exists.
3. **Count the cost.** Log GPU memory and step time vs an SFT run from folder `01`. Quantify PPO's
   tax — then re-read §8.
4. **PPO vs DPO (cross-folder).** After folder `04`, align the *same* SFT model on the *same*
   preferences with both. Compare result quality, engineering effort, and wall-clock. Decide for
   yourself which you'd ship.
5. **Find the critic's cost.** Estimate how much memory the value model + its optimizer state adds.
   That number is roughly what GRPO saves (folder `05`).

---

## 12. References

- PPO (Schulman et al. 2017) — https://arxiv.org/abs/1707.06347
- GAE (Schulman et al. 2015) — https://arxiv.org/abs/1506.02438
- InstructGPT / RLHF with PPO (Ouyang et al. 2022) — https://arxiv.org/abs/2203.02155
- Learning to summarize from human feedback (Stiennon et al. 2020) — https://arxiv.org/abs/2009.01325
- The N+ implementation details of RLHF with PPO (TL;DR case study) — https://arxiv.org/abs/2403.17031
- TRL `PPOTrainer` docs — https://huggingface.co/docs/trl/ppo_trainer
- KL approximation (Schulman, k1/k3 estimators) — http://joschu.net/blog/kl-approx.html

---

**Next:** [`04-dpo`](../04-dpo) — Direct Preference Optimization. The same RLHF objective from §2, but
solved in closed form so you train *directly on preference pairs* — no reward model, no value model,
no rollouts, no KL babysitting. This is why most of PPO's use case evaporated.
