# 02 · Reward Modeling

> **Teaching a model to *judge*.** SFT (folder `01`) taught a model to imitate good answers. But
> imitation can't tell you that answer A is *better* than answer B — and that comparison is the fuel
> for every RL method in this course. A **reward model (RM)** turns human (or AI) preferences into a
> scalar signal a policy can be optimized against.

**You will:** train a Bradley-Terry **outcome reward model** on a real preference dataset
(`trl-lib/ultrafeedback_binarized`) end-to-end on a single A100/H100, score responses with it, and
then build a **process reward model (PRM)** that grades reasoning *step by step*.

> **Online/offline lens (recurring theme):** training the RM is itself *offline* supervised
> learning — a fixed dataset of comparisons, no generation loop. But the RM exists to enable
> *online* RL: in folder `03` (PPO) and `05` (GRPO) the policy generates fresh samples and the RM
> scores them on the fly. **The RM is the bridge from offline data to online optimization.**

---

## Table of contents
1. [Where reward models fit](#1-where-reward-models-fit)
2. [The Bradley-Terry model and the loss](#2-the-bradley-terry-model-and-the-loss)
3. [Architecture: a causal LM with a scalar head](#3-architecture-a-causal-lm-with-a-scalar-head)
4. [Preference data: where it comes from, what it looks like](#4-preference-data-where-it-comes-from-what-it-looks-like)
5. [What to watch while training](#5-what-to-watch-while-training)
6. [Reward hacking & over-optimization (the central pathology)](#6-reward-hacking--over-optimization-the-central-pathology)
7. [Outcome RM vs Process RM (PRM)](#7-outcome-rm-vs-process-rm-prm)
8. [RM-based RLHF vs reward-free DPO](#8-rm-based-rlhf-vs-reward-free-dpo)
9. [Honest verdict: are explicit RMs still used?](#9-honest-verdict-are-explicit-rms-still-used)
10. [Evaluation](#10-evaluation)
11. [Run it](#11-run-it)
12. [Common failure modes](#12-common-failure-modes)
13. [Exercises](#13-exercises)
14. [References](#14-references)

---

## 1. Where reward models fit

```
 SFT model ──► [ Reward Model ] ──► PPO / GRPO optimize the policy against the RM
   (01)          (this folder)         (03, 05)
                      ▲
              preference data
          (humans or a judge LLM rank pairs)
```

Classic **RLHF** is three stages: (1) SFT, (2) train an RM on preference comparisons, (3) use RL to
push the policy toward high RM-reward outputs while staying close to the SFT model. This folder is
stage 2 — the piece that converts "people prefer this answer" into a differentiable score.

Why not just SFT on the preferred answers? Because **"good" is often easier to *rank* than to
*write***. You may not be able to author the perfect response, but you can reliably say which of two
responses is better. Reward modeling is how we exploit that — and it lets us push the model *beyond*
the quality of any single demonstration (the behavioral-cloning ceiling from folder `01`).

---

## 2. The Bradley-Terry model and the loss

Given a prompt `x`, a chosen response `y⁺`, and a rejected response `y⁻`, we want a reward function
`r_θ(x, y)` that scores `y⁺` higher. The **Bradley-Terry** model (1952) says the probability that
`y⁺` is preferred is:

$$p(y^+ \succ y^- \mid x) = \sigma\big(r_\theta(x, y^+) - r_\theta(x, y^-)\big)$$

We train by maximizing the likelihood of the observed preferences — i.e. minimizing:

$$\mathcal{L}(\theta) = -\,\mathbb{E}_{(x,y^+,y^-)\sim\mathcal{D}}\Big[\log \sigma\big(r_\theta(x, y^+) - r_\theta(x, y^-)\big)\Big]$$

Read it plainly: **push the chosen reward above the rejected reward; the sigmoid+log makes the
penalty large when the model gets the ordering wrong and saturates once the margin is comfortable.**

Three consequences worth internalizing:

- **Only *differences* matter.** The loss depends on `r(x,y⁺) − r(x,y⁻)`, never on absolute values.
  The RM is therefore **underdetermined up to an additive constant** — add 5 to every reward and the
  loss is unchanged. This is why absolute reward values are *not* meaningful across models or even
  across training runs; only relative ranking is. (And it motivates reward centering — §6.)
- **It's a *relative* signal, not an absolute quality score.** An RM trained on "helpfulness" pairs
  learns the axis your annotators cared about — nothing more. Garbage/biased preferences → garbage
  reward axis.
- **No reasoning supervision.** The outcome RM sees only final responses and a binary preference. It
  has no idea *why* one is better. (That gap is exactly what PRMs in §7 address.)

> **Reward centering.** Because of the additive-constant freedom, reward scale can drift during
> training and destabilize downstream RL. TRL's `center_rewards_coefficient` (recommended `1e-2`)
> adds a small auxiliary loss pulling rewards toward zero-mean. We turn it on in `default.yaml`.

---

## 3. Architecture: a causal LM with a scalar head

An RM is **not** a generative model. We take a decoder LM and replace the language-modeling head
(which projects to `vocab_size` logits) with a **single linear layer projecting to one scalar** —
the reward. In transformers terms: `AutoModelForSequenceClassification` with `num_labels=1`. TRL's
`RewardTrainer` does this automatically; you just pass the base model name.

Practical details that bite people:

- **The score head is randomly initialized.** When you load a causal LM as a classifier, the new
  scalar head starts from noise — that's expected; training learns it. (For LoRA you *must* keep
  this head trainable: `modules_to_save=["score"]`. `train.py` adds it for you.)
- **Which base model?** Convention: initialize the RM from **your SFT model** (folder `01`'s output)
  so RM and policy share a backbone and tokenizer. This folder uses the released `Qwen/Qwen3-4B`
  instruct model so it stands alone — but swap `model_name_or_path` to your SFT checkpoint to do it
  the "real" way.
- **The reward is read at the last token.** The scalar is taken from the final non-pad position, so
  the RM scores the *complete* (prompt+response) sequence. Padding/truncation correctness matters
  (`max_length`).

---

## 4. Preference data: where it comes from, what it looks like

The data is the product. An RM is exactly as good as the preferences it's trained on.

### Format (TRL auto-applies the chat template for conversational data)
```python
# Conversational preference, implicit prompt (what UltraFeedback looks like):
{"chosen":   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "GOOD"}],
 "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "BAD"}]}

# Standard preference, explicit prompt:
{"prompt": "...", "chosen": "GOOD", "rejected": "BAD"}
```

### Where preferences come from
- **Human preferences (RLHF).** Annotators rank pairs. Highest quality, slow and expensive,
  noisy/subjective at the margins.
- **AI feedback (RLAIF).** A strong "judge" LLM ranks pairs (often with a rubric). Cheap and
  scalable; this is how most large preference sets (incl. UltraFeedback) are built today. Risk: you
  inherit the judge's biases.
- **Implicit/behavioral.** Thumbs-up/down, accept/reject, edits — real product signal, but messy.

> **Annotation pathologies to know:** annotators (human *and* LLM) systematically prefer **longer**,
> more **confident**, better **formatted** answers — independent of correctness. These biases get
> baked into the RM and become the levers a policy will later exploit (§6). Length bias is the most
> famous; mitigations include length-balancing the data and length-penalizing the reward.

---

## 5. What to watch while training

`RewardTrainer` logs these — read them, don't just watch loss:

- **`accuracy`** — fraction of pairs where the RM scored chosen > rejected. *This is the headline
  metric.* On UltraFeedback-style data, ~0.65–0.75 held-out is typical; >0.8 is strong (or a sign of
  an easy/leaky split). 0.5 means the RM learned nothing.
- **`margin`** — mean `r(chosen) − r(rejected)`. Should grow then stabilize. Exploding margin with
  flat/rising eval loss = overfitting.
- **`mean_reward` / `min` / `max`** — the reward distribution. With centering on, `mean_reward`
  should hover near 0. A distribution that runs away is a stability red flag.
- **Eval accuracy vs train accuracy** — the usual overfitting gap. RMs overfit *fast* (often <1
  epoch); preference data is smaller and noisier than SFT data.

Hyperparameters that matter most: **1 epoch** (RMs overfit quickly), **LR** (full-FT ~1–2e-5,
LoRA ~1e-3), **`max_length`** large enough not to truncate responses (truncation corrupts the
comparison), and **`center_rewards_coefficient`** for scale stability.

---

## 6. Reward hacking & over-optimization (the central pathology)

This is the most important concept in the folder, because the RM is a **proxy** for what you
actually want, and optimizing a proxy hard enough always finds its cracks. **Goodhart's law: when a
measure becomes a target, it ceases to be a good measure.**

In RLHF this shows up as **reward over-optimization**: as RL pushes the policy to higher *RM reward*,
true quality first rises, then **peaks and declines** even as RM reward keeps climbing. The policy
has found inputs where the RM is wrong — exploiting its blind spots rather than getting better.

Classic hacks a policy discovers:
- **Length exploitation** — pad answers because the RM (from biased data, §4) rewards length.
- **Sycophancy / over-confidence** — agree with the user, assert confidently; the RM likes it.
- **Format gaming** — bullet points, headers, boilerplate the RM associates with "good."
- **Distribution drift** — the policy generates text unlike anything in the RM's training data,
  where the RM's scores are essentially undefined.

Mitigations (you'll use these in folders `03`/`05`):
- **KL penalty to the reference (SFT) model.** The dominant lever: penalize drifting far from the
  SFT policy, keeping generations in-distribution for the RM. (Central to PPO and GRPO.)
- **Reward centering / normalization** — `center_rewards_coefficient` here; reward whitening in RL.
- **RM ensembles** — disagreement among RMs flags out-of-distribution inputs (Coste et al. 2023).
- **Better/length-debiased data**, and **iterative RM retraining** on fresh on-policy samples.
- **Early stopping by *true* eval**, not by RM reward — never trust the proxy as your final metric.

`score.py` lets you probe your own RM for these biases directly (try a correct-but-terse answer vs a
verbose-but-wrong one). **Internalize this now** — it's why half of online-RL engineering exists.

---

## 7. Outcome RM vs Process RM (PRM)

Everything above is an **Outcome Reward Model (ORM)**: one score for the *whole* response. For
multi-step reasoning (math, code, agents) that's a blunt instrument — a correct final answer can come
from flawed reasoning, and a single wrong step dooms an otherwise-sound chain, but the ORM gives one
number either way.

A **Process Reward Model (PRM)** scores **each reasoning step**. Architecturally it's a
**token-classification** model (`AutoModelForTokenClassification`, `num_labels=2`): it emits a
correct/incorrect label at each step boundary. It trains on **stepwise supervision** —
`prompt` + `completions` (list of steps) + `labels` (one per step):

```python
{"prompt": "Musa has 45 students...",
 "completions": ["Step 1: ...", "Step 2: ...", "Step 3: ... The answer is 20"],
 "labels": [True, False, False]}   # one boolean per step
```

| | Outcome RM (ORM) | Process RM (PRM) |
|---|---|---|
| Supervises | final response | every reasoning step |
| Model head | sequence classification (1 scalar) | token classification (per-step label) |
| Label cost | cheap (final answer check / pairwise) | expensive (step-level annotation) |
| Best for | general helpfulness, chat | math/code/agentic reasoning |
| Failure mode | rewards right answer, wrong reasoning | needs costly/automated step labels |
| Used in | classic RLHF, RM-as-verifier | reasoning RL, best-of-N reranking, search |

**Why PRMs matter now:** they give *dense, localized* feedback for reasoning, enable step-level
reranking / tree search at inference, and inform reasoning-RL. The honest caveat from the original
process-vs-outcome paper (Uesato et al. 2022) still holds: for *final-answer* accuracy, outcome
supervision is often competitive and far cheaper; process supervision wins on *reasoning* validity.
Modern recipes increasingly **automate** step labels (e.g. Math-Shepherd's rollout-based labeling)
to make PRMs affordable. The trade-off — dense correctness signal vs labeling cost — is the thing to
remember.

> `train_prm.py` + `configs/prm.yaml` train a small PRM on `trl-lib/math_shepherd`. PRMs are small
> by design (step data is scarce), so a 0.6B model is appropriate. Note `PRMTrainer` is in
> `trl.experimental` — a research API that may change.

---

## 8. RM-based RLHF vs reward-free DPO

You'll hear "DPO doesn't need a reward model." True, and it's worth understanding *why* before
folder `04`. **DPO** (Direct Preference Optimization) shows that for the specific RLHF objective, the
optimal policy has a closed form, so you can train *directly on preference pairs* and skip both the
explicit RM and the RL loop. The reward becomes *implicit* in the policy itself.

So when do you still train an explicit RM?

- **You need a reusable scorer**, not just a tuned policy — for RL (PPO/GRPO), best-of-N sampling /
  reranking, data filtering, or eval. DPO gives you a policy, not a portable reward function.
- **Online RL.** PPO/GRPO need a reward callable on *fresh on-policy samples*; DPO is offline on a
  fixed pair set.
- **Verifiable/process rewards.** RLVR (folder `05`) and PRMs are reward-model territory by nature.

Rule of thumb: **just want a better chat model from preference pairs? → DPO (no RM).** **Need to
optimize against a signal online, or reuse a scorer? → train an RM.**

---

## 9. Honest verdict: are explicit RMs still used?

**Yes — but the center of gravity moved.** A nuanced, current picture:

- **For preference tuning of chat models, explicit RMs were largely displaced by DPO & friends**
  (folder `04`). Most teams aligning a chat model on preference pairs no longer train a standalone
  Bradley-Terry RM. This is the biggest shift since classic RLHF.
- **But RMs came roaring back through RL-with-verifiable-rewards and reasoning.** GRPO/RLVR (folder
  `05`) optimize against a *reward* — often a programmatic verifier (does the test pass? is the math
  answer correct?), sometimes a learned RM or PRM. Reward design *is* the job there.
- **RM-as-judge / LLM-as-judge** is now ubiquitous for eval and data curation, a close cousin of the
  Bradley-Terry RM.
- **PRMs are an active frontier** for reasoning models (reranking, search, process feedback).

So learn the Bradley-Terry RM not because you'll always train one for chat alignment, but because
**reward modeling is the conceptual core of all RL post-training** — and the failure modes (§6) are
universal. Skip this and PPO/GRPO will feel like magic that mysteriously breaks.

---

## 10. Evaluation

- **Held-out preference accuracy** (logged as `eval_accuracy`) — the primary number. `train.py`
  prints it at the end.
- **RewardBench-style evaluation** — the community standard: accuracy across curated chat / safety /
  reasoning / chat-hard categories. Run your RM against it to see *where* it's reliable, not just an
  aggregate. (We don't bundle it to keep the folder self-contained; it's the natural next step.)
- **Calibration & reward distributions** — plot reward histograms for chosen vs rejected; healthy
  RMs show separated, sensibly-scaled distributions, not a runaway scale.
- **Probe for known biases** — use `score.py` on adversarial pairs (terse-correct vs verbose-wrong)
  to measure length/sycophancy bias *before* you optimize against the RM.
- **PRM:** step-level accuracy (does the predicted step label match the gold label?), as in the
  `train_prm.py` doc example.

> The RM's *real* test is downstream: does optimizing against it actually improve the policy (by true
> eval), or does it get hacked (§6)? An RM with great held-out accuracy can still be a bad optimization
> target. Always validate end-to-end.

---

## 11. Run it

### Setup
```bash
cd 02-reward-modeling
uv sync                 # base deps
uv sync --extra flash   # optional: FlashAttention-2 speedup for the headline run
uv run hf auth login
uv run wandb login && export WANDB_PROJECT=post-training-lab
```

### 1) Smoke test first (always)
```bash
uv run python train.py --config configs/smoke.yaml
```

### 2) Headline outcome-RM run (single A100/H100)
```bash
uv run python train.py --config configs/default.yaml
```
Full fine-tune of Qwen3-4B as a Bradley-Terry RM on 30k UltraFeedback pairs, 1 epoch. Watch
`accuracy`/`margin` in the wandb run. Outputs in `outputs/qwen3-4b-ultrafeedback-rm/`.

### 3) Score responses with your RM
```bash
uv run python score.py --model_path outputs/qwen3-4b-ultrafeedback-rm
# probe for bias:
uv run python score.py --model_path outputs/qwen3-4b-ultrafeedback-rm \
  --prompt "What's 17 x 23?" \
  --responses "391." \
              "Great question! Let me walk you through this step by step in detail... the answer is 392."
```

### 4) Advanced: train a Process Reward Model
```bash
uv run python train_prm.py --config configs/prm.yaml
# quick smoke: uv run python train_prm.py --config configs/prm.yaml --max_train_samples 256 --max_steps 5
```

### Multi-GPU — no code changes
```bash
uv run accelerate launch train.py --config configs/default.yaml
uv run torchrun --nproc_per_node=4 train.py --config configs/default.yaml
```

---

## 12. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Eval accuracy stuck at ~0.5 | LR too high/low, or label/format mismatch | sanity-check data fields; try LR 1–2e-5; verify chat template applied |
| Train acc ≫ eval acc | overfitting (RMs overfit fast) | 1 epoch, fewer steps, more/cleaner pairs |
| Rewards explode / NaN | scale drift, fp16 overflow | enable `center_rewards_coefficient`, use bf16, lower LR |
| RM prefers longer/wordier answers always | length bias in data | length-balance data, length-penalize reward, debias pairs |
| Downstream RL improves RM reward but quality drops | reward hacking / over-opt (§6) | KL penalty, RM ensemble, early-stop on true eval |
| OOM | optimizer state + 2× sequences per pair | grad checkpointing (on), smaller batch + more grad-accum, LoRA |
| Truncated responses | `max_length` too small | raise `max_length` to cover real response lengths |
| LoRA RM doesn't learn | score head frozen | ensure `modules_to_save=["score"]` (handled in `train.py`) |
| `PRMTrainer` import error / API change | it's `trl.experimental` | pin TRL; check the experimental API for your version |

---

## 13. Exercises

1. **Measure length bias.** Use `score.py` on 10 (terse-correct, verbose-wrong) pairs. How often does
   your RM prefer the verbose wrong answer? Quantify the bias you'll later have to defend against.
2. **Centering ablation.** Train with `center_rewards_coefficient: 0.0` vs `0.01`. Compare the
   `mean_reward` trajectory and reward histograms. What did centering buy?
3. **Init from your SFT model.** Point `model_name_or_path` at folder `01`'s output instead of the
   released instruct model. Does sharing the policy backbone change held-out accuracy?
4. **ORM vs PRM intuition.** On a Math-Shepherd example, compare what the ORM (final score) and the
   PRM (per-step labels) tell you about a solution with a correct answer but a wrong middle step.
5. **Overfit on purpose.** Train 3 epochs and watch eval accuracy peak then fall. Find the epoch
   where the RM is best — and note it's *not* where train loss is lowest.

---

## 14. References

- TRL `RewardTrainer` — https://huggingface.co/docs/trl/reward_trainer
- TRL `PRMTrainer` — https://huggingface.co/docs/trl/prm_trainer
- Bradley & Terry (1952), rank analysis of paired comparisons — https://www.jstor.org/stable/2334029
- InstructGPT (RLHF with a learned RM) — https://arxiv.org/abs/2203.02155
- UltraFeedback — https://arxiv.org/abs/2310.01377
- Reward over-optimization (Gao et al. 2022) — https://arxiv.org/abs/2210.10760
- RM ensembles vs reward hacking (Coste et al. 2023) — https://arxiv.org/abs/2312.09244
- Process vs outcome supervision (Uesato et al. 2022) — https://arxiv.org/abs/2211.14275
- Let's Verify Step by Step (PRMs, Lightman et al. 2023) — https://arxiv.org/abs/2305.20050
- Math-Shepherd (automated PRM labels) — https://arxiv.org/abs/2312.08935
- RewardBench — https://arxiv.org/abs/2403.13787

---

**Next:** [`03-ppo`](../03-ppo) — now that we can *score* responses, classic RLHF closes the loop:
use RL to push the policy toward high reward. We'll build the intuition for PPO — and confront why,
despite defining the RLHF era, it's been largely superseded in practice.
