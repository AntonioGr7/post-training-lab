# 05 · GRPO / RLVR (Group Relative Policy Optimization)

> **The flagship folder — online RL, done right for reasoning.** GRPO is how DeepSeek-R1 was trained:
> it keeps PPO's online loop (generate, score, improve on your own samples) but throws out the two
> things that made PPO painful. **No value model** (the group of samples *is* the baseline) and **no
> learned reward model** (the reward is a *programmatic verifier* — RLVR). The result is online RL
> that fits one GPU, rarely diverges, and produces emergent step-by-step reasoning.

**You will:** train Qwen3-4B to solve grade-school math (GSM8K) with **verifiable rewards** — the
reward is literally "is the final answer correct?" — watch accuracy and chain-of-thought length
climb, design reward functions, drive generation with vLLM, and learn the frontier debate about
GRPO's trust region (the **DPPO** critique).

> **Online/offline lens — the payoff of folders 03→04→05:** PPO (folder `03`) was online but heavy
> and fragile. DPO (folder `04`) was stable but **offline — it can't explore beyond its preference
> pairs**. GRPO goes back online to recover exploration, but pays only a fraction of PPO's cost. This
> is the synthesis: *online exploration without the critic, without a reward model, without the KL
> babysitting.* When your reward is verifiable, this is the method.

---

## Table of contents
1. [Where GRPO sits: online RL, minus the baggage](#1-where-grpo-sits-online-rl-minus-the-baggage)
2. [The core idea: group-relative advantage (no critic)](#2-the-core-idea-group-relative-advantage-no-critic)
3. [RLVR: verifiable rewards replace the reward model](#3-rlvr-verifiable-rewards-replace-the-reward-model)
4. [The GRPO loop, step by step](#4-the-grpo-loop-step-by-step)
5. [Designing reward functions](#5-designing-reward-functions)
6. [Hyperparameters & what to watch](#6-hyperparameters--what-to-watch)
7. [vLLM & the training–inference mismatch](#7-vllm--the-traininginference-mismatch)
8. [The trust region & the DPPO frontier](#8-the-trust-region--the-dppo-frontier)
9. [Honest verdict: is GRPO SOTA?](#9-honest-verdict-is-grpo-sota)
10. [Run it](#10-run-it)
11. [Common failure modes](#11-common-failure-modes)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Where GRPO sits: online RL, minus the baggage

```
 SFT (01) ──► GRPO (this folder): RL directly against a VERIFIER, no reward model, no critic
                  │
   ┌──────────────┴───────────────────────────────────────────────┐
   │  repeat, per prompt:                                            │
   │   1. policy generates a GROUP of G completions   ← online       │
   │   2. a VERIFIER scores each (correct? formatted?) ← RLVR         │
   │   3. advantage = (reward − group mean) / group std ← no critic   │
   │   4. push up above-average completions, down below-average      │
   └─────────────────────────────────────────────────────────────────┘
```

GRPO is a **variant of PPO** (folder `03`). Recall PPO's four models: policy, reference, reward
model, **value model**. GRPO deletes two of them:

| Model | PPO | GRPO |
|---|---|---|
| Policy (trained) | ✅ | ✅ |
| Value/critic (trained) | ✅ | ❌ **gone** — group mean is the baseline |
| Reward model (frozen) | ✅ | ❌ **gone** — a verifier function instead (RLVR) |
| Reference (frozen) | ✅ | **optional** — only if `beta > 0` (KL term) |

So a typical GRPO run holds **one trained model** (plus an *optional* frozen reference). That's
roughly an SFT-sized footprint, with online RL's exploration on top. The catch GRPO inherits from
PPO is that **generation is in the loop** — so generation is the bottleneck, and §7 (vLLM) matters.

---

## 2. The core idea: group-relative advantage (no critic)

PPO needs a value model to answer "was this token better than expected?" GRPO answers it for *free*
by sampling a **group** of `G` completions for the *same* prompt and comparing them to each other.

For prompt `q`, sample completions `o_1 … o_G`, score each to get rewards `r_1 … r_G`. The advantage
of completion `i` is just its reward standardized within the group:

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$

That's the whole trick that gives the method its name. A completion that beat its group-mates gets a
positive advantage (push its tokens *up*); one that did worse gets negative (push *down*). The
group's mean reward **is** the baseline — no learned critic, no GAE, no `vf_coef` to balance. This is
exactly the model (and the instability source) that PPO §3/§7 spent a whole section on; GRPO simply
removes it.

The policy-gradient update then uses PPO's familiar clipped surrogate (the ratio `π_θ/π_θ_old`
clipped to `1±ε`) to take safe steps, optionally minus a `β·KL` penalty to a reference:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\,\mathbb{E}\Big[\min\big(\rho_{i,t}\hat{A}_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i\big) - \beta\,\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_\text{ref}]\Big]$$

> **A modern wrinkle the defaults encode:** TRL defaults `beta = 0.0` — i.e. **no KL term and no
> reference model at all**. Recent work (R1, DAPO, Open-Reasoner-Zero) found the KL leash is often
> unnecessary for verifiable-reward RL and just costs memory/throughput. We keep `beta = 0.0` and
> explain when to turn it back on (§6). This is the opposite of PPO, where the KL leash was *the*
> master dial — because there the reward was a hackable learned model; here it's ground truth (§3).

---

## 3. RLVR: verifiable rewards replace the reward model

**RLVR = RL with Verifiable Rewards.** The single biggest reason GRPO took off for reasoning: in
domains with a checkable answer (math, code, formal proofs), you don't *need* a learned reward model
at all. You **verify**:

- **Math (this folder):** parse the model's final answer, compare to the gold answer → reward 1/0.
- **Code:** run unit tests → reward = fraction passing.
- **Format:** regex-check that the output has the required structure → small shaping reward.

Contrast with the learned reward model of folders `02`/`03`. That RM was a *proxy*, and optimizing a
proxy hard invites **reward hacking / over-optimization** (folder `02` §6): the policy finds
gibberish the RM mistakenly loves. A verifier has **no proxy gap** for the core signal — a correct
GSM8K answer is correct. This makes the reward cheap, dense enough, and far harder to hack on the
dimension that matters. It's why the whole reward-modeling apparatus of `02`/`03` can be skipped here.

> **The honest caveat:** RLVR only applies where you *can* verify. "Be more helpful / less toxic /
> better styled" is **not** verifiable — that's preference territory, so you'd use DPO (`04`) or a
> reward model + online RL (`06`). And even verifiable rewards can be *gamed on unspecified axes*
> (e.g. a format reward that the model satisfies without solving anything — §5). RLVR shrinks the
> reward-hacking surface; it doesn't eliminate it.

---

## 4. The GRPO loop, step by step

Each training step (see [train.py](train.py) + [rewards.py](rewards.py)):

1. **Sample prompts** and, for each, **generate `num_generations` (G) completions** on-policy
   (`temperature` > 0 so the group is *diverse* — see §6). ← the expensive, online part; vLLM (§7).
2. **Verify.** Run the reward functions on every completion. Here: `correctness_reward` (answer
   matches gold) + `format_reward` (parseable `\boxed{}`). Multiple rewards are **summed** (optionally
   `reward_weights`).
3. **Group-relative advantages.** Standardize rewards within each prompt's group (§2). If all G
   completions in a group score identically, std = 0 → **zero advantage, zero learning signal** for
   that prompt (watch `frac_reward_zero_std`, §6).
4. **Optimize.** Clipped-surrogate policy-gradient update (for `num_iterations` passes), optionally
   minus `β·KL`. Token-level loss normalization per `loss_type` (§8).
5. **Repeat.** No critic to update, no reward model to maintain — just policy steps.

Emergent behavior to look for: over training the model's completions get **longer and more
structured** (it learns that working through steps raises the chance of a correct boxed answer). That
emergent chain-of-thought, with *no supervised reasoning traces*, is the headline DeepSeek-R1 result.

---

## 5. Designing reward functions

A TRL reward function takes `completions` (and any dataset column as a kwarg) and returns a
`list[float]`. Ours, in [rewards.py](rewards.py):

```python
def correctness_reward(completions, ground_truth, **kwargs) -> list[float]:
    # 1.0 if the extracted \boxed{} answer equals the gold answer, else 0.0  ← the real signal
def format_reward(completions, **kwargs) -> list[float]:
    # 0.2 if a well-formed \boxed{...} is present                            ← small shaping term
```

Principles that matter in practice:

- **Make the verifiable signal dominant.** Correctness is worth 1.0; format only 0.2. If format
  rewards rival correctness, the policy learns to emit the *format* without solving the problem —
  reward hacking on the cheap axis. Keep shaping rewards small and bounded.
- **Robust extraction.** Models phrase answers a dozen ways. We prefer `\boxed{...}`, fall back to
  the last number, and normalize (strip commas/`$`/trailing dots). Brittle parsers *undercount*
  correct answers → a noisy, weaker reward.
- **Dense beats sparse where you can afford it.** Pure 0/1 correctness on hard problems can mean
  *every* completion in a group scores 0 (std = 0, no signal). A format/partial-credit term keeps
  some gradient flowing early. (For code: fraction of tests passed > all-or-nothing.)
- **`None` for "not applicable."** A reward function may return `None` for samples it doesn't apply
  to (e.g. multi-task: a math reward returns `None` on coding prompts). TRL skips them.

TRL also ships ready-made rewards (`from trl.rewards import accuracy_reward`, math-verify based) and
supports **reward models** as `reward_funcs` (pass a model path → it's loaded as a scalar classifier,
exactly folder `02`'s RM). We hand-roll functions here so the mechanics are visible and dependency-light.

---

## 6. Hyperparameters & what to watch

**Knobs that matter (in rough priority):**

| Param | Default here | What it does / how to tune |
|---|---|---|
| `num_generations` (G) | 8 | Group size = baseline quality. Larger G ⇒ lower-variance advantage but linearly more generation. 8–16 typical. |
| `temperature` | 1.0 | **Diversity within a group.** Too low ⇒ identical completions ⇒ std≈0 ⇒ no signal. Online RL *needs* exploration; don't drop this to 0. |
| `max_completion_length` | 1024 | Room to reason. Dominates generation cost. Too short truncates the CoT and the answer. |
| `beta` (KL) | 0.0 | KL-to-reference. **0 = no KL, no reference model** (modern default). Set ~0.04 to re-add the leash + a frozen reference (more stable, more memory). |
| `loss_type` | `dapo` | Token-level loss normalization (§8). `dapo`/`dr_grpo` fix the length bias of plain `grpo`. |
| `scale_rewards` | `true`/`"group"` | Divide advantage by group std. `"none"` (Dr.GRPO) avoids a difficulty bias; `"batch"` is a robust middle ground. |
| `learning_rate` | 1e-6 | Online RL ⇒ tiny LR (like PPO, unlike SFT). |
| `epsilon` | 0.2 | PPO-style ratio clip — the trust region (§8). |
| `num_iterations` (μ) | 1 | Optimization passes per generation batch. >1 reuses rollouts (cheaper) but drifts off-policy. |

**Metrics to watch (TRL logs all of these):**

- **`reward`** — mean total reward. *The* success signal; should climb. With our rewards, ~1.2 max
  (1.0 correct + 0.2 format).
- **`reward/correctness_reward/mean`** — task accuracy, basically. Watch this, not just `reward`.
- **`frac_reward_zero_std`** — fraction of prompts where every completion scored the same (std=0 ⇒
  **no gradient**). High and rising = your prompts are too easy or too hard for the current policy, or
  `temperature` is too low. A key health metric unique to group methods.
- **`completions/mean_length`** — should *grow* as the model learns to reason. A sudden collapse to
  very short or a blow-up to max-length (then truncation) are both red flags.
- **`kl`** — only logged if `beta > 0`. If you enabled the leash, watch it stay bounded.
- **`clip_ratio/*`** — how often the trust-region clip bites. Persistently high ⇒ updates slamming the
  wall ⇒ instability (this is exactly the quantity §8's DPPO critique is about).
- **`reward_std`** — diversity of rewards across the batch; collapsing toward 0 means learning stalls.

> **Watch the actual completions** (`log_completions: true`). RLVR's verifier guards the *answer*, but
> the model can still drift into degenerate reasoning, language-mixing, or format gaming that the
> numbers hide. Read the sample tables.

---

## 7. vLLM & the training–inference mismatch

Generation is GRPO's wall-clock bottleneck (just like PPO). The standard fix is **vLLM**, a
high-throughput inference engine. TRL offers two modes:

- **Colocate** (`vllm_mode: colocate`, our default): vLLM shares the training GPU. One process, no
  server — ideal for **single-GPU**. Tune `vllm_gpu_memory_utilization` (we use 0.3) so vLLM's KV
  cache and the training model + activations coexist without OOM.
- **Server** (`vllm_mode: server`): vLLM runs on *separate* GPUs via `trl vllm-serve`. Best when you
  have dedicated inference GPUs. Don't share GPUs with the trainer or you'll hit NCCL errors.

```yaml
use_vllm: true
vllm_mode: colocate
vllm_gpu_memory_utilization: 0.3
```
*(Requires `uv sync --extra vllm`. The smoke config sets `use_vllm: false` and uses plain
`generate()` so it runs with no extra dependency.)*

**The training–inference mismatch (know this one).** vLLM and the training framework are
*mathematically* the same policy but *numerically* different (different kernels, precision,
batching). So completions are effectively sampled from `π_inference` while gradients are taken w.r.t.
`π_train` — turning on-policy RL subtly **off-policy**, which can bias the gradient and, on long
rollouts, destabilize training. TRL corrects this by default with **Truncated Importance Sampling**
(`vllm_importance_sampling_correction=True`): it reweights by `π_train/π_inference`, clipping outliers.
This mismatch is a live research area — and a direct lead-in to §8.

---

## 8. The trust region & the DPPO frontier

GRPO inherits PPO's **ratio-clipping trust region**: bound `ρ = π_θ/π_θ_old` to `[1−ε, 1+ε]` so no
single update moves the policy too far (folder `03` §4). It works, but the frontier is actively
rethinking it.

**What TRL already ships** — alternative `loss_type`s that refine the trust region/normalization:
- **`dapo`** (our default) & **`dr_grpo`**: fix the *length bias* of the original `grpo` loss
  (sample-level normalization under-penalizes long wrong answers). Token-level / constant
  normalization instead.
- **`sapo`**: replaces *hard* clipping with a **smooth, temperature-controlled soft gate** — instead
  of zeroing the gradient outside the trust region, it decays it, keeping signal from
  "near-on-policy" tokens while damping extreme deviations.
- **Importance sampling** (TIS/MIS, §7): corrects the train/inference mismatch.

**The frontier critique — DPPO (arXiv 2602.04879, "Rethinking the Trust Region in LLM RL").** The
argument: **ratio clipping is the wrong trust region.** Because it acts on the *probability ratio*, it
**over-penalizes low-probability tokens** (slowing learning) and **under-penalizes high-probability
tokens** (allowing large, destabilizing updates). On long, on-policy rollouts — long-horizon and
**agentic** RL — this asymmetry compounds into a growing training–inference mismatch and eventual
**collapse**. DPPO (Divergence Proximal Policy Optimization) replaces ratio-clipping with a
**divergence-based trust region** (Total-Variation or KL, via cheap *Binary* and *Top-K*
approximations). Reported to beat GRPO on both **stability and final performance** (AIME24,
Qwen3-30B-A3B).

> **Practitioner status (2026-06):** DPPO is implemented in **verl**
> (`LOSS_MODE=dppo_tv` / `dppo_kl`; `vanilla` = GRPO baseline) —
> [verl DPPO docs](https://verl.readthedocs.io/en/latest/algo/dppo.html) — and is **not in TRL** as
> of this writing. TRL's nearest in-house answer is `loss_type="sapo"` (soft trust region) plus the
> importance-sampling corrections. **Teach GRPO hands-on here (TRL); reach for verl if you hit
> trust-region collapse on long-horizon/agentic rollouts.** Re-check whether TRL has added a
> divergence loss when you revisit this folder.

The takeaway: GRPO is the workhorse, but the *trust region itself* is not settled science — and the
failure mode it papers over (ratio-clip asymmetry on long rollouts) is exactly where the next round
of methods is competing.

---

## 9. Honest verdict: is GRPO SOTA?

**Yes — for verifiable-reward / reasoning RL, GRPO (and its close kin RLOO, DAPO, etc.) is the
current default, and this is the most actively-developed area in post-training.** It delivered the
DeepSeek-R1 reasoning breakthrough, it runs on accessible hardware, and it sidesteps both of PPO's
worst problems (the critic, the hackable reward model).

Be precise about the boundaries:

- **It needs a verifier.** GRPO's superpower is RLVR — and that only exists where you can *check*
  correctness (math, code, formal tasks). For taste/helpfulness/safety (no verifier), use DPO (`04`)
  or online RL against a reward model (`06`). Don't force a verifier where none is honest.
- **Generation-bound.** Like all online RL, it's slow without fast inference (§7). vLLM is not
  optional at scale.
- **The trust region is contested.** On long-horizon/agentic rollouts, GRPO's ratio-clipping can
  destabilize — the DPPO frontier (§8). For frontier long-horizon work, watch this space (verl).
- **Reward hacking is reduced, not gone.** Shaping rewards and unspecified axes can still be gamed
  (§5). Read the completions.

> **The arc completes:** PPO (`03`) taught the machinery; DPO (`04`) showed you can skip the loop when
> you only have preferences; GRPO (`05`) shows that when you have a *verifier*, you go back online and
> get genuine exploration — emergent reasoning — for an SFT-sized cost. Folder `06` then tackles the
> infrastructure (online DPO, reward-model-in-the-loop, eval harnesses, reward-hacking defenses) that
> makes online methods production-real.

---

## 10. Run it

### Setup
```bash
cd 05-grpo
uv sync --extra vllm          # vLLM powers the headline run's generation (single-GPU colocate)
uv run hf auth login
uv run wandb login && export WANDB_PROJECT=post-training-lab
```
> The committed `uv.lock` pins **torch 2.11** here (vLLM's upper bound), slightly behind folders
> 01–04's 2.12 — that's the real cost of bundling vLLM, and the folder is still self-contained.

### 1) Smoke test first (no vLLM needed)
```bash
uv run python train.py --config configs/smoke.yaml
```
Qwen3-0.6B, 64 prompts, G=4, 5 steps (~minutes). Confirms the loop spins: sample a group → verify →
group-relative update. Uses plain `generate()`, so no vLLM dependency.

### 2) The headline run
```bash
uv run accelerate launch train.py --config configs/default.yaml
```
Qwen3-4B on GSM8K with verifiable rewards + vLLM. Watch **`reward/correctness_reward/mean`** climb,
**`completions/mean_length`** grow (emergent reasoning), and **`frac_reward_zero_std`** stay low, in
wandb — and read the logged completion tables.

> ⏱ **Expected runtime (rough):** ~**4–8 h** on a single A100-80GB for 5k prompts × G=8. **Estimate,
> not measured** — GRPO is *generation-bound* (like PPO, assume ±2×); vLLM and `num_generations` /
> `max_completion_length` dominate. H100 ≈ 0.5–0.6×. Scales with prompts × G × completion length.

### 3) Measure the gain
```bash
uv run python generate.py --model_path Qwen/Qwen3-4B                    # baseline accuracy
uv run python generate.py --model_path outputs/qwen3-4b-gsm8k-grpo      # after GRPO
```
Reports GSM8K exact-match accuracy (same verifier used as the training reward — that symmetry is the
point of RLVR) and prints worked solutions so you can read the reasoning.

### LoRA / multi-GPU
Uncomment the `use_peft` block in the config for the low-memory path (reference = adapter disabled).
For scale, `accelerate launch` with DeepSpeed ZeRO-3 and/or vLLM **server mode** on dedicated GPUs
(`trl vllm-serve --model ...`, then `vllm_mode: server`).

### Try a different trust region — no code change
```bash
uv run accelerate launch train.py --config configs/default.yaml --loss_type dr_grpo   # Dr.GRPO
uv run accelerate launch train.py --config configs/default.yaml --loss_type sapo      # soft trust region
uv run accelerate launch train.py --config configs/default.yaml --beta 0.04           # re-enable the KL leash
```

---

## 11. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `reward` flat near 0 | every completion wrong ⇒ std 0 ⇒ no signal; or LR too low | add a format/partial-credit reward, raise `temperature`, use easier prompts first, check the verifier parses answers |
| `frac_reward_zero_std` high | groups not diverse (temp too low) or prompts too easy/hard | raise `temperature`, raise G, curriculum the data |
| `completions/mean_length` explodes → truncation | length gaming / no termination | shorten `max_completion_length` is *not* the fix — check reward shaping; consider `mask_truncated_completions`; `dapo`/`dr_grpo` loss |
| Reward rises but answers look wrong | reward hacking on a shaping term (e.g. format) | shrink the shaping reward relative to correctness; tighten the verifier |
| OOM with vLLM | colocate memory contention | lower `vllm_gpu_memory_utilization`, lower batch/`G`, enable vLLM sleep mode, or use LoRA / server mode |
| Training destabilizes on long rollouts | ratio-clip trust-region asymmetry (§8) | try `loss_type: sapo`, ensure importance-sampling correction is on; consider verl/DPPO for long-horizon |
| Accuracy stalls then degrades | over-optimization / off-policy drift (`num_iterations`>1) | set `num_iterations: 1`, lower LR, add small `beta` |
| `kl` not logged | `beta = 0` (no KL term) — expected | set `beta > 0` to enable the leash + reference |

---

## 12. Exercises

1. **Feel the group.** Train with `num_generations` = 4, 8, 16. Plot `reward` variance and wall-clock.
   Find the point of diminishing returns on baseline quality vs generation cost.
2. **Kill diversity.** Set `temperature: 0.3`. Watch `frac_reward_zero_std` rise and learning stall.
   Explain why a *group* method dies without intra-group diversity.
3. **Reward hacking on cue.** Crank `format_reward` to 1.0 (equal to correctness). Observe the model
   emit perfect `\boxed{}` formatting while accuracy stagnates. Then fix it.
4. **Length bias.** Compare `loss_type: grpo` vs `dapo` vs `dr_grpo`. Measure `completions/mean_length`
   and accuracy. Reproduce the length-bias finding.
5. **KL on/off.** Run `beta: 0.0` vs `0.04`. Compare stability, memory, and final accuracy. Decide
   whether the leash earns its cost for a verifiable task.
6. **GRPO vs PPO (cross-folder).** Conceptually map each PPO model/loss term (folder `03`) to its GRPO
   counterpart or its deletion. Quantify the memory GRPO saves by dropping the critic.
7. **GRPO vs DPO (cross-folder).** GSM8K answers are verifiable; UltraFeedback prefs (folder `04`) are
   not. Argue which method fits which, and why you *couldn't* swap them.
8. **The frontier.** Read the DPPO paper (§8) and the verl `dppo_tv`/`dppo_kl` docs. Describe the
   ratio-clip asymmetry in your own words and when it would bite your task.

---

## 13. References

- GRPO / DeepSeekMath (Shao et al. 2024) — https://arxiv.org/abs/2402.03300
- DeepSeek-R1 (incentivizing reasoning via RL, 2025) — https://arxiv.org/abs/2501.12948
- Understanding R1-Zero-Like Training / Dr.GRPO (Liu et al. 2025) — https://arxiv.org/abs/2503.20783
- DAPO (open-source RL at scale, 2025) — https://arxiv.org/abs/2503.14476
- Open-Reasoner-Zero (2025) — https://arxiv.org/abs/2503.24290
- SAPO (soft trust region, Qwen team) — https://arxiv.org/abs/2511.20347
- **DPPO — Rethinking the Trust Region in LLM RL (2026)** — arXiv 2602.04879 · verl: https://verl.readthedocs.io/en/latest/algo/dppo.html
- Lite PPO / "Tricks or Traps" (2025) — https://arxiv.org/abs/2508.08221
- TRL `GRPOTrainer` docs — https://huggingface.co/docs/trl/grpo_trainer
- KL approximation (Schulman) — http://joschu.net/blog/kl-approx.html

---

**Next:** [`06-online-infra`](../06-online-infra) — the infrastructure that makes online methods
production-real: **Online DPO** (preferences, but generated on-policy), vLLM-backed sampling loops,
**reward-hacking** mitigation, and a lightweight **eval harness** tying the course together. GRPO gave
you online RL with a verifier; folder `06` handles online RL when the signal is a *reward model*, and
how to keep it honest.
