# 06 · Online Infrastructure — Online DPO, vLLM, reward hacking & eval

> **Where the course's recurring lens finally closes.** Every folder asked: *online or offline — do
> you train on fresh samples from the current policy, or on a fixed dataset?* This folder makes that
> the subject. **Online DPO** keeps offline DPO's simple loss but generates its preference pairs
> *on-policy, every step* — eliminating the staleness that offline DPO (folder `04`) couldn't escape.
> And because it scores those generations with a *learned* reward model, **reward hacking is back** —
> so this is also the course's home for the infrastructure that keeps online RL honest: KL control,
> EOS/length defenses, judges, and a real **eval harness**.

**You will:** run Online DPO (Qwen3-4B) scored on-policy by *folder `02`'s reward model* — tying the
course together — accelerate it with vLLM, learn to detect and mitigate reward hacking, build a
reward-model-as-judge eval harness, and place the rest of TRL's experimental online family (RLOO,
NashMD, XPO) on the map.

> **Online/offline lens — the synthesis:** PPO (`03`) online + heavy. DPO (`04`) offline + simple but
> can't escape its pair distribution. GRPO (`05`) online + verifiable, but only where a verifier
> exists. **Online DPO (`06`) is the fourth quadrant: online + preference-based** — for taste/quality
> signals (no verifier) where you still want on-policy freshness. This folder is less a single new
> algorithm than the *operational layer* that makes any online method shippable.

---

## Table of contents
1. [Where this folder sits: completing the lens](#1-where-this-folder-sits-completing-the-lens)
2. [Online DPO / OAIF: the mechanism](#2-online-dpo--oaif-the-mechanism)
3. [Why on-policy matters: staleness & off-policy drift](#3-why-on-policy-matters-staleness--off-policy-drift)
4. [The scorer: reward model vs LLM judge](#4-the-scorer-reward-model-vs-llm-judge)
5. [Reward hacking, returned — and how to fight it](#5-reward-hacking-returned--and-how-to-fight-it)
6. [vLLM & the memory reality of online methods](#6-vllm--the-memory-reality-of-online-methods)
7. [The experimental online family (RLOO, NashMD, XPO)](#7-the-experimental-online-family-rloo-nashmd-xpo)
8. [The eval harness](#8-the-eval-harness)
9. [Honest verdict](#9-honest-verdict)
10. [Run it](#10-run-it)
11. [Common failure modes](#11-common-failure-modes)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. Where this folder sits: completing the lens

The course's four alignment quadrants, finally all on the table:

| | **Offline** (fixed data) | **Online** (on-policy generation) |
|---|---|---|
| **Preference signal** (taste, no verifier) | **DPO** (`04`) | **Online DPO** (this folder) |
| **Reward signal** (RM or verifier) | (RM training, `02`) | **PPO** (`03`, RM) · **GRPO** (`05`, verifier) |

Online DPO fills the last cell: you have *preference-style* feedback (a reward model or a judge, not a
checkable answer), **and** you want the on-policy freshness that offline DPO lacks. The price is the
same as every online method — generation in the loop, more models in memory, and a *learned* scorer
you can over-optimize. The back half of this lesson is about paying that price safely.

---

## 2. Online DPO / OAIF: the mechanism

Online DPO (Guo et al. 2024, "Online AI Feedback") is almost embarrassingly simple given folder `04`:

```
each training step, per prompt:
  1. sample TWO completions from the CURRENT policy        (y1, y2 ~ π_θ)   ← on-policy
  2. score both with a reward model (or LLM judge)         (r1, r2)
  3. chosen = argmax(r), rejected = argmin(r)              ← build a pair on the fly
  4. take ONE offline-DPO step on (prompt, chosen, rejected)
```

That's it: **DPO's exact loss** (folder `04` §2 — the log-ratio sigmoid with `beta`), but the pair is
manufactured each step from the model's *own current* outputs and labeled by the scorer. No value
model, no PPO clip — it inherits DPO's simplicity. What it adds back is generation and a scorer.

TRL exposes the familiar DPO knobs (`beta`, `loss_type` ∈ {`sigmoid`, `ipo`}) plus generation knobs
(`max_new_tokens`, `temperature`, `missing_eos_penalty`). The trainer lives in
`trl.experimental.online_dpo` — the whole online-preference family is 🧪 experimental.

> The logged metrics blend DPO and PPO: `rewards/accuracies` & `rewards/margins` (the *implicit* DPO
> reward, as in `04`) **plus** `objective/scores` (the *external* RM score) and `objective/rlhf_reward`
> & `objective/kl` (PPO-style, as in `03`). Two reward notions side by side — watch both.

---

## 3. Why on-policy matters: staleness & off-policy drift

Offline DPO (folder `04`) has a structural blind spot the paper names directly:

1. **The pairs are fixed.** They were collected once, before training. The model never gets feedback
   on the *new* behaviors it develops mid-training.
2. **The pairs are off-policy.** They were usually sampled from a *different* model. DPO assumes you're
   nudging relative to your own distribution, but you're really being pulled toward another model's
   outputs — and as your policy drifts during training, the mismatch grows.

Online DPO removes both: every pair is sampled from the policy *as it is right now*, and labeled
*right now*. The feedback tracks the model. Empirically (OAIF) this beats both offline DPO and even
PPO-RLHF on several tasks, with a fraction of PPO's machinery. It is the cleanest demonstration of
*why the online/offline axis is the one that matters* — same loss as `04`, materially better results,
purely from closing the sampling loop.

The cost is exactly what folder `03` warned about: you're now generating during training (slow → §6)
and optimizing a learned proxy on-policy (hackable → §5).

---

## 4. The scorer: reward model vs LLM judge

Online DPO needs to *rank* the two on-policy completions. Two ways, both supported via `reward_funcs`:

- **Reward model (this folder's default).** A scalar RM — *exactly folder `02`'s output*. We default
  `reward_model_path` to `../02-reward-modeling/outputs/qwen3-4b-ultrafeedback-rm` so the course ties
  together: the RM you trained on UltraFeedback prefs in `02` now judges the policy's fresh
  generations in `06`. Cheap per-call, but it's a frozen proxy — it can be hacked and it never
  improves (§5).
- **LLM-as-judge (OAIF's original form).** Prompt a strong LLM to pick the better of the two
  completions (a `PairwiseJudge`). Upside: no RM to train, and the feedback is **steerable by
  instruction** ("prefer concise, factual answers") — OAIF's key finding. Downside: latency/cost per
  call, and the judge has its own biases (verbosity, position, self-preference).

Swapping one for the other is a one-line change to `reward_funcs`. Reward models scale cheaply; judges
are flexible and need no training. Many production loops blend them (RM for throughput, periodic judge
audits to catch RM drift).

> `reward_funcs` also accepts a **custom callable** (same contract as folder `05`'s reward functions)
> or a **list** that gets summed — so you can fold a verifiable check *and* an RM *and* a judge into
> one online signal.

---

## 5. Reward hacking, returned — and how to fight it

This is the heart of the folder. GRPO (`05`) largely dodged reward hacking by using a *verifier*
(a correct answer can't be faked). Online DPO optimizes a **learned reward model** on-policy — which
is precisely the over-optimization setup of folder `02` §6, now in a live loop. The policy will find
and exploit the RM's blind spots **faster** than PPO did, because the feedback is on-policy and tight.

Classic hacks and the defenses TRL gives you:

| Hack | What you see | Defense (knob) |
|---|---|---|
| **Length gaming** | completions balloon; RM (length-biased) scores them up; quality flat | `missing_eos_penalty` (penalize no-EOS); cap `max_new_tokens`; length-penalized/normalized RM |
| **Drift to gibberish the RM loves** | `objective/scores` ↑ while readable quality ↓ | `beta` / KL leash to the reference — the master anti-hack dial (folder `03` §2) |
| **Mode collapse** | outputs homogenize; entropy drops | watch `objective/entropy`; lower LR; raise `beta`; keep `temperature` up |
| **RM blind-spot exploitation** | great RM score, bad on a *held-out* judge | **eval on a judge the policy never trained against** (§8); RM ensembles; early-stop on true metric |

The meta-lesson: **a rising reward-model score is necessary but never sufficient.** The RM is a proxy;
optimizing any proxy hard eventually diverges it from the truth (Goodhart). So the eval harness (§8)
deliberately measures with a scorer separate from the training reward, and flags length blow-ups. In
online RLHF, *your evaluation discipline is your safety system* — not any single training knob.

---

## 6. vLLM & the memory reality of online methods

Online DPO holds **three models**: the trained policy, a frozen reference (for the KL/DPO term), and
the frozen reward model. Add on-policy generation and this is the **tightest memory profile in the
course** — three 4B models won't leave much room on 80 GB once you add optimizer state and a vLLM KV
cache. Practical responses:

- **vLLM for generation** (`use_vllm: true`). Colocate mode shares the training GPU (single-GPU);
  server mode (`trl vllm-serve`) offloads generation to separate GPUs. Tune
  `vllm_gpu_memory_utilization` *down* (we use 0.25) because three models already crowd the device.
- **LoRA** (uncomment `use_peft`): the reference becomes the adapter-disabled policy (no second
  copy), and the policy's trainable footprint shrinks. Strongly recommended here.
- **Smaller RM**: the scorer needn't match the policy size — a 0.5–1B RM judges a 4B policy fine.

This memory squeeze is itself the lesson: online preference methods are powerful but operationally
demanding, which is *why* offline DPO (`04`) remains the default unless on-policy feedback clearly
pays for itself.

---

## 7. The experimental online family (RLOO, NashMD, XPO)

Online DPO is one member of TRL's online-alignment family — all 🧪 experimental. Where they sit:

| Trainer | Idea | When |
|---|---|---|
| **`OnlineDPOTrainer`** 🧪 | on-policy DPO pairs from an RM/judge (this folder) | preference signal + want on-policy freshness |
| **`RLOOTrainer`** ⚡️ | REINFORCE Leave-One-Out: like GRPO, a *group* baseline (leave-one-out mean) instead of a critic — lighter than PPO, no value model | reward-based online RL; a simpler PPO alternative (close cousin of GRPO `05`) |
| **`NashMD`** 🧪⚡️ | game-theoretic: seek the *Nash equilibrium* of a preference game (mirror descent) rather than optimize a point reward | research; preference signals with intransitivity |
| **`XPO`** 🧪⚡️ | Exploratory Preference Optimization: online DPO + an exploration bonus for principled coverage | research; when exploration of the response space matters |

The honesty signal again: the entire online-preference family is experimental in TRL, while **offline
DPO is not**. Translation: for preference alignment, offline DPO is the production default; reach into
this online family when staleness/off-policy drift is demonstrably costing you, and expect rougher
edges. **RLOO** is the most production-ready here — think of it as "GRPO's preference/RM-scored
sibling," and a lighter PPO replacement when you have a reward model rather than a verifier.

---

## 8. The eval harness

[eval.py](eval.py) is the course's tying-together piece — it reuses a reward model (folder `02`) to
*evaluate*, not train. Give it two policies and an RM judge:

```bash
uv run python eval.py \
    --model_a Qwen/Qwen3-4B \
    --model_b outputs/qwen3-4b-online-dpo \
    --reward_model ../02-reward-modeling/outputs/qwen3-4b-ultrafeedback-rm
```

It generates on a held-out prompt set, scores both models, and reports **mean reward**, **win-rate
(B beats A)**, and **mean completion length** — plus a tripwire that warns if B got much longer than A
(a length-gaming red flag, §5).

Two disciplines baked in: (1) **head-to-head win-rate** is the metric that matters, not absolute
scores; (2) ideally the eval RM is *not* the training RM — optimizing against one and grading with the
same one hides exactly the over-optimization you most need to catch. Treat this as a starting skeleton
to extend (add an LLM judge, task-specific checks, a diversity metric).

---

## 9. Honest verdict

**Online DPO is a real, often-superior alternative to offline DPO — but it is not the default, and
the experimental flag is earned.** When on-policy feedback is available and affordable, OAIF-style
training beats offline DAP and even PPO on quality, with far less machinery than PPO. But:

- **Operationally heavy.** Three models + generation + vLLM tuning. Offline DPO (`04`) gets ~most of
  the alignment benefit for a fraction of the trouble — start there; graduate here when staleness
  bites.
- **Reward hacking is live.** You're optimizing a learned proxy on-policy; without KL/EOS discipline
  and honest eval (§5, §8), it degrades faster than PPO. The infrastructure *is* the method.
- **Experimental in TRL.** APIs in this family move; treat them as research-grade.

> **Where the course leaves you:** you now hold the full map — SFT (`01`) → reward models (`02`) →
> PPO (`03`) → DPO (`04`) → GRPO (`05`) → online infra (`06`). You can place any new method on the
> online/offline × preference/reward grid, pick the right tool for a given signal and budget, and run
> it on one GPU. Folder `07` (distillation) is the remaining lever: not a new way to *align*, but a way
> to *transfer* what an aligned/reasoning model knows into a smaller, deployable one.

---

## 10. Run it

### Setup
```bash
cd 06-online-infra
uv sync --extra vllm
uv run hf auth login
uv run wandb login && export WANDB_PROJECT=post-training-lab
```
> As in folder `05`, the `--extra vllm` lock pins **torch 2.11** (vLLM's upper bound), slightly behind
> folders 01–04's 2.12. Self-contained, just a hair behind.

### 1) Smoke test first (no vLLM, no folder 02 needed)
```bash
uv run python train.py --config configs/smoke.yaml
```
Qwen3-0.6B policy + a released tiny RM (`trl-lib/Qwen2-0.5B-Reward`), 5 steps. Confirms the loop:
generate two completions → score → DPO step. Uses plain `generate()`.

### 2) The headline run
First build folder `02`'s reward model (or point `reward_model_path` at any scalar RM), then:
```bash
uv run accelerate launch train.py --config configs/default.yaml
```
Watch **`objective/rlhf_reward`** climb and **`objective/kl`** stay bounded — and read the logged
completions for hacking (length, degeneration). This is PPO's dashboard (folder `03` §6) on a much
simpler trainer.

> ⏱ **Expected runtime (rough):** ~**3–6 h** on a single A100-80GB for 10k prompts, 1 epoch.
> **Estimate, not measured** — generation-bound (assume ±2×); `max_new_tokens` and vLLM dominate.
> H100 ≈ 0.5–0.6×. Memory is the binding constraint (3 models) — use LoRA if you OOM.

### 3) Evaluate (the harness)
```bash
uv run python eval.py --model_a Qwen/Qwen3-4B --model_b outputs/qwen3-4b-online-dpo \
    --reward_model ../02-reward-modeling/outputs/qwen3-4b-ultrafeedback-rm
```

### LoRA / multi-GPU
Uncomment `use_peft` in the config (strongly recommended — 3 models is tight). For scale, `accelerate
launch` with DeepSpeed ZeRO and vLLM **server mode** on dedicated GPUs.

---

## 11. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at load | 3 models + generation don't fit | **LoRA** (`use_peft`), smaller RM, lower `vllm_gpu_memory_utilization`, server-mode vLLM |
| `objective/scores` ↑ but quality ↓ | reward hacking the RM | raise `beta`; `missing_eos_penalty`; eval on a *different* judge (§8); early-stop |
| Completions balloon to max length | length gaming / no EOS | `missing_eos_penalty`, lower `max_new_tokens` |
| `objective/kl` explodes | policy fleeing the reference | raise `beta`; lower LR |
| Two completions identical → no signal | `temperature` too low | raise `temperature` (needs >0 for a meaningful pair) |
| RM loads but tokenizer errors | RM/policy tokenizer mismatch | fine — RM re-tokenizes; ensure the RM path is a seq-classification model |
| Reward flat | LR too low / RM uninformative | raise LR slightly; sanity-check the RM scores with `../02`'s score.py |

---

## 12. Exercises

1. **Online vs offline, head-to-head.** Align the same model on the same prompts with offline DPO
   (`04`) and Online DPO. Use [eval.py](eval.py) to compare win-rate. Quantify what on-policy bought.
2. **Provoke a hack.** Remove `missing_eos_penalty` and lower `beta` to ~0.01. Watch length and
   `objective/scores` climb while `eval.py` flags the length tripwire. Then fix it.
3. **RM vs judge.** Swap the reward model for an LLM `PairwiseJudge`. Compare the aligned behavior and
   discuss steerability (try instructing the judge to prefer concise answers).
4. **Eval integrity.** Run `eval.py` with the *training* RM, then with a *different* RM/judge. Show how
   using the same scorer for training and eval hides over-optimization.
5. **RLOO contrast.** Read TRL's `RLOOTrainer`. Map it to GRPO (`05`) and PPO (`03`): what baseline
   does it use, and what does it drop? When would you pick it over Online DPO?
6. **Cost accounting.** Log GPU memory and step time vs offline DPO (`04`). Put a number on the online
   tax — and decide when it's worth paying.

---

## 13. References

- Online DPO / OAIF (Guo et al. 2024) — https://arxiv.org/abs/2402.04792
- DPO (Rafailov et al. 2023) — https://arxiv.org/abs/2305.18290
- RLOO (Ahmadian et al. 2024) — https://arxiv.org/abs/2402.14740
- Nash-MD / learning from preferences as a game (Munos et al. 2023) — https://arxiv.org/abs/2312.00886
- XPO — Exploratory Preference Optimization (Xie et al. 2024) — https://arxiv.org/abs/2405.21046
- Scaling laws for reward-model over-optimization (Gao et al. 2022) — https://arxiv.org/abs/2210.10760
- The N+ implementation details of RLHF with PPO (TL;DR) — https://arxiv.org/abs/2403.17031
- TRL `OnlineDPOTrainer` docs — https://huggingface.co/docs/trl/online_dpo_trainer
- TRL judges — https://huggingface.co/docs/trl/judges

---

**Next:** [`07-distillation`](../07-distillation) — knowledge distillation (GKD / on-policy). Not a new
way to *align*, but a way to *transfer*: compress a large aligned/reasoning teacher into a small,
deployable student — the lineage behind DeepSeek-R1's distilled models and the "cold start" you
previewed in `01`/`05`.
