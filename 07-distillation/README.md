# 07 · Knowledge Distillation — GKD (on-policy) & the R1 cold-start lineage

> **The course's last lever — and a different one.** Folders `01`–`06` made *one* model better at a
> task (align it, reason with it). Distillation is orthogonal: it **transfers** what a big, expensive
> model knows into a **small, deployable** one. It's how a 0.6B model ends up punching far above its
> weight, and it's the machinery behind DeepSeek-R1's distilled series and the "cold-start" SFT you
> previewed in `01`/`05`. We teach **GKD** (Generalized Knowledge Distillation): distillation done
> *on-policy*, the same insight that fixed alignment in folders `03`/`06` applied to compression.

**You will:** distill Qwen3-4B (teacher) into Qwen3-0.6B (student) with `GKDTrainer`, learn the two
dials (`lmbda`, `beta`) that span the whole supervised↔on-policy spectrum, understand *why* naive KD
suffers the same train/inference mismatch online RL solved, contrast on-policy GKD against the offline
"SFT-on-traces" cold-start (the R1 recipe), and place the variants (Sequence-KD, MiniLLM/reverse-KL)
on the map.

> **Online/offline lens — one last time.** The lens that organized the whole course applies here too.
> **Supervised KD** trains the student on a *fixed* set of teacher outputs (offline) → the student is
> graded on sequences it would never generate, then must walk an unpracticed path at inference
> (exposure bias). **GKD** trains the student on *its own* generations scored by the teacher
> (on-policy) → it practices exactly the distribution it will face. Same fix, same reason, third time.

---

## Table of contents
1. [Where this folder sits: transfer, not alignment](#1-where-this-folder-sits-transfer-not-alignment)
2. [Why distill at all? The decision framework](#2-why-distill-at-all-the-decision-framework)
3. [GKD: the mechanism and its two dials](#3-gkd-the-mechanism-and-its-two-dials)
4. [The KD family: supervised, Seq-KD, on-policy, reverse-KL](#4-the-kd-family-supervised-seq-kd-on-policy-reverse-kl)
5. [On-policy GKD vs offline SFT-on-traces (the R1 cold start)](#5-on-policy-gkd-vs-offline-sft-on-traces-the-r1-cold-start)
6. [Teacher & student: vocab, capacity, memory](#6-teacher--student-vocab-capacity-memory)
7. [Honest verdict](#7-honest-verdict)
8. [Run it](#8-run-it)
9. [Common failure modes](#9-common-failure-modes)
10. [Exercises](#10-exercises)
11. [References](#11-references)
12. [Course wrap-up](#12-course-wrap-up)

---

## 1. Where this folder sits: transfer, not alignment

Every previous folder optimized a model *against a signal*: a label (`01`), a preference (`02`,`04`,
`06`), a reward or verifier (`03`,`05`). Distillation optimizes a model *against another model*. The
signal is the **teacher's output distribution** — far richer than a hard label, because a teacher that
says "the answer is Paris" with 92% mass on "Paris" and 5% on "the capital is Paris" teaches the
student *both* the answer and the teacher's uncertainty. These "soft targets" (Hinton et al. 2015) are
the original reason KD beats training the small model from scratch on hard labels.

Two reasons this is the right place to end the course:
- It's the **deployment** lever. Alignment makes a model good; distillation makes a *good* model
  **cheap enough to serve**. Most production LLM stacks ship a distilled student, not the teacher.
- It's the **R1 lineage**. DeepSeek-R1 produced reasoning traces from a large RL-trained model, then
  **distilled** them into small dense models that inherited much of the reasoning — and the same
  "generate traces from a strong model, SFT a student on them" move is the **cold start** you saw
  previewed in `01` (SFT foundation) and `05` (GRPO needs a reasonable starting policy). This folder
  names that move and shows its *on-policy* upgrade.

---

## 2. Why distill at all? The decision framework

Distill when you need a model **smaller than the one that has the capability you want**. Concretely:

| Situation | Distill? | Why |
|---|---|---|
| You have a strong large model; need low latency / cheap serving | **Yes** | Compress capability into a deployable size — the core use case |
| You need reasoning in a small model, but small models won't RL from scratch | **Yes** | Distill a big reasoning teacher's traces → student cold-start (then optionally RL, `05`) |
| You have lots of unlabeled prompts but few labels | **Yes** | The teacher labels them (soft targets) — KD as cheap supervision |
| You want the student to *exceed* the teacher | **No** | KD's ceiling is roughly the teacher; to surpass it you need a stronger signal (RL with a verifier, `05`) |
| The teacher and student have different tokenizers/vocab | **Careful** | Token-level KD (incl. GKD) needs a shared vocab; cross-tokenizer KD is its own research problem (§6) |
| You can just deploy the big model affordably | **No** | Distillation always loses *something*; skip the step if you don't need the compression |

The honest framing: distillation **transfers** capability downward, it rarely **creates** new
capability. Its job is efficiency. If your problem is "my model isn't good enough," distillation is the
wrong folder — go back to `01`–`06`. If it's "my *good* model is too big to serve," this is the tool.

---

## 3. GKD: the mechanism and its two dials

Start with **naive supervised KD**: collect a dataset of (prompt, teacher-output), then minimize the
KL between the student and teacher token distributions on those fixed sequences. This is just SFT with
soft targets — and it carries SFT's offline flaw. The targets are sequences the *teacher* produced.
The student is never graded on what *it* would produce, so at inference, the moment it drifts off the
teacher's trajectory it's in unpracticed territory and errors compound (**exposure bias** — the exact
train/inference mismatch GKD's paper is titled after).

**GKD** (Agarwal et al. 2023) closes the loop, identically to how Online DPO (`06`) closed DPO's:

```
each training step, for a batch:
  with probability `lmbda`:                         ← the on-policy fraction
      the STUDENT generates a completion             (y ~ π_student, on-policy)
  else:
      use the dataset's (teacher/reference) target   (off-policy)
  the TEACHER scores that sequence token-by-token   (full distribution per position)
  minimise a generalized Jensen-Shannon divergence between student and teacher,
      interpolated by `beta`, on every token
```

The student learns to mimic the teacher **on the student's own distribution** — it practices the path
it will actually walk. Two dials span the entire method:

- **`lmbda` — the on-policy fraction** (0→1). `0.0` = pure supervised KD on the dataset's targets;
  `1.0` = every sequence is student-generated (fully on-policy); in between, mixed per batch. **The
  paper's headline result: higher `lmbda` (more on-policy) consistently wins** — same lesson as the
  rest of the course. The cost is also the same: on-policy means generation in the loop (slower).

- **`beta` — the divergence shape** (0→1, via the generalized JSD). `0.0` ≈ **forward KL**
  (mode-*covering*: the student tries to put mass everywhere the teacher does — "cover all the
  teacher's modes"). `1.0` ≈ **reverse KL** (mode-*seeking*: the student commits to the teacher's
  dominant modes and ignores the tails). **Reverse KL is often better for a small student** that
  *can't* represent the teacher's full distribution — better to nail the main modes than to smear thin
  mass trying to cover everything it lacks the capacity for. The optimal `beta` is task-dependent;
  it's the second thing to sweep after `lmbda`.

> A third flag, **`seq_kd`**, switches on Sequence-Level KD: the *teacher* generates whole sequences
> and the student is SFT'd on them. `seq_kd=True, lmbda=0.0` is "SFT on teacher generations" — the
> offline cold-start recipe (§5) — expressible right inside the same trainer.

`GKDTrainer` is a thin wrapper around `SFTTrainer` (folder `01`) plus a teacher and this loss, so every
SFT knob (`max_length`, packing, LR schedule) carries over. It lives in `trl.experimental.gkd` (🧪).

---

## 4. The KD family: supervised, Seq-KD, on-policy, reverse-KL

All four are reachable from the same two dials (plus `seq_kd`) — which is the elegant part:

| Variant | Setting | What the student learns from | When |
|---|---|---|---|
| **Supervised / word-level KD** | `lmbda=0`, `seq_kd=False` | teacher's per-token distribution on a *fixed* dataset | cheapest; baseline; targets already exist |
| **Sequence-level KD (Seq-KD)** | `lmbda=0`, `seq_kd=True` | SFT on the *teacher's own* generations | the classic "generate traces, SFT student" cold-start (§5) |
| **On-policy GKD** | `lmbda>0` (try `1.0`) | teacher feedback on the *student's* generations | best quality; fixes exposure bias; costs generation |
| **Reverse-KL emphasis** | raise `beta`→`1.0` | teacher's dominant modes (mode-seeking) | small/low-capacity student that can't cover the full distribution |
| **MiniLLM** 🧪 | (separate trainer) | reverse-KL distillation with policy-gradient tricks | research; reverse-KL taken to its conclusion |

The two production-relevant points: (1) **on-policy (`lmbda>0`) is the quality win** and the reason GKD
exists; (2) **reverse-KL (`beta`↑) matters more the smaller your student is.** MiniLLM (Gu et al. 2023)
is the dedicated reverse-KL method — worth knowing as "what reverse-KL looks like pushed all the way,"
but GKD's `beta` already gives you the dial. Both are 🧪 in TRL: distillation trainers are research-grade
in the library even though the *technique* is thoroughly production-proven (every major lab ships
distilled models).

---

## 5. On-policy GKD vs offline SFT-on-traces (the R1 cold start)

This is the contrast the course has been building toward. There are **two** ways to distill a teacher,
and they are the offline/online split one final time:

**Offline: SFT on teacher traces (the "cold start").** Generate a big batch of (prompt → answer)
sequences from the teacher *once*, then SFT the student on them as a fixed dataset. This is exactly
what folder `01` does, with the teacher as the data source — and it's literally the **DeepSeek-R1
recipe**: take a strong reasoning model, harvest its chains-of-thought, SFT smaller dense models on
those traces to give them a reasoning "cold start" (after which you *can* RL them, `05`). It's simple,
embarrassingly parallel (generate once, train many), and needs the teacher only at data-gen time — no
teacher in GPU memory during training. In GKD terms it's `seq_kd=True, lmbda=0`.

**On-policy: GKD.** Keep the teacher live and score the *student's own* generations each step
(`lmbda>0`). The student gets feedback on its actual trajectory, fixing the exposure bias that fixed
traces leave in. Higher quality per token of data — at the cost of holding the teacher in memory and
generating during training.

| | Offline SFT-on-traces (R1 cold start) | On-policy GKD |
|---|---|---|
| Teacher needed during training | No (only to pre-generate) | **Yes** (scores every step) |
| Distribution the student trains on | teacher's (fixed, off-policy) | **student's own** (on-policy) |
| Exposure bias | present | **addressed** |
| Cost | cheap; generate once, reuse | generation in the loop (slower) |
| Scales to huge teachers | easily (gen offline, even via API) | bounded by fitting teacher+student in memory |
| TRL | folder `01` SFT (or `seq_kd=True, lmbda=0` here) | `GKDTrainer`, `lmbda>0` (this folder) |

**The practitioner's synthesis:** the offline cold-start is the **workhorse** — it's how most distilled
models (R1-Distill included) are actually made, because it scales to teachers too big to co-locate and
even to API-only teachers. **GKD is the quality upgrade** when you *can* hold the teacher alongside the
student and want to squeeze out the exposure-bias gap. A common pipeline uses both: offline
SFT-on-traces for the cold start, then a short on-policy GKD (or RL, `05`) pass to refine. You now have
every piece of that pipeline in this course.

---

## 6. Teacher & student: vocab, capacity, memory

**Shared vocabulary is mandatory.** Token-level KD (supervised KD *and* GKD) aligns the teacher's and
student's distributions **position-by-position over the same vocabulary**. Distill across model
families (different tokenizers) and the distributions aren't comparable token-for-token. The clean,
supported path is a **teacher and student from the same family** — here Qwen3-4B → Qwen3-0.6B, same
tokenizer. (Cross-tokenizer KD exists — ULD, MinED, logit alignment — but it's a research frontier, not
a `GKDTrainer` config flag.)

**Capacity sets the ceiling and steers `beta`.** A 0.6B student cannot represent everything a 4B
teacher knows; distillation's job is to transfer as much as the student *can* hold. The bigger the
teacher↔student gap, the more **reverse-KL** (`beta`↑, mode-seeking) tends to help — don't ask a small
student to smear probability over modes it can't afford; have it commit to the teacher's best ones.

**Memory: the gentlest profile in the course's online half.** On-policy GKD holds **two** models — the
trained student and the frozen teacher — plus generation. But the student is *small* (that's the whole
point), so the binding cost is the teacher. Qwen3-4B teacher + Qwen3-0.6B student + on-policy
generation sits comfortably on one 80 GB GPU — far easier than `06`'s three-4B-model squeeze. If you
distill from a much larger teacher (e.g. 70B), either (a) go offline (§5 — generate traces once, drop
the teacher) or (b) serve the teacher on separate GPUs. LoRA on the student is available but rarely
needed; full-FT of a sub-1B student is cheap.

---

## 7. Honest verdict

**Distillation is production-essential and GKD is the right *technique*, but read the two flags
honestly.** Distillation is not optional industry knowledge — it's how capable models become
*deployable* models, and the R1-Distill series proved it transfers even hard-won reasoning. Within it:

- **On-policy GKD > supervised KD**, for the same reason on-policy beats offline everywhere else in
  this course (it fixes exposure bias). When you can hold the teacher in memory, prefer `lmbda>0`.
- **But the offline cold-start (SFT-on-traces) is the workhorse**, because it scales to teachers too
  big to co-locate or that you only have via an API — which describes most frontier teachers. GKD is
  the refinement, not the default first move for very large teachers.
- **Experimental in TRL** (`trl.experimental.gkd`, and MiniLLM 🧪). The *technique* is battle-tested;
  the TRL *trainer* is research-grade and its API can move. Treat accordingly.
- **It transfers, it doesn't create.** The student's ceiling is ~the teacher. To exceed the teacher you
  need a genuinely stronger signal — a verifier and RL (`05`), not more distillation.

> **Bottom line:** to ship a small model, distill. Use the **offline cold-start** to scale and to
> bootstrap reasoning (the R1 move); reach for **on-policy GKD** to close the exposure-bias gap when
> the teacher fits alongside the student. Then, if you need to *surpass* the teacher, hand off to RL.

---

## 8. Run it

### Setup
```bash
cd 07-distillation
uv sync
uv run hf auth login
uv run wandb login && export WANDB_PROJECT=post-training-lab
```
> No vLLM extra here: GKD's on-policy generation runs through the trainer's own loop, and the memory
> profile (small student + one teacher) is easy. Lock pins torch 2.12 like folders 01–04 (no vLLM
> upper bound to dodge). `--extra flash` / `--extra quant` are optional as elsewhere.

### 1) Smoke test first
```bash
uv run python train.py --config configs/smoke.yaml
```
Qwen3-1.7B teacher → Qwen3-0.6B student (same family → shared vocab), 5 steps. Confirms the loop:
student generates → teacher scores tokens → JSD step. Plain `generate()`, `report_to: none`.

### 2) The headline run
```bash
uv run accelerate launch train.py --config configs/default.yaml
```
Distills Qwen3-4B → Qwen3-0.6B on 50k Tulu-3 messages, `lmbda=0.5`, `beta=0.5`. Watch the JSD/loss
fall; the metric that matters is downstream quality, so follow with `generate.py`.

> ⏱ **Expected runtime (rough):** ~**4–8 h** on a single A100-80GB for 50k samples, 1 epoch, at
> `lmbda=0.5`. **Estimate, not measured** — on-policy generation dominates, so it scales with `lmbda`
> and `max_new_tokens` (assume ±2×). `lmbda=0` (supervised KD) is much faster; `lmbda=1.0` slower.
> H100 ≈ 0.5–0.6×.

### 3) Compare student vs teacher (did it transfer?)
```bash
uv run python generate.py --student outputs/qwen3-0.6b-gkd --teacher Qwen/Qwen3-4B --base Qwen/Qwen3-0.6B
```
Reads teacher / distilled-student / undistilled-student side by side — quality at 0.6B *inference cost*
is the whole deliverable.

### Tuning the dials
Sweep **`lmbda`** first (0 → 0.5 → 1.0): higher is usually better, slower. Then **`beta`** (0.5 → 1.0):
push toward reverse-KL the larger the teacher↔student gap. For the offline cold-start instead, set
`seq_kd: true, lmbda: 0.0` (SFT on the teacher's own generations) — no on-policy generation.

### LoRA / multi-GPU
Uncomment `use_peft` to LoRA the student (rarely needed at <1B). `accelerate launch` scales cleanly;
for a *large* teacher, prefer the offline route (§5) or serve the teacher on separate GPUs.

---

## 9. Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Crash: vocab/shape mismatch in the loss | teacher & student have different tokenizers | use a **same-family** teacher/student (shared vocab) — §6 |
| OOM at load | teacher too big to co-locate with student + generation | smaller teacher, **LoRA student**, or go **offline** (§5 — pre-generate traces, drop teacher) |
| Training is very slow | high `lmbda` → lots of on-policy generation | lower `lmbda`, cut `max_new_tokens`; or use offline Seq-KD for the bulk then a short on-policy pass |
| Student fluent but bland / low-diversity | `beta` too high (over mode-seeking) for this gap | lower `beta` toward forward-KL (mode-covering) |
| Student covers junk modes / incoherent | `beta` too low for a small student | raise `beta` toward reverse-KL (mode-seeking) |
| Student ≈ no better than base | `lmbda=0` exposure bias, or teacher too close in size | raise `lmbda`; pick a stronger teacher; verify teacher actually outperforms on your prompts |
| NaNs with a Gemma teacher/student | Gemma2 soft-capping needs the right attn kernel | set `attn_implementation` to a flash-attn2 kernel (TRL's Gemma note) |
| Student never emits EOS / rambles | length/stop behavior not transferred | ensure chat template + EOS are correct; check `max_new_tokens`; inspect with `generate.py` |

---

## 10. Exercises

1. **Distill vs SFT, head-to-head.** Train the *same* Qwen3-0.6B two ways on the *same* Tulu-3 data:
   plain SFT (folder `01`) and GKD against the 4B teacher. Compare with `generate.py`. Quantify what
   the teacher's soft targets bought over hard labels.
2. **Sweep `lmbda`.** Run `lmbda ∈ {0.0, 0.5, 1.0}` (supervised → fully on-policy). Plot quality vs
   wall-clock. Confirm (or challenge) the paper's "more on-policy is better" — and price the cost.
3. **Sweep `beta`.** With `lmbda` fixed, run `beta ∈ {0.0, 0.5, 1.0}` (forward → reverse KL). Where on
   the mode-covering↔mode-seeking axis does *this* teacher/student gap land best? Relate to capacity.
4. **Offline cold-start (the R1 move).** Set `seq_kd: true, lmbda: 0.0` to SFT the student on the
   teacher's generations. Compare against on-policy GKD on quality *and* training time. When is the
   offline workhorse good enough?
5. **Cold-start → RL.** Take your distilled student and run it as the *starting policy* for GRPO on
   GSM8K (folder `05`). Does the distilled cold-start RL better than the base 0.6B? (This is the R1
   pipeline end-to-end, on one GPU.)
6. **Capacity wall.** Distill the same teacher into two students (e.g. 0.6B and 1.7B). Measure how much
   of the teacher each recovers. Where does the capacity ceiling (§2, §6) bite?

---

## 11. References

- GKD — On-Policy Distillation of LMs (Agarwal et al. 2023) — https://arxiv.org/abs/2306.13649
- Distilling the Knowledge in a Neural Network (Hinton et al. 2015) — https://arxiv.org/abs/1503.02531
- Sequence-Level Knowledge Distillation (Kim & Rush 2016) — https://arxiv.org/abs/1606.07947
- MiniLLM — Knowledge Distillation of LLMs / reverse-KL (Gu et al. 2023) — https://arxiv.org/abs/2306.08543
- DeepSeek-R1 (distillation into small dense models; the cold start) — https://arxiv.org/abs/2501.12948
- TRL `GKDTrainer` docs — https://huggingface.co/docs/trl/gkd_trainer
- TRL `SFTTrainer` (GKD's base) — https://huggingface.co/docs/trl/sft_trainer

---

## 12. Course wrap-up

You've now built the full post-training toolkit, one runnable folder at a time:

| Folder | Technique | The one-line "why it exists" |
|---|---|---|
| `01` | **SFT** | teach the format/behavior — the foundation everything else builds on |
| `02` | **Reward modeling** | turn human preferences into a scalar signal RL can optimize |
| `03` | **PPO** | the original RLHF loop — taught in full, then honestly retired |
| `04` | **DPO** | PPO's objective in closed form — the offline preference default |
| `05` | **GRPO / RLVR** | online RL with *verifiable* rewards — the reasoning era |
| `06` | **Online infra** | Online DPO, vLLM, reward-hacking defenses, eval — making online RL shippable |
| `07` | **Distillation** | transfer capability into a small, deployable model — the R1 lineage |

> **For the production picture** — what a real 2026 lab actually runs end-to-end, and what's faded
> (PPO, standalone chat RMs, value models) — see [**"What modern labs actually run in
> production"**](../README.md#what-modern-labs-actually-run-in-production-2026) on the course landing page.

The two ideas that recur and that you should carry out of this course:

1. **The online/offline lens.** Whether you train on a fixed dataset or on fresh samples from the
   current model is *the* axis that explains DPO vs Online DPO, SFT-traces vs GKD, and why GRPO needs
   vLLM. On-policy wins on quality; it costs generation. Every folder is a point on that axis.
2. **Optimizing a proxy is always provisional.** Reward models (`02`), DPO's implicit reward (`04`),
   GKD's teacher (`07`) — each is a stand-in for what you actually want. Optimize any proxy hard enough
   and it diverges from the truth (Goodhart). Your *evaluation discipline* (`06` §8), not any single
   training knob, is what keeps you honest.

Go ship a small, aligned, reasoning model — on one GPU.

— *End of the Post-Training Lab.*
