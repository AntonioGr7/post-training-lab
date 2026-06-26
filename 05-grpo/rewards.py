"""Verifiable reward functions for GRPO on GSM8K — the heart of RLVR.

RLVR (RL with Verifiable Rewards) replaces a *learned* reward model (folders 02/03) with a
**programmatic verifier**: for math, you can simply *check whether the final answer is correct*.
No reward model to train, no reward to over-optimize against (folder 02 §6) — the reward is ground
truth. This is why GRPO took off for reasoning: the signal is cheap, dense enough, and un-hackable
in the way a learned RM is.

A TRL reward function takes `completions` (+ any dataset columns as kwargs, here `ground_truth`) and
returns a `list[float]`, one reward per completion. The trainer sums (optionally weights) multiple
reward functions. We define two — the classic DeepSeek-R1 pair:

  • correctness_reward  — the verifiable signal: does the boxed answer equal the gold answer?
  • format_reward       — a small shaping reward for emitting a parseable `\\boxed{...}` answer.

`trl.rewards` also ships a ready-made `accuracy_reward` (math-verify based). We hand-roll ours so the
mechanics are visible and the dependency footprint stays small.
"""

import re

# A system prompt that asks for step-by-step reasoning and a parseable final answer. Putting the
# answer in \boxed{} gives the verifier an unambiguous target to extract.
SYSTEM_PROMPT = (
    "You are a careful mathematician. Reason step by step, then give your final numeric answer "
    "on the last line inside \\boxed{}, e.g. \\boxed{42}."
)

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _text(completion) -> str:
    """Completions are plain strings (standard format) or message lists (conversational)."""
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]  # conversational: last assistant turn


def _normalize(s: str) -> str:
    return s.strip().replace(",", "").replace("$", "").rstrip(".")


def extract_answer(text: str) -> str | None:
    """Pull the model's final answer: prefer \\boxed{...}, else fall back to the last number."""
    boxed = _BOXED.findall(text)
    if boxed:
        return _normalize(boxed[-1])
    nums = _NUM.findall(text)
    return _normalize(nums[-1]) if nums else None


def correctness_reward(completions, ground_truth, **kwargs) -> list[float]:
    """1.0 if the extracted final answer matches the gold answer, else 0.0. The verifiable signal."""
    rewards = []
    for completion, gold in zip(completions, ground_truth):
        pred = extract_answer(_text(completion))
        rewards.append(1.0 if pred is not None and pred == _normalize(gold) else 0.0)
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """0.2 if the completion contains a well-formed \\boxed{...}. Shapes toward parseable outputs.

    Keep format rewards SMALL relative to correctness — otherwise the policy learns to emit the
    format without solving the problem (a mild form of reward hacking; see README §7)."""
    return [0.2 if _BOXED.search(_text(c)) else 0.0 for c in completions]


def gsm8k_gold(answer: str) -> str:
    """GSM8K answers look like a worked solution ending in '#### 42'. Extract the gold number."""
    return _normalize(answer.split("####")[-1])
