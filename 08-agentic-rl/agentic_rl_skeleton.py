"""Annotated PSEUDO-CODE of the multi-turn GRPO loop for agentic RL — READ IT, DON'T RUN IT.

This folder (08) is theory + frontier map: real multi-turn agentic RL needs an environment, a sandbox
cluster, and a distributed framework (verl / SkyRL / rLLM), NOT a TRL script. See README §8. The point
of this file is to make the mechanics of README §4-§5 concrete and inspectable in ~120 lines:

    • the ROLLOUT is a whole episode (scaffold loop), not one model.generate()
    • LOSS MASKING: gradient only on the policy's OWN tokens; environment tokens are masked (§4)
    • GROUP-RELATIVE advantage: GRPO's critic-free baseline, now over K *episodes* (§4)
    • the REWARD is the verifier's TERMINAL outcome (tests pass), optionally shaped (§7)
    • CREDIT ASSIGNMENT: vanilla = broadcast the one terminal advantage to all tokens (§5) — and the
      hook where turn-level / GiGPO-style methods would replace that broadcast.

The functions `policy`, `env`, `verifier` are interfaces you would back with a real model server
(vLLM/SGLang), a real sandbox (Docker/Kubernetes), and a real test harness. Calls below are illustrative.

This file intentionally does NOT import torch/trl and is NOT a working trainer. py_compile-clean only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ----------------------------------------------------------------------------------------------------
# Data structures: a trajectory is an interleaving of POLICY spans and ENVIRONMENT spans.
# ----------------------------------------------------------------------------------------------------


@dataclass
class Span:
    """A contiguous run of tokens with a source. `is_policy` decides whether it gets a loss."""
    token_ids: list[int]
    is_policy: bool          # True = the model wrote it (train on it); False = env output (MASK it)
    turn: int                # which agent turn this span belongs to (for turn-level credit, §5)


@dataclass
class Trajectory:
    spans: list[Span] = field(default_factory=list)
    reward: float = 0.0      # the TERMINAL outcome reward (filled in after the verifier runs)

    @property
    def loss_mask(self) -> list[bool]:
        """README §4, detail #2: True where a token gets a gradient (policy tokens only).

        Training on environment tokens teaches the model to predict tool output it does not control —
        noise that destabilizes everything. Every agentic framework masks them out.
        """
        mask: list[bool] = []
        for span in self.spans:
            mask.extend([span.is_policy] * len(span.token_ids))
        return mask


# ----------------------------------------------------------------------------------------------------
# (1) ROLLOUT: run the scaffold loop in the environment until the agent submits or hits the budget.
#     This REPLACES single-turn model.generate(prompt). (README §2, §4 change #1)
# ----------------------------------------------------------------------------------------------------


def rollout_episode(policy, env, instruction: str, max_turns: int = 100) -> Trajectory:
    traj = Trajectory()
    obs = env.reset(instruction)                       # initial observation: the task + repo state
    for turn in range(max_turns):
        # The model emits a thought + a tool call (policy tokens — these will be trained on).
        action_tokens, tool_call = policy.act(obs)
        traj.spans.append(Span(action_tokens, is_policy=True, turn=turn))

        if tool_call.name == "submit":
            break

        # The environment executes the tool and returns output (NON-policy tokens — masked).
        obs, obs_tokens = env.step(tool_call)
        traj.spans.append(Span(obs_tokens, is_policy=False, turn=turn))

    return traj


# ----------------------------------------------------------------------------------------------------
# (2) REWARD: the verifier's terminal outcome, optionally shaped. (README §4 change #3, §7)
# ----------------------------------------------------------------------------------------------------


def score_episode(verifier, env, traj: Trajectory, max_turns: int = 100) -> float:
    # Backbone: verifiable outcome reward — 1.0 if the hidden test suite passes, else 0.0 (DeepSWE).
    passed = verifier.run_hidden_tests(env)
    reward = 1.0 if passed else 0.0

    # Shaping (§7): penalize malformed tool calls and unfinished/over-long trajectories.
    n_turns = 1 + max(s.turn for s in traj.spans)
    if n_turns >= max_turns:
        reward -= 0.1                                  # unfinished-trajectory penalty
    # (a real recipe also zero-rewards invalid-tool-call FORMAT and may add per-turn token penalties)
    return reward


# ----------------------------------------------------------------------------------------------------
# (3) GROUP-RELATIVE ADVANTAGE: GRPO's critic-free baseline (folder 05), now over K EPISODES. (§4)
# ----------------------------------------------------------------------------------------------------


def group_advantages(rewards: list[float]) -> list[float]:
    """Subtract the group mean — no value model. Same trick as folder 05, episode-level here."""
    mean = sum(rewards) / len(rewards)
    # (DeepSWE's "GRPO++" drops the std-normalization that vanilla GRPO does here — README §6/Dr.GRPO)
    return [r - mean for r in rewards]


def assign_token_credit(traj: Trajectory, episode_advantage: float) -> list[float]:
    """README §5 — THE hard problem, and the hook where the frontier diverges.

    VANILLA (trajectory-level): broadcast the single episode advantage to EVERY policy token. Simple,
    high-variance, and 'wrong' over long horizons (the brilliant fix and 20 useless `ls`es get equal
    credit). This is what GRPO/DeepSWE do by default.

    FRONTIER (turn-level / GiGPO / hindsight): replace the uniform broadcast with a per-turn or
    per-step advantage. That is the single most active research area in agentic RL. The swap happens
    RIGHT HERE — same loop, different credit function.
    """
    per_token: list[float] = []
    for span in traj.spans:
        adv = episode_advantage if span.is_policy else 0.0   # masked tokens carry no credit
        per_token.extend([adv] * len(span.token_ids))
    return per_token
    # turn-level alternative (sketch):
    #   per_turn_adv = turn_level_advantages(traj)           # e.g. MT-GRPO (2505.11821) / GiGPO (2505.10978)
    #   ... broadcast per_turn_adv[span.turn] instead of the single episode_advantage ...


# ----------------------------------------------------------------------------------------------------
# (4) THE OUTER LOOP: for each task, sample K episodes, score, advantage, masked policy-gradient step.
# ----------------------------------------------------------------------------------------------------


def train_step(policy, env, verifier, instruction: str, group_size: int = 8) -> None:
    # Sample a GROUP of K episodes for the SAME task (GRPO needs the group for its baseline).
    # In a real system these K rollouts run ASYNC across a sandbox cluster — the infra problem (§8).
    trajs = [rollout_episode(policy, env, instruction) for _ in range(group_size)]
    for traj in trajs:
        traj.reward = score_episode(verifier, env, traj)

    advantages = group_advantages([t.reward for t in trajs])

    for traj, adv in zip(trajs, advantages):
        token_advantages = assign_token_credit(traj, adv)
        mask = traj.loss_mask
        # A real trainer now computes the clipped/sequence-level policy-gradient loss (GSPO/DPPO, §6)
        # over the policy tokens only, weighting each by its token advantage, and backprops it:
        #
        #   loss = policy_gradient_loss(logits, token_ids, token_advantages, mask, trust_region=GSPO)
        #   loss.backward(); optimizer.step()
        #
        # masked-out (environment) tokens contribute ZERO to the loss by construction.
        _ = (token_advantages, mask)   # illustrative no-op; see verl/SkyRL for the real implementation


if __name__ == "__main__":
    print(__doc__)
    print("This is annotated pseudo-code (README §12). It does not run. Read verl / SkyRL for the real thing.")
