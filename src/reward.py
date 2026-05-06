from __future__ import annotations

from logging_schema import StepRecord, WorkflowOutcome

# Normalization reference values — clamp each component to [0, 1]
MAX_COST_USD: float = 0.10     # $0.10 per step → normalized cost = 1.0
MAX_LATENCY_MS: float = 30_000  # 30 seconds → normalized time = 1.0


class RewardFunction:
    """
    Scalar reward for a single step within a completed workflow.

    r = success_score − λ_cost · cost_norm − λ_time · time_norm

    All components are normalized to [0, 1] before applying λ weights,
    giving a reward range of approximately [−0.15, 1.0] for typical runs.

    The breakdown is stored in step.local_reward_components so analyses
    can later determine which component dominated the signal.
    """

    def __init__(self, lambda_cost: float = 0.1, lambda_time: float = 0.05):
        self.lambda_cost = lambda_cost
        self.lambda_time = lambda_time

    def compute_step_reward(self, step: StepRecord, outcome: WorkflowOutcome) -> float:
        success_score = 1.0 if outcome.critic_approved and step.success else 0.0
        cost_norm = min(step.cost_estimate_usd / MAX_COST_USD, 1.0)
        time_norm = min(step.latency_ms / MAX_LATENCY_MS, 1.0)

        r = success_score - self.lambda_cost * cost_norm - self.lambda_time * time_norm

        step.local_reward_components = {
            "success": success_score,
            "cost": cost_norm,
            "time": time_norm,
            "reward": r,
        }
        return r

    def compute_workflow_rewards(
        self, outcome: WorkflowOutcome
    ) -> list[float]:
        """Compute and store rewards for all steps in a completed workflow."""
        return [self.compute_step_reward(rec, outcome) for rec in outcome.step_records]
