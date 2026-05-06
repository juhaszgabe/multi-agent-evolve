from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from router.actions import RouterAction


@dataclass
class StepRecord:
    step_id: str
    workflow_id: str
    timestamp: float
    state_features: dict
    planner_suggestion: RouterAction
    chosen_action: RouterAction
    agent_name: str
    tool_name: Optional[str]
    tool_input: dict
    tool_output: dict
    success: bool
    error: Optional[str]
    latency_ms: int             # rounded at log time; float precision not needed for RL
    input_tokens: int
    output_tokens: int
    cost_estimate_usd: float
    local_reward_components: dict = field(default_factory=dict)  # populated by RewardFunction


@dataclass
class WorkflowOutcome:
    workflow_id: str
    question: str
    success: bool
    critic_approved: bool
    total_latency_ms: float
    total_cost_usd: float
    step_records: list = field(default_factory=list)  # list of StepRecord


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _to_serializable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


def save_step_record(record: StepRecord, log_dir: str = "logs") -> None:
    path = os.path.join(log_dir, "step_records", f"{record.workflow_id}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(record), default=_to_serializable) + "\n")


def save_workflow_summary(outcome: WorkflowOutcome, log_dir: str = "logs") -> None:
    path = os.path.join(log_dir, "workflows", f"{outcome.workflow_id}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary = {
        "workflow_id": outcome.workflow_id,
        "question": outcome.question,
        "success": outcome.success,
        "critic_approved": outcome.critic_approved,
        "total_latency_ms": outcome.total_latency_ms,
        "total_cost_usd": outcome.total_cost_usd,
        "step_count": len(outcome.step_records),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_to_serializable)
