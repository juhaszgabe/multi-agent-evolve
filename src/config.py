from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


def _default_model_per_role() -> dict:
    sonnet = os.getenv("MODEL_SONNET", "meta/llama-3.3-70b-instruct")
    haiku = os.getenv("MODEL_HAIKU", "meta/llama-3.1-8b-instruct")
    return {
        "planner": sonnet,
        "data_analysis": sonnet,
        "visualization": haiku,
        "verification": sonnet,
        "synthesis": sonnet,
    }


def _default_action_space() -> list:
    return [
        {"agent_role": "data_analysis",  "tool_name": "python_sandbox"},
        {"agent_role": "data_analysis",  "tool_name": "sql_query"},
        {"agent_role": "data_analysis",  "tool_name": "python_sandbox+statistical_test"},
        {"agent_role": "visualization",  "tool_name": "python_sandbox"},
        {"agent_role": "visualization",  "tool_name": "chart_generator"},
        {"agent_role": "verification",   "tool_name": None},
        {"agent_role": "synthesis",      "tool_name": None},
    ]


@dataclass
class Config:
    # Router selection: "planner" | "random" | "static" | "bandit"
    router_type: str = "planner"

    # Model to use per agent role
    model_per_role: dict = field(default_factory=_default_model_per_role)

    # Reward function weights
    lambda_cost: float = 0.1
    lambda_time: float = 0.05

    # Retry limits
    max_retries_analyst: int = 3
    max_retries_critic: int = 2

    # Valid (agent_role, tool_name) combinations for the router
    action_space: list = field(default_factory=_default_action_space)

    # Single temperature for all LLM calls (set to 0.0 for eval runs)
    temperature: float = 0.1

    # Reproducibility
    random_seed: int = 42

    # BanditRouter
    bandit_alpha: float = 0.25
    bandit_state_path: Optional[str] = None  # .npz file; None = in-memory only

    # Skill/error memory (requires pip install .[memory])
    enable_memory: bool = False
    skill_write_threshold: float = 0.8


def load_config(path: str | None = None) -> Config:
    if path is None:
        return Config()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})


def config_to_dict(config: Config) -> dict:
    """Serialize Config to a plain dict (for LangGraph state, which requires serializable values)."""
    import dataclasses
    return dataclasses.asdict(config)


def config_from_dict(d: dict) -> Config:
    return Config(**{k: v for k, v in d.items() if k in Config.__dataclass_fields__})
