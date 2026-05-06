from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Valid agent_role values — strings kept (not Enum) for direct LLM interop.
VALID_ROLES = frozenset({"data_analysis", "visualization", "verification", "synthesis", "skip"})


@dataclass
class RouterAction:
    agent_role: str           # one of VALID_ROLES
    tool_name: Optional[str]  # matches Tool.name; None for pure-LLM agents (verification, synthesis)


@dataclass
class TaskState:
    # One-hot over TASK_TYPES (len=7)
    task_type: list = field(default_factory=lambda: [0.0] * 7)
    has_temporal_keyword: bool = False
    csv_size_bucket: int = 0       # 0=small(<1k rows), 1=medium, 2=large(>100k)
    num_columns_bucket: int = 0    # 0=narrow(<5 cols), 1=medium, 2=wide(>20)
    has_numeric_target: bool = False
    # Placeholder for sentence-transformers embedding (384-dim hook)
    task_embedding: list = field(default_factory=lambda: [0.0] * 384)

    def to_float_array(self) -> list[float]:
        """Serialize to flat float array for the bandit feature vector."""
        scalars = [
            float(self.has_temporal_keyword),
            float(self.csv_size_bucket),
            float(self.num_columns_bucket),
            float(self.has_numeric_target),
        ]
        return self.task_type + scalars + self.task_embedding
