from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
