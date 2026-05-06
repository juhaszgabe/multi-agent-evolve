from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    cost_estimate: float = 0.0


@dataclass
class Tool:
    """Wraps a typed tool function with metadata needed by the Router and logging."""
    name: str
    description: str
    fn: Callable        # underlying typed function; call as tool.fn(...)
    input_schema: dict  # JSON Schema dict describing fn's arguments
