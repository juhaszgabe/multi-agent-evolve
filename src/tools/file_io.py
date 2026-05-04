import os
import time
from .base import ToolResult


def file_io(action: str, path: str, content: str = None) -> ToolResult:
    start = time.time()
    try:
        if action == "write":
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(success=True, output={"path": path}, latency_ms=(time.time() - start) * 1000)
        elif action == "read":
            with open(path, "r", encoding="utf-8") as f:
                return ToolResult(success=True, output={"content": f.read()}, latency_ms=(time.time() - start) * 1000)
        else:
            return ToolResult(success=False, output=None, error=f"Unknown action: {action}")
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e), latency_ms=(time.time() - start) * 1000)
