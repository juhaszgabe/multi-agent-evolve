import os
import sys
import subprocess
import tempfile
import time
from .base import ToolResult


def python_sandbox(code: str, timeout: int = 30) -> ToolResult:
    start = time.time()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        latency = (time.time() - start) * 1000
        if result.returncode == 0:
            return ToolResult(
                success=True,
                output={"stdout": result.stdout, "stderr": result.stderr, "files_created": []},
                latency_ms=latency,
            )
        return ToolResult(
            success=False,
            output={"stdout": result.stdout, "stderr": result.stderr},
            error=result.stderr.strip() or f"Exit code {result.returncode}",
            latency_ms=latency,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            output=None,
            error=f"Timeout after {timeout}s",
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e), latency_ms=(time.time() - start) * 1000)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
