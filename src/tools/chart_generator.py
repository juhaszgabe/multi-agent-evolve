import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .base import ToolResult


def chart_generator(
    chart_type: str,
    data: dict,
    title: str,
    x_label: str,
    y_label: str,
    output_dir: str = "outputs",
) -> ToolResult:
    start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            ax.bar(data["x"], data["y"])
        elif chart_type == "line":
            ax.plot(data["x"], data["y"], marker="o")
        elif chart_type == "scatter":
            ax.scatter(data["x"], data["y"])
        elif chart_type == "pie":
            ax.pie(data["values"], labels=data["labels"], autopct="%1.1f%%")
        elif chart_type == "histogram":
            ax.hist(data["values"], bins=data.get("bins", 10))
        else:
            plt.close(fig)
            return ToolResult(success=False, output=None, error=f"Unknown chart type: {chart_type}")

        ax.set_title(title)
        if chart_type not in ("pie",):
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        path = os.path.join(output_dir, f"chart_{int(time.time() * 1000)}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return ToolResult(success=True, output={"image_path": path}, latency_ms=(time.time() - start) * 1000)
    except Exception as e:
        if fig:
            plt.close(fig)
        return ToolResult(success=False, output=None, error=str(e), latency_ms=(time.time() - start) * 1000)
