from __future__ import annotations

from router.actions import TaskState

TASK_TYPES = ["aggregation", "filter", "compare", "stat", "viz", "text", "verification"]

TEMPORAL_KEYWORDS = {
    "month", "year", "trend", "daily", "weekly", "quarterly",
    "over time", "date", "annual", "hourly", "seasonal",
}

# Keywords that hint at each task type (used for one-hot classification)
_TASK_KEYWORDS: dict[str, list[str]] = {
    "aggregation":  ["sum", "total", "average", "mean", "count", "max", "min", "aggregate"],
    "filter":       ["filter", "where", "only", "exclude", "subset", "condition"],
    "compare":      ["compare", "difference", "vs", "versus", "between", "growth", "change"],
    "stat":         ["correlation", "significant", "test", "p-value", "anova", "t-test", "regression"],
    "viz":          ["chart", "plot", "graph", "visualize", "show", "display", "trend line"],
    "text":         ["describe", "summarize", "explain", "narrative", "report", "write"],
    "verification": ["verify", "check", "validate", "confirm", "correct", "error"],
}


class TaskStateExtractor:
    def extract(
        self,
        task_description: str,
        csv_metadata: dict,
        prior_results: dict,
    ) -> TaskState:
        desc_lower = task_description.lower()

        # One-hot task_type via keyword matching (first match wins)
        task_type = [0.0] * len(TASK_TYPES)
        for i, ttype in enumerate(TASK_TYPES):
            if any(kw in desc_lower for kw in _TASK_KEYWORDS[ttype]):
                task_type[i] = 1.0
                break
        # Default to "text" (index 5) if nothing matched
        if sum(task_type) == 0.0:
            task_type[5] = 1.0

        # Temporal keyword detection
        has_temporal = any(kw in desc_lower for kw in TEMPORAL_KEYWORDS)

        # CSV size bucket (from schema_inspector output)
        shape = csv_metadata.get("shape", {})
        rows = shape.get("rows", 0)
        if rows < 1_000:
            size_bucket = 0
        elif rows < 100_000:
            size_bucket = 1
        else:
            size_bucket = 2

        # Column count bucket
        cols = shape.get("columns", 0)
        if cols < 5:
            col_bucket = 0
        elif cols <= 20:
            col_bucket = 1
        else:
            col_bucket = 2

        # Numeric target: any numeric column name appears in the task description
        dtypes = csv_metadata.get("dtypes", {})
        numeric_cols = [col for col, dt in dtypes.items() if "int" in dt or "float" in dt]
        has_numeric_target = any(col.lower() in desc_lower for col in numeric_cols)

        return TaskState(
            task_type=task_type,
            has_temporal_keyword=has_temporal,
            csv_size_bucket=size_bucket,
            num_columns_bucket=col_bucket,
            has_numeric_target=has_numeric_target,
            task_embedding=[0.0] * 384,  # hook for sentence-transformers
        )
