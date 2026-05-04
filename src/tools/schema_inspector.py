import time
import pandas as pd
from .base import ToolResult


def schema_inspector(path: str) -> ToolResult:
    start = time.time()
    try:
        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
        numeric_df = df.select_dtypes(include="number")
        schema = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "null_ratios": {col: round(float(df[col].isna().mean()), 3) for col in df.columns},
            "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
            "sample_rows": df.head(3).to_dict(orient="records"),
            "numeric_stats": numeric_df.describe().to_dict() if not numeric_df.empty else {},
        }
        return ToolResult(success=True, output=schema, latency_ms=(time.time() - start) * 1000)
    except Exception as e:
        return ToolResult(success=False, output=None, error=str(e), latency_ms=(time.time() - start) * 1000)
